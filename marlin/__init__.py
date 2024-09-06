# Copyright (C) Marlin.2024 Elias Frantar (elias.frantar@ist.ac.at)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
import torch
import torch.nn as nn


import marlin_cuda

def mul(A, B, C, s, workspace, thread_k=-1, thread_n=-1, sms=-1, max_par=16, print_enable=True, user_specified_blockidx=0, user_specified_threadidx=0):
    """Marlin FP16xINT4 multiply; can be used within `torch.compile`.
    @A: `torch.half` input matrix of shape `(m, k)` in standard row-major layout
    @B: `torch.int` weight matrix of original shape `(k, n)` in Marlin format; see `Layer.pack()`
    @C: `torch.half` out matrix of shape `(m, n)` in standard row-major layout
    @s: `torch.half` scales of shape `(m / groupsize, n)`
    @workspace: `torch.int` tensor with at least `n / 128 * max_par` entries that are all zero
    @thread_k: `k` size of a thread_tile in `B` (can usually be left as auto -1)
    @thread_n: `n` size of a thread_tile in `B` (can usually be left as auto -1)
    @sms: number of SMs to use for the kernel (can usually be left as auto -1)
    @max_par: maximum number of batch 64 problems to solve in parallel for large input sizes
    """
    marlin_cuda.mul(A, B, C, s, workspace, thread_k, thread_n, sms, max_par, print_enable, user_specified_blockidx, user_specified_threadidx)


# Precompute permutations for Marlin weight and scale shuffling 

def _get_perms():
    perm = []
    for i in range(32):
        perm1 = []
        col = i // 4
        for block in [0, 1]:
            for row in [
                2 * (i % 4),
                2 * (i % 4) + 1,
                2 * (i % 4 + 4),
                2 * (i % 4 + 4) + 1
            ]:
                perm1.append(16 * row + col + 8 * block)
        for j in range(4):
            perm.extend([p + 256 * j for p in perm1])
    #import pdb; pdb.set_trace()
    perm = np.array(perm)
    interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])
    perm = perm.reshape((-1, 8))[:, interleave].ravel()
    perm = torch.from_numpy(perm)
    scale_perm = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single = []
    for i in range(4):
        scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return perm, scale_perm, scale_perm_single

_perm, _scale_perm, _scale_perm_single = _get_perms()

#_perm如下, 意思是, 一个线程要获取 4个数(for m16n8k16), 8个数(for m16n16k16), 这个4个数
# 需要分别对应tensorcore的一个线程的4个点, 见 https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-16816-float
# so, 如果要处理 m16n16k16, 那么一个线程就需要8个点, 如下的_perm数据, 就是 这8个点在真实Bweight的上下标
# 256 512 768的意思就是, 这个线程, 在连续4个B sub-tile上的需要的tensor core的8个点的下标, 这个4 也和cuda代码
# matmul的 for (int j = 0; j < 4/* */; j++) 对应. 这样 32个线程(1个warp), 就可以处理4个b subtile, 8个warp
# 就可以把Btile的一个iter处理完.
#tid=0
#[[   0,  128,    8,  136,   16,  144,   24,  152],
# [ 256,  384,  264,  392,  272,  400,  280,  408],
# [ 512,  640,  520,  648,  528,  656,  536,  664],
# [ 768,  896,  776,  904,  784,  912,  792,  920],
#tid=1
# [  32,  160,   40,  168,   48,  176,   56,  184],
# [ 288,  416,  296,  424,  304,  432,  312,  440],
# [ 544,  672,  552,  680,  560,  688,  568,  696],
# [ 800,  928,  808,  936,  816,  944,  824,  952],
# ...
#tid=31
#  [ 103,  231,  111,  239,  119,  247,  127,  255],
#  [ 359,  487,  367,  495,  375,  503,  383,  511],
#  [ 615,  743,  623,  751,  631,  759,  639,  767],
#  [ 871,  999,  879, 1007,  887, 1015,  895, 1023]])


class Layer(nn.Module):
    """PyTorch compatible Marlin layer; 4-bit (symmetric grouped) linear layer without bias."""

    def __init__(self, infeatures, outfeatures, groupsize=-1):
        """Create an empty Marlin layer.
        @infeatures: number of input features (must be divisible by 128)
        @outfeatures: number of output features (must be divisible by 256)
        @groupsize: quantization groupsize (must be -1 or 128)
        """
        super().__init__()
        if groupsize not in [-1, 128]:
            raise ValueError('Only groupsize -1 and 128 are supported.')
        if infeatures % 128 != 0 or outfeatures % 256 != 0:
            raise ValueError('`infeatures` must be divisible by 128 and `outfeatures` by 256.')
        if groupsize == -1:
            groupsize = infeatures
        if infeatures % groupsize != 0:
            raise ValueError('`infeatures` must be divisible by `groupsize`.')
        self.k = infeatures
        self.n = outfeatures
        self.groupsize = groupsize
        self.register_buffer('B', torch.empty((self.k // 16, self.n * 16 // 8), dtype=torch.int))
        self.register_buffer('s', torch.empty((self.k // groupsize, self.n), dtype=torch.half))
        # 128 is currently the minimum `tile_n`, hence it gives the maximum workspace size; 16 is the default `max_par`
        self.register_buffer('workspace', torch.zeros(self.n // 128 * 16, dtype=torch.int), persistent=False)

    def forward(self, A):
        C = torch.empty(A.shape[:-1] + (self.s.shape[1],), dtype=A.dtype, device=A.device)
        mul(A.view((-1, A.shape[-1])), self.B, C.view((-1, C.shape[-1])), self.s, self.workspace)
        return C

    def pack(self, linear, scales):
        """Pack a fake-quantized linear layer into this actual Marlin representation.
        @linear: fake-quantized `torch.nn.Linear` layer to convert (must be of type `torch.half`)
        @scales: corresponding quantization scales of shape `(infeatures, groups)`
        """ 
        if linear.weight.dtype != torch.half:
            raise ValueError('Only `torch.half` weights are supported.')
        tile = 16
        maxq = 2 ** 4 - 1 # = 15
        s = scales.t()
        w = linear.weight.data.t()
        if self.groupsize != self.k:
            w = w.reshape((-1, self.groupsize, self.n))
            w = w.permute(1, 0, 2)
            w = w.reshape((self.groupsize, -1))
            s = s.reshape((1, -1))
        w = torch.round(w / s).int()
        w += (maxq + 1) // 2 # make sign number to unsighned 0-15, 4bits
        w = torch.clamp(w, 0, maxq)
        if self.groupsize != self.k:
            w = w.reshape((self.groupsize, -1, self.n))
            w = w.permute(1, 0, 2)
            w = w.reshape((self.k, self.n)).contiguous()
            s = s.reshape((-1, len(_scale_perm)))[:, _scale_perm]
        else:
            s = s.reshape((-1, len(_scale_perm_single)))[:, _scale_perm_single]
        s = s.reshape((-1, self.n)).contiguous()
        #import pdb; pdb.set_trace()
        w = w.reshape((self.k // tile, tile, self.n // tile, tile))
        w = w.permute((0, 2, 1, 3))
        w = w.reshape((self.k // tile, self.n * tile))
        res = w
        res = res.reshape((-1, _perm.numel()))[:, _perm].reshape(res.shape)
        #import pdb; pdb.set_trace()
        qq = np.zeros((res.shape[0], res.shape[1] // 8), dtype=np.uint32)
        res = res.cpu().numpy().astype(np.uint32)
        #  p res[0][0:8] = [ 5,  5,  5,  1, 10,  2,  8,  5] 8个数 每个占用4bit, 合并起来一个int32 1479152981
        for i in range(8):
            qq |= res[:, i::8] << 4 * i
        #import pdb; pdb.set_trace()
        qq = torch.from_numpy(qq.astype(np.int32)).to(w.device)
        self.B[:, :] = qq.to(self.B.device)
        self.s[:, :] = s.to(self.s.device) #0.3533


def replace_linear(module, name_filter=lambda n: True, groupsize=-1, name=''):
    """Recursively replace all `torch.nn.Linear` layers by empty Marlin layers.
    @module: top-level module in which to perform the replacement 
    @name_filter: lambda indicating if a layer should be replaced
    @groupsize: marlin groupsize
    @name: root-level name
    """
    if isinstance(module, Layer):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        name1 = name + '.' + attr if name != '' else attr
        if isinstance(tmp, nn.Linear) and name_filter(name1):
            setattr(
                module, attr, Layer(tmp.in_features, tmp.out_features, groupsize=groupsize)
            )
    for name1, child in module.named_children():
        replace_linear(child, name_filter, groupsize=groupsize, name=name + '.' + name1 if name != '' else name1)
