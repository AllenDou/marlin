
#ifndef HHH
#define HHH

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>

template <typename T, int n>
struct Vec {
  T elems[n];
  __device__ T& operator[](int i) {
    return elems[i];
  }
};
using I4 = Vec<int, 4>;
using FragA = Vec<half2, 4>;
using FragB = Vec<half2, 2>;
using FragC = Vec<float, 4>;
using FragS = Vec<half2, 1>; // quantization scales

// Instruction for loading a full 16x16 matrix fragment of operand A from shared memory, directly in tensor core layout.
__device__ inline void ldsm4(FragA& frag_a, const void* smem_ptr) {
  uint32_t* a = reinterpret_cast<uint32_t*>(&frag_a);
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
    : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(smem)
  );
}


template <
  const int threads /*256*/, // number of threads in a threadblock
  const int thread_m_blocks /*4*/, // number of 16x16 blocks in the m dimension (batchsize) of the threadblock 
  const int thread_n_blocks /*16*/, // same for n dimension (output) 
  const int thread_k_blocks /*4*/, // same for k dimension (reduction)
  const int stages /*4*/, // number of stages for the async global->shared fetch pipeline
  const int group_blocks = -1 /*8*/ // number of consecutive 16x16 blocks with a separate quantization scale
>
__global__ void Marlin(
  int slice_iters
) {

  //int slice_iters = 64;
  //int stages = 4;
  int b_sh_wr_iters = 2;
  //int thread_m_blocks = 4;
  
  auto matmul = [&] (int k) {
    // We have the m dimension as the inner loop in order to encourage overlapping dequantization and matmul operations.
    #pragma unroll
    for (int j = 0; j < 4/*4 sub tile in a warp (4 warps/row)*/; j++) {
      // I4 frag_b_quant[2]; annotate by zixiao.
      //int b_quant = frag_b_quant[k % 2][j];
      //int b_quant_shift = b_quant >> 8;
      //FragB frag_b0 = dequant(b_quant);
      // If there are no groups, we can just scale the final output once and can avoid doing so for each weight.
      //if (group_blocks/*8*/ != -1)
      //  scale(frag_b0, frag_s[k % 2][j], 0);
      //FragB frag_b1 = dequant(b_quant_shift);
      //if (group_blocks/*8*/ != -1)
      //  scale(frag_b1, frag_s[k % 2][j], 1);
      #pragma unroll
      for (int i = 0; i < thread_m_blocks/*4*/; i++) {
        //mma(frag_a[k % 2][i], frag_b0, frag_c[i][j][0]);
        //mma(frag_a[k % 2][i], frag_b1, frag_c[i][j][1]);
        printf("call mma \n");
        printf("call mma \n");
      }
    }
  };

  while (slice_iters/* one iter for a tile */) {
    //printf("--------\n");
    // We unroll over both the global fetch and the register load pipeline to ensure all shared memory accesses are
    // static. Note that both pipelines have even length meaning that the next iteration will always start at index 0.
    //#pragma unroll
    for (int pipe = 0; pipe < stages/*4*/;) {
      //#pragma unroll
      for (int k = 0; k < b_sh_wr_iters/*2*/; k++) { // call 64 mma inst in total.
        //fetch_to_registers(k + 1, pipe % stages/*4*/);
        // k 的range是 0和1
        if (k == b_sh_wr_iters - 2 /*k=0*/) {
          //fetch_to_shared((pipe + stages/*4*/ - 1) % stages/*4*/, pipe, slice_iters >= stages/*4*/);
          pipe++;
          //wait_for_stage();
        }
        //!!! when k==1, no pipe++
        matmul(k);
      }
      //printf("call 32*2 mma, slice_iters=%d pipe=%d k=0,1\n", slice_iters, pipe);
      slice_iters--;
      if (slice_iters == 0)
        break;
    }
    if (slice_iters == 0) {
      break;
    }
  }
}

int main() {
  /*Vec<half2, 4>*/ //FragA frag_a[2][thread_m_blocks/*4*/];
  /*Vec<int, 4>*/   //I4 frag_b_quant[2];
  /*Vec<float, 4>*/ //FragC frag_c[thread_m_blocks/*4*/][4][2];
  /*Vec<half2, 1>*/ //FragS frag_s[2][4];

  //int data[] = {1, 2, 3, 4};
  //ldsm4(frag_a[0][0], data);

  const int THREADS = 256;
  const int STAGES = 4; // 4 pipeline stages fit into shared memory
  const int SHARED_MEM = 96 * 1024; // max shared memory on compute capability 8.6 (< 8.0)
  const int THREAD_M_BLOCKS = 4;
  const int THREAD_K_BLOCKS = 4;
  const int THREAD_N_BLOCKS = 16;
  const int GROUP_BLOCKS = 8;
  int blocks = 1;
  cudaStream_t stream = 0;
  cudaFuncSetAttribute( \
    Marlin<THREADS, THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS, STAGES, GROUP_BLOCKS>, \
    cudaFuncAttributeMaxDynamicSharedMemorySize, \
    SHARED_MEM \
  );
  Marlin<THREADS, THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS, STAGES, GROUP_BLOCKS>\
  <<<blocks, THREADS, SHARED_MEM, stream/*0*/>>>(1); 
  cudaDeviceSynchronize();

  return 0;
}
#endif