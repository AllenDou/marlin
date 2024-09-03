
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

int slice_iters = 64;
int stages = 4;
int b_sh_wr_iters = 2;
int thread_m_blocks = 4;

// Instruction for loading a full 16x16 matrix fragment of operand A from shared memory, directly in tensor core layout.
__device__ inline void ldsm4(FragA& frag_a, const void* smem_ptr) {
  uint32_t* a = reinterpret_cast<uint32_t*>(&frag_a);
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
    : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(smem)
  );
}

int main() {
  /*Vec<half2, 4>*/ FragA frag_a[2][thread_m_blocks/*4*/];
  /*Vec<int, 4>*/   I4 frag_b_quant[2];
  /*Vec<float, 4>*/ FragC frag_c[thread_m_blocks/*4*/][4][2];
  /*Vec<half2, 1>*/ FragS frag_s[2][4];

  //int data[] = {1, 2, 3, 4};
  //ldsm4(frag_a[0][0], data);

  return 0;

  while (slice_iters/* one iter for a tile */) {
    // We unroll over both the global fetch and the register load pipeline to ensure all shared memory accesses are
    // static. Note that both pipelines have even length meaning that the next iteration will always start at index 0.
    #pragma unroll
    for (int pipe = 0; pipe < stages/*4*/;) {
      #pragma unroll
      for (int k = 0; k < b_sh_wr_iters/*2*/; k++) { // call 64 mma inst in total.
        //fetch_to_registers(k + 1, pipe % stages/*4*/);
        // k 的range是 0和1
        if (k == b_sh_wr_iters - 2 /*k=0*/) {
          //fetch_to_shared((pipe + stages/*4*/ - 1) % stages/*4*/, pipe, slice_iters >= stages/*4*/);
          pipe++;
          //wait_for_stage();
        }
        //!!! when k==1, no pipe++
        //matmul(k);
        printf("matmul 32 mma. pipe=%d k=%d\n", pipe, k);
      }
      slice_iters--;
      if (slice_iters == 0)
        break;
    }
    printf("----\n");
  }
}
