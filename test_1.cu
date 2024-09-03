
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
int slice_iters = 1;
int stages = 4;
int b_sh_wr_iters = 2;

int main() {

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
        printf("matmul 32 mma.\n");
      }
      slice_iters--;
      if (slice_iters == 0)
        break;
    }
  }
}