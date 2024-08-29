M = 8
K = 8
N = 8
Mtile = 4
Ntile = 4

for m in range(0, M, Mtile):
    for n in range(0, N, Ntile):
        # Now we know m & n, so we can load Mtile & Ntile from global memory to shared memory
        for k in range(0, K):
            # k loop should be outside of Mtile & Ntile,
            # cause, Mtile & Ntile will be load in-chip.
            print(f"{k=}")
            for i in range(0, Mtile):
                for j in range(0, Ntile):
                    # Now we know i & j, so we can load sub-tile (from index k)
                    # from shared memory into register
                    row = m + i
                    col = n + j
                    #if m == 0 and n == 0:
                    print(f"C[{row}][{col}] += A[{row}][{k}] (mma) B[{k}][{col}];")