if True:
    M = 8
    K = 8
    N = 8
    Mtile = 4
    Ntile = 4

    for m in range(0, M, Mtile):
        for n in range(0, N, Ntile):
            for k in range(0, K):
                # k loop should be outside of Mtile & Ntile,
                # cause, Mtile & Ntile wile be load in-chip.
                print(f"{k=}")
                for i in range(0, Mtile):
                    for j in range(0, Ntile):
                        row = m + i
                        col = n + j
                        #if m == 0 and n == 0:
                        print(f"C[{row}][{col}] += A[{row}][{k}] * B[{k}][{col}];")                    
