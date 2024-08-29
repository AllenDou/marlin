

# !!!
# !!! this implement is not needed, refer https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/ 
# !!!
if True:
    M = 8
    K = 8
    N = 8
    Mtile = 4
    Ktile = 2
    Ntile = 4

    for m in range(0, M, Mtile):
        for n in range(0, N, Ntile):
            for k in range(0, K, Ktile):
                for i in range(0, Mtile, Ktile):
                    for j in range(0, Ntile, Ktile):
                        for kk in range(0, Ktile):
                            for kkk in range(0, Ktile):
                                row = m + i + kk 
                                col = n + j + kkk
                                #if m == 0 and n == 0:
                                #if row == 8 or col == 8:
                                #    import pdb; pdb.set_trace()
                                print(f"C[{row}][{col}] += A[{row}][{k+kk + kkk}] * B[{k+kk+kkk}][{col}];")
