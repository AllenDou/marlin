import os

M = 8
K = 8
N = 8
Mtile = 4
Ntile = 4

for m in range(0, M, Mtile):
    for n in range(0, N, Ntile):
        for k in range(0, K):
            for i in range(0, Mtile):
                for j in range(0, Ntile):
                    row = m + i
                    col = n + j
                    if m == 0 and n == 0:
                        print(f"C[{row}][{col}] += A[{row}][{k}] * B[{k}][{col}];")                    


#M = 8
#K = 8
#N = 8
#Mtile = 4
#Ktile = 2
#Ntile = 4
#
#for m in range(0, M, Mtile):
#    for n in range(0, N, Ntile):
#        for k in range(0, K):
#            for i in range(0, Mtile):
#                for j in range(0, Ntile):
#                    for kk in range(0, Ktile):
#                        row = m + i
#                        col = n + j
#                        if m == 0 and n == 0:
#                            print(f"C[{row}][{col}] += A[{row}][{k}] * B[{k}][{col}];")                    
