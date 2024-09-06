import numpy as np
np.set_printoptions(threshold=np.inf)

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
            print(f"{row=:2} {col=:2} {block=:1} {16*row + col + 8*block=:3}")
    for j in range(4):
        perm.extend([p + 256 * j for p in perm1])
        print(f"{[p + 256 * j for p in perm1]}")
perm = np.array(perm)
interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])
perm = perm.reshape((-1, 8))[:, interleave].ravel()
print(perm)
