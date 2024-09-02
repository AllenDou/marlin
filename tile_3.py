
thread_m_blocks = 4
stages = 4
b_sh_wr_iters = 2


def matmul(k):
    for j in range(4):
        for i in range(thread_m_blocks):
            print("mma_%d %d %d %d" % (k, i, j, 0))
            print("mma_%d %d %d %d" % (k, i, j, 1))

pipe = 0
while pipe < stages:
    for k in range(b_sh_wr_iters):
        print(f"{k=} fetching to register")
        if k == b_sh_wr_iters-2:
            print(f"{k=} fetching to shared")
            pipe += 1
        print(f"{pipe=} {k=}")
    print()

