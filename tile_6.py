
stages = 4
slice_iters = 64
b_sh_wr_iters = 2

def start_pipes() :
    for i in range(0, stages - 1):
        print(f"fetch_to_shared(commit group) pipe={i} off={i} B_ptr/s_gl_rd incr")
    print("wait_for_stages(wait group 2)")
    print("fetch_to_register iter=0 pipe=0")
    print("a_gl_rd incr")

start_pipes()

while (slice_iters):
    print("")
    for pipe in range(0, stages):
        print(f"{pipe=}")
        for k in range(0, b_sh_wr_iters):
            print(f"iter={k}")
            print(f"fetch_to_register next_iter={(k+1) % b_sh_wr_iters} pipe={pipe % stages}")
            if k == b_sh_wr_iters - 2:
                print(f"fetch_to_shared(commit group) next_pipe={(pipe + stages - 1) % stages} off={pipe} B_ptr/s_gl_rd incr")
                pipe += 1
                print(f"wait_for_stages(wait group 2)")
            print(f"{slice_iters=} matmul(iter={k})")
        slice_iters -= 1 
        if slice_iters == 0:
            break  
    print("a_gl_rd incr")
    if slice_iters == 0:
        print("")
        print("slice_iters=0")
        print(f"wait_for_stages(wait group all)")
