
stages = 4
slice_iters = 64
b_sh_wr_iters = 2

def start_pipes() :
    for i in range(0, stages - 1):
        print(f"fetch_to_shared pipe={i} off={i}")
    print("fetch_to_register iter=0 pipe=0")

start_pipes()

while (slice_iters):
    print("")
    for pipe in range(0, stages):
        print(f"{pipe=}")
        for k in range(0, b_sh_wr_iters):
            print(f"fetch_to_register next_iter={(k+1) % b_sh_wr_iters} pipe={pipe % stages}")
            if k == b_sh_wr_iters - 2:
                print(f"fetch_to_shared next_pipe={(pipe + stages - 1) % stages} off={pipe}")
                pipe += 1
                #print(f"wait_for_stages")
            print(f"{slice_iters=} matmul(iter={k})")
        slice_iters -= 1 
        if slice_iters == 0:
            break  
