thread_m_blocks = 4
thread_n_blocks = 16
prob_n = 4096
prob_m = 64
# when global_reduce, slice_col slice_row should not be 0/0
slice_col = 0
slice_row = 0


threadIdx_x=0
first = True
last = False

active_threads = int(32 * thread_n_blocks / 4)

#for threadIdx_x in range(active_threads):
if threadIdx_x < active_threads:
  c_gl_stride = prob_n / 8;
  c_gl_wr_delta_o = 8 * c_gl_stride
  c_gl_wr_delta_i = 4 * int(active_threads / 32)
  c_gl_wr = c_gl_stride * int((threadIdx_x % 32) / 4) + 4 * int(threadIdx_x / 32) + threadIdx_x % 4
  c_gl_wr += (2 * thread_n_blocks) * slice_col
  c_sh_wr_delta = active_threads
  c_sh_wr = threadIdx_x

  row = int((threadIdx_x % 32) / 4)

  if not first:
    for i in range(thread_m_blocks*4):
        print(f"sh[{c_sh_wr + c_sh_wr_delta * i }] C[{c_gl_wr + c_gl_wr_delta_o*int(i/2) + c_gl_wr_delta_i * (i % 2)}] {i<(thread_m_blocks - 1)*4 or 8 * int(i / 2) + row < prob_m}")

  for i in range(thread_m_blocks*4):
    if i < (thread_m_blocks - 1) * 4 or 8 * int(i / 2) + row < prob_m :
      if not first:
        #int4 c_red = sh[c_sh_wr + i * c_sh_wr_delta];
        print(f"c_red = sh[{c_sh_wr + i * c_sh_wr_delta}]")
        for j in range(4*2):
          print(f"frag_c[{4 * 2 * 4 * int(i / 4) + 4 * j + (i % 4)}] += c_red[{j}]")

      if not last:
        for j in range(4*2):
          print(f"c[{j}] = frag_c[{4 * 2 * 4 * int(i / 4) + 4 * j + (i % 4)=}]")
        print(f"{threadIdx_x=:3} {i=:2} C[{c_gl_wr+c_gl_wr_delta_o*int(i/2)+c_gl_wr_delta_i*(i%2)}] = c(int4)")
