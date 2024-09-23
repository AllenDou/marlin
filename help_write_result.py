threads = 256
thread_m_blocks = 4
thread_n_blocks = 16
threadIdx_x = 0

prob_n = 4096
slice_col = 0
slice_row = 0
prob_m = 64

c_gl_stride = int(prob_n/8)
c_sh_stride = 2 * thread_n_blocks + 1
c_gl_wr_delta = c_gl_stride * int(threads/(2*thread_n_blocks))
c_sh_rd_delta = c_sh_stride * int(threads / (2*thread_n_blocks))

c_gl_wr = c_gl_stride * int(threadIdx_x/(2*thread_n_blocks)) + (threadIdx_x % (2*thread_n_blocks))
c_gl_wr += (2*thread_n_blocks) * slice_col
c_sh_wr = (4*c_sh_stride) * int((threadIdx_x % 4) / 4) + (threadIdx_x % 32) % 4
c_sh_wr += 32*int(threadIdx_x/32)
c_sh_rd = c_sh_stride * int(threadIdx_x/(2*thread_n_blocks)) + (threadIdx_x % (2*thread_n_blocks))
c_gl_wr_end = c_gl_stride * prob_m

if threadIdx_x / 32 < thread_n_blocks / 4:
    for i in range(thread_m_blocks):
        for j in range(4):
            #import pdb; pdb.set_trace()
            wr = c_sh_wr + 8*j
            print(f"{int(c_sh_wr)=}")
            print(f"{int(wr+(4*c_sh_stride) * 0 + 0)=:4}, frag_c[{i}][{j}][{0}][{0}], frag_c[{i}][{j}][{0}][{1}] frag_s[{int(j/2)}][{2*(j%2)+0}]")
            print(f"{int(wr+(4*c_sh_stride) * 8 + 0)=:4}, frag_c[{i}][{j}][{0}][{2}], frag_c[{i}][{j}][{0}][{3}] frag_s[{int(j/2)}][{2*(j%2)+0}]")
            print(f"{int(wr+(4*c_sh_stride) * 0 + 4)=:4}, frag_c[{i}][{j}][{1}][{0}], frag_c[{i}][{j}][{1}][{1}] frag_s[{int(j/2)}][{2*(j%2)+1}]")
            print(f"{int(wr+(4*c_sh_stride) * 8 + 4)=:4}, frag_c[{i}][{j}][{1}][{2}], frag_c[{i}][{j}][{1}][{3}] frag_s[{int(j/2)}][{2*(j%2)+1}]")
        c_sh_wr += 16 * (4*c_sh_stride)

print("")
for i in range(int(16*thread_m_blocks/(threads/(2*thread_n_blocks)))):
  if c_gl_wr < c_gl_wr_end:
    print(f"C[{int(c_gl_wr)}]=sh[{int(c_sh_rd)}]")
    c_gl_wr += c_gl_wr_delta
    c_sh_rd += c_sh_rd_delta
