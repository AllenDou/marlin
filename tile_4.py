
a_gl_stride = 512
a_sh_stride = 8
a_gl_rd_delta_o = 8
thread_m_blocks = 4
thread_k_blocks = 4
thread_n_blocks = 16
b_gl_stride = 2048
b_sh_stride = 128
slice_col = 0
slice_row = 0
b_gl_rd_delta_o = 8192
s_gl_stride = 512
group_blocks = 8
s_sh_stride = 32

for threadIdx_x in range(256):
    a_gl_rd = a_gl_stride * int(threadIdx_x / a_gl_rd_delta_o) + (threadIdx_x % a_gl_rd_delta_o)
    a_gl_rd += a_gl_rd_delta_o * slice_row
    a_sh_wr = a_sh_stride * int(threadIdx_x / a_gl_rd_delta_o) + (threadIdx_x % a_gl_rd_delta_o)

    row = int(a_sh_wr / a_gl_rd_delta_o)
    a_sh_wr_trans = a_gl_rd_delta_o * row + (a_sh_wr % a_gl_rd_delta_o) ^ row

    a_sh_rd = a_sh_stride * ((threadIdx_x % 32) % 16) + int((threadIdx_x % 32) / 16)
    a_sh_rd += 2 * int(int(threadIdx_x / 32) / int(thread_n_blocks / 4))

    b_gl_rd = b_gl_stride * int(threadIdx_x / b_sh_stride) + (threadIdx_x % b_sh_stride)
    b_gl_rd += b_sh_stride * slice_col
    b_gl_rd += b_gl_rd_delta_o * slice_row
    b_sh_wr = threadIdx_x
    b_sh_rd = threadIdx_x

    s_gl_rd = s_gl_stride * int((thread_k_blocks * slice_row) / group_blocks) + s_sh_stride * slice_col + threadIdx_x
    s_sh_wr = threadIdx_x
    s_sh_rd = 8 * (int(threadIdx_x / 32) % int(thread_n_blocks / 4)) + int((threadIdx_x % 32) / 4)
    #s_sh_rd = 8 * (int(threadIdx_x / 32) % int(thread_n_blocks / 4)) + int((threadIdx_x % 32) % 4)
    print(f"{threadIdx_x=:3}  A:{a_gl_rd=:5} {a_sh_wr=:3} {a_sh_wr_trans=:3} {a_sh_rd=:3},  B:{b_gl_rd=:4} {b_sh_wr=:3} {b_sh_rd=:3},  S:{s_gl_rd=:3} {s_sh_wr=:3} {s_sh_rd=:2}")
