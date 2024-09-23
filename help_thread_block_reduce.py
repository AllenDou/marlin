threads = 256
b_sh_stride = 128
thread_m_blocks = 4
threadIdx_x = 255
red_off = threads / b_sh_stride / 2  # 1

print(f"{threadIdx_x=}")
#import pdb; pdb.set_trace()
if (red_off >= 1):
  red_idx = int(threadIdx_x / b_sh_stride) # 0/1
  red_sh_stride = b_sh_stride * 4 * 2
  red_sh_delta = b_sh_stride 
  red_sh_rd = red_sh_stride * int(threadIdx_x / b_sh_stride) + int(threadIdx_x % b_sh_stride)

  for m_block in range(0, thread_m_blocks):
    i = red_off
    while i>0:
      if i <= red_idx and red_idx < 2 * i :
        for j in range(0, 4*2):
          red_sh_wr = int(red_sh_delta * j + (red_sh_rd - red_sh_stride * i))
          #sh[red_sh_wr] = reinterpret_cast<int4*>(&frag_c)[4 * 2 * m_block + j];
          print(f"{m_block=} {red_sh_rd=:3} {red_sh_wr=:3} <- frag_c[{4*2*m_block+j}]")
      i = int(i / 2)

    if red_idx == 0 :
      for i in range(0, 4*2):
        #float* c_rd = reinterpret_cast<float*>(&sh[red_sh_delta * i + red_sh_rd]);
        print(f"{m_block=} {red_sh_rd=:3} {red_sh_delta*i + red_sh_rd=:3} -> frag_c[{4*2*m_block+i}]")
        for j in range(4):
          #reinterpret_cast<FragC*>(frag_c)[4 * 2 * m_block + i][j] += c_rd[j];
          #print(f"frag_c[{4*2*m_block+i}][{j}]")
          pass
    
