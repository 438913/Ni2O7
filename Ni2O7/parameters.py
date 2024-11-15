# Mc表示O原子离Ni原子的曼哈顿距离不超过Mc
# 即|xO - xNi| + |yO - yNi| < = Mc
Mc = 1
holes = 6
Ni_position = [(-1, 0), (1, 0)]
# 保证Mc是奇数
Mc = Mc + Mc % 2 - 1
# 如果Sz_set中包含'All_states'，则包含所有自旋的情况
Sz_set = [0]
A = 6.0
B = 0.15
C = 0.58
Upp = 4.0
ed = {'d3z2r2': 0.046,
      'dx2y2': 0.0,
      'dxy': 0.823,
      'dxz': 0.706,
      'dyz': 0.706}
ep = 2.47
tpd = 1.38
tpp = 0.537
Norb = 4
if Norb == 4:
    O1_orbs = ['px']
    O2_orbs = ['py']
    O_orbs = O1_orbs + O2_orbs
    Ni_orbs = ['d3z2r2', 'dx2y2']
num_vals = 10
