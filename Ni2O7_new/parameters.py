# Mc表示O原子离Ni原子的曼哈顿距离不超过Mc
# 即|xO - xNi| + |yO - yNi| < = Mc
Mc = 1
holes = 4
Ni_position = [(-1, 0), (1, 0)]
# 保证Mc是奇数
Mc = Mc + Mc % 2 - 1
# 如果Sz_set中包含'All_states'，则包含所有自旋的情况
Sz_set = [0]
A = 6.0
B = 0.15
C = 0.58
Upp = 4.0
pressures = [0, 4, 8, 16, 29.5]
eds = {'d3z2r2': [0.046, 0.054, 0.060, 0.072, 0.095],
       'dx2y2': [0.0 for _ in range(5)],
       'dxy': [0.823, 0.879, 0.920, 0.997, 1.06],
       'dxz': [0.706, 0.761, 0.804, 0.887, 0.94],
       'dyz': [0.706, 0.761, 0.804, 0.887, 0.94]}
eps = [2.47, 2.56, 2.62, 2.75, 2.9]
tpds = [1.38, 1.43, 1.46, 1.52, 1.58]
tpps = [0.537, 0.548, 0.554, 0.566, 0.562]
Norb = 4
if Norb == 4:
    O1_orbs = ['px']
    O2_orbs = ['py']
    O_orbs = O1_orbs + O2_orbs
    Ni_orbs = ['d3z2r2', 'dx2y2']
num_vals = 10
