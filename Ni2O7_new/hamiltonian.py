import numpy as np
from itertools import combinations, combinations_with_replacement
import scipy.sparse as sp

import parameters as pam
from variational_space import check_in_vs_condition, count_inversions

holes = pam.holes
O1_orbs = pam.O1_orbs
O2_orbs = pam.O2_orbs
O_orbs = pam.O_orbs
Ni_orbs = pam.Ni_orbs
Ni_position = pam.Ni_position
directions_to_vecs = {'L': (-1, 0), 'R': (1, 0), 'U': (0, 1), 'D': (0, -1),
                      'UL': (-1, 1), 'UR': (1, 1), 'DL': (-1, -1), 'DR': (1, -1)}
tpp_nn_hop_dir = ['UL', 'UR', 'DL', 'DR']


def set_tpd_tpp(Norb, tpd, tpp):
    """
    设置tpd和tpp的相关参数
    :param Norb: hopping的轨道数
    :param tpd: p,d轨道之间的hopping积分
    :param tpp: p,p轨道之间的hopping积分
    :return: tpd_nn_hop_dir: p,d轨道之间的hopping方向, 字典, key为轨道, value为方向列表
                tpd_orbs: p,d轨道集合
                tpd_nn_hop_fac: p,d轨道之间的hopping因子, 字典, key为(轨道1, 方向1, 轨道2), value为因子
                tpp_nn_hop_fac: p,p轨道之间的hopping因子, 字典, key为(方向1, 轨道1, 轨道2), value为因子
    """
    tpd_nn_hop_dir = {}
    tpd_orbs = {}
    tpd_nn_hop_fac = {}
    tpp_nn_hop_fac = {}
    if Norb == 4:
        tpd_nn_hop_dir = \
            {'d3z2r2': ['L', 'R', 'U', 'D'],
             'dx2y2': ['L', 'R', 'U', 'D']}
        tpd_orbs = {'d3z2r2', 'dx2y2'}
        tpd_nn_hop_fac = \
            {('d3z2r2', 'L', 'px'): -tpd / np.sqrt(3),
             ('d3z2r2', 'R', 'px'): tpd / np.sqrt(3),
             ('d3z2r2', 'U', 'py'): tpd / np.sqrt(3),
             ('d3z2r2', 'D', 'py'): -tpd / np.sqrt(3),
             ('dx2y2', 'L', 'px'): tpd,
             ('dx2y2', 'R', 'px'): -tpd,
             ('dx2y2', 'U', 'py'): tpd,
             ('dx2y2', 'D', 'py'): -tpd}
        tpp_nn_hop_fac = \
            {('UR', 'px', 'py'): -tpp,
             ('UL', 'px', 'py'): tpp,
             ('DL', 'px', 'py'): -tpp,
             ('DR', 'px', 'py'): tpp}
    return tpd_nn_hop_dir, tpd_orbs, tpd_nn_hop_fac, tpp_nn_hop_fac


def get_interaction_mat(A, sym):
    """
    根据A, sym得到interaction_mat
    :param A:
    :param sym: 对称类型
    :return: Stot, Sz_set: 总自旋量子数，z方向的自旋量子数集合
                A or B_sym: 1表示A对称，-1表示B对称，0表示不对称
                state_order: 状态序号，字典，key为状态，value为序号
                interaction_mat: 相互作用矩阵(列表)，表示两个状态之间的相互作用
    """
    B = pam.B
    C = pam.C
    Stot = 0
    Sz_set = []
    AorB_sym = 0
    state_order = {}
    interaction_mat = {}
    if sym == '1A1':
        Stot = 0
        Sz_set = [0]
        AorB_sym = 1
        fac = np.sqrt(2)
        state_order = \
            {('d3z2r2', 'd3z2r2'): 0,
             ('dx2y2', 'dx2y2'): 1,
             ('dxy', 'dxy'): 2,
             ('dxz', 'dxz'): 3,
             ('dyz', 'dyz'): 3}
        interaction_mat = \
            [[A + 4. * B + 3. * C, 4. * B + C, 4. * B + C, fac * (B + C)],
             [4. * B + C, A + 4. * B + 3. * C, C, fac * (3. * B + C)],
             [4. * B + C, C, A + 4. * B + 3. * C, fac * (3. * B + C)],
             [fac * (B + C), fac * (3. * B + C), fac * (3. * B + C), A + 7. * B + 4. * C]]
    if sym == '1B1':
        Stot = 0
        Sz_set = [0]
        AorB_sym = -1
        fac = np.sqrt(3)
        state_order = \
            {('d3z2r2', 'dx2y2'): 0,
             ('dxz', 'dxz'): 1,
             ('dyz', 'dyz'): 1}
        interaction_mat = [[A + 2. * C, 2. * B * fac],
                           [2. * B * fac, A + B + 2. * C]]
    if sym == '1A2':
        Stot = 0
        Sz_set = [0]
        AorB_sym = 0
        state_order = {('dx2y2', 'dxy'): 0}
        interaction_mat = [[A + 4. * B + 2. * C]]
    if sym == '3A2':
        Stot = 1
        Sz_set = [-1, 0, 1]
        AorB_sym = 0
        state_order = {('dx2y2', 'dxy'): 0,
                       ('dxz', 'dyz'): 1}
        interaction_mat = [[A + 4. * B, 6. * B],
                           [6. * B, A - 5. * B]]
    if sym == '3B1':
        Stot = 1
        Sz_set = [-1, 0, 1]
        AorB_sym = 0
        state_order = {('d3z2r2', 'dx2y2'): 0}
        interaction_mat = [[A - 8. * B]]
    if sym == '1B2':
        Stot = 0
        Sz_set = [0]
        AorB_sym = 0
        fac = np.sqrt(3)
        state_order = {('d3z2r2', 'dxy'): 0,
                       ('dxz', 'dyz'): 1}
        interaction_mat = [[A + 2. * C, 2. * B * fac],
                           [2. * B * fac, A + B + 2. * C]]
    if sym == '3B2':
        Stot = 1
        Sz_set = [-1, 0, 1]
        AorB_sym = 0
        state_order = {('d3z2r2', 'dxy'): 0}
        interaction_mat = [[A - 8. * B]]
    if sym == '1E':
        Stot = 0
        Sz_set = [0]
        AorB_sym = 0
        fac = np.sqrt(3)
        state_order = \
            {('d3z2r2', 'dxz'): 0,
             ('d3z2r2', 'dyz'): 1,
             ('dx2y2', 'dxz'): 2,
             ('dx2y2', 'dyz'): 3,
             ('dxy', 'dxz'): 4,
             ('dxy', 'dyz'): 5}
        interaction_mat = \
            [[A + 3. * B + 2. * C, 0, -B * fac, 0, 0, -B * fac],
             [0, A + 3. * B + 2. * C, 0, B * fac, -B * fac, 0],
             [-B * fac, 0, A + B + 2. * C, 0, 0, -3. * B],
             [0, B * fac, 0, A + B + 2. * C, 3. * B, 0],
             [0, -B * fac, 0, 3. * B, A + B + 2. * C, 0],
             [-B * fac, 0, -3. * B, 0, 0, A + B + 2. * C]]
    if sym == '3E':
        Stot = 1
        Sz_set = [-1, 0, 1]
        AorB_sym = 0
        fac = np.sqrt(3)
        state_order = \
            {('d3z2r2', 'dxz'): 0,
             ('d3z2r2', 'dyz'): 1,
             ('dx2y2', 'dxz'): 2,
             ('dx2y2', 'dyz'): 3,
             ('dxy', 'dxz'): 4,
             ('dxy', 'dyz'): 5}
        interaction_mat = \
            [[A + B, 0, -3. * B * fac, 0, 0, -3. * B * fac],
             [0, A + B, 0, 3. * B * fac, -3. * B * fac, 0],
             [-3. * B * fac, 0, A - 5. * B, 0, 0, 3. * B],
             [0, 3. * B * fac, 0, A - 5. * B, -3. * B, 0],
             [0, -3. * B * fac, 0, -3. * B, A - 5. * B, 0],
             [-3. * B * fac, 0, 3. * B, 0, 0, A - 5. * B]]
    return Stot, Sz_set, AorB_sym, state_order, interaction_mat


def find_hopping_index(vs, pre_state, tmp_state, state_other):
    """
    查找hopping的索引
    :param vs:
    :param pre_state:
    :param tmp_state:
    :param state_other:
    :return:
    """
    pre_state_all = (pre_state,) + state_other
    tmp_state_all = (tmp_state,) + state_other
    diff_inversions = count_inversions(pre_state_all) - count_inversions(tmp_state_all)
    if diff_inversions % 2 == 0:
        phase = 1
    else:
        phase = -1
    pre_state_all = tuple(sorted(pre_state_all))
    tmp_state_all = tuple(sorted(tmp_state_all))
    row = vs.get_index(pre_state_all)
    col = vs.get_index(tmp_state_all)
    return row, col, phase


def create_tpd_nn_matrix(vs, tpd_nn_hop_dir, tpd_nn_hop_fac):
    """
    创建tpd轨道之间的hopping矩阵tpd_nn_matrix,
    :param vs:
    :param tpd_nn_hop_dir:
    :param tpd_nn_hop_fac:
    :return:
    """
    print("start create_tpd_nn_matrix")
    print("==========================")
    Ni_states = vs.Ni_states
    states_one = vs.states_1
    dim = vs.dim
    rows = []
    cols = []
    data = []
    for pre_state in Ni_states:
        x1, y1, orb1, s1 = pre_state
        for direction in tpd_nn_hop_dir[orb1]:
            vx, vy = directions_to_vecs[direction]
            x2, y2 = x1 + vx, y1 + vy
            if y2 % 2 == 0:
                O_orbs2 = O1_orbs
            else:
                O_orbs2 = O2_orbs
            for orb2 in O_orbs2:
                orb12 = (orb1, direction, orb2)
                if orb12 not in tpd_nn_hop_fac.keys():
                    continue
                value = tpd_nn_hop_fac[orb12]
                tmp_state = (x2, y2, orb2, s1)
                filter_state = filter(lambda x: x != pre_state and x != tmp_state, states_one)
                states_other = combinations(filter_state, holes - 1)
                for state_other in states_other:
                    row, col, ph = find_hopping_index(vs, pre_state, tmp_state, state_other)
                    if row is not None and col is not None:
                        rows += [row, col]
                        cols += [col, row]
                        data += [value * ph, value * ph]
    return sp.coo_matrix((data, (rows, cols)), shape=(dim, dim))


def create_tpp_nn_matrix(vs, tpp_nn_hop_fac):
    """
    创建tpp轨道之间的hopping矩阵tpp_nn_matrix,
    :param vs:
    :param tpp_nn_hop_fac:
    :return:
    """
    print("start create_tpp_nn_matrix")
    print("==========================")
    O_states = vs.O_states
    states_one = vs.states_1
    dim = vs.dim
    rows = []
    cols = []
    data = []
    for pre_state in O_states:
        x1, y1, orb1, s1 = pre_state
        for direction in tpp_nn_hop_dir:
            vx, vy = directions_to_vecs[direction]
            x2, y2 = x1 + vx, y1 + vy
            if not check_in_vs_condition(x2, y2):
                continue
            for orb2 in O_orbs:
                orb12 = [direction, orb1, orb2]
                orb12 = tuple(sorted(orb12))
                if orb12 not in tpp_nn_hop_fac.keys():
                    continue
                value = tpp_nn_hop_fac[orb12]
                tmp_state = (x2, y2, orb2, s1)
                filter_state = filter(lambda x: x != pre_state and x != tmp_state, states_one)
                states_other = combinations(filter_state, holes - 1)
                for state_other in states_other:
                    row, col, ph = find_hopping_index(vs, pre_state, tmp_state, state_other)
                    if row is not None and col is not None:
                        rows.append(row)
                        cols.append(col)
                        data.append(value * ph)
    return sp.coo_matrix((data, (rows, cols)), shape=(dim, dim))


def create_Esite_matrix(vs, A, ed, ep):
    """
    创建Onsite矩阵
    :param vs:
    :param A:
    :param ed:
    :param ep:
    :return:
    """
    print("start create_Esite_matrix")
    print("==========================")
    dim = vs.dim
    lookup_tbl = vs.lookup_tbl
    rows = []
    cols = []
    data = []
    for i in range(dim):
        state = lookup_tbl[i]
        diag_el = 0
        Ni1 = 0
        Ni2 = 0
        for x, y, orb, _ in state:
            if orb in Ni_orbs:
                diag_el += ed[orb]
            else:
                diag_el += ep
            if (x, y) == Ni_position[0]:
                Ni1 += 1
            if (x, y) == Ni_position[1]:
                Ni2 += 1
        if Ni1 != 2:
            diag_el += A + abs(Ni1 - 2) * A / 2.0
        if Ni2 != 2:
            diag_el += A + abs(Ni2 - 2) * A / 2.0
        data.append(diag_el)
        rows.append(i)
        cols.append(i)
    return sp.coo_matrix((data, (rows, cols)), shape=(dim, dim))


def create_interaction_matrix_d(vs, d_double, d_part, S_val, Sz_val, AorB_sym, A):
    """
    创建相互作用矩阵
    :param vs:
    :param d_double:
    :param d_part:
    :param S_val:
    :param Sz_val:
    :param AorB_sym:
    :param A:
    :return:
    """
    print("start create_interaction_matrix_d")
    print("==========================")
    dim = vs.dim
    data = []
    row = []
    col = []
    exist_orb34 = combinations_with_replacement(Ni_orbs, 2)
    exist_orb34 = list(exist_orb34)
    channels = ['1A1', '1A2', '3A2', '1B1', '3B1', '1E', '3E', '1B2', '3B2']
    for sym in channels:
        Stot, Sz_set, AorB, state_order, interaction_mat = get_interaction_mat(A, sym)
        sym_orbs = state_order.keys()
        for part_idx, tbl_idx in enumerate(d_double):
            count = []
            states = vs.lookup_tbl[tbl_idx]
            hole_idx1, hole_idx2 = d_part[part_idx]
            orb1, orb2 = states[hole_idx1][-2], states[hole_idx2][-2]
            orb1, orb2 = sorted([orb1, orb2])
            orb12 = (orb1, orb2)
            S12 = S_val[tbl_idx]
            Sz12 = Sz_val[tbl_idx]
            if orb12 not in sym_orbs or S12 != Stot or Sz12 not in Sz_set:
                continue
            if orb1 == orb2 == 'dxz' or orb1 == orb2 == 'dyz':
                if AorB_sym[tbl_idx] != AorB:
                    continue
            mat_idx1 = state_order[orb12]
            for mat_idx2, orb34 in enumerate(sym_orbs):
                if orb34 == ('dyz', 'dyz'):
                    mat_idx2 -= 1
                if orb34 not in exist_orb34:
                    continue
                for s1 in ['up', 'dn']:
                    for s2 in ['up', 'dn']:
                        hole1 = states[hole_idx1][: 2] + (orb34[0], s1)
                        hole2 = states[hole_idx2][: 2] + (orb34[1], s2)
                        if hole1 == hole2:
                            continue
                        inter_states = list(states)
                        inter_states[hole_idx1], inter_states[hole_idx2] = hole1, hole2
                        inter_states = sorted(inter_states)
                        inter_states = tuple(inter_states)
                        inter_idx = vs.get_index(inter_states)
                        if inter_idx is None or inter_idx in count:
                            continue
                        S34, Sz34 = S_val[inter_idx], Sz_val[inter_idx]
                        if S34 != S12 or Sz34 != Sz12:
                            continue
                        if orb34 == ('dxz', 'dxz') or orb34 == ('dyz', 'dyz'):
                            if AorB_sym[inter_idx] != AorB:
                                continue
                        value = interaction_mat[mat_idx1][mat_idx2]
                        data.append(value); row.append(tbl_idx); col.append(inter_idx)
                        count.append(inter_idx)
    return sp.coo_matrix((data, (row, col)), shape=(dim, dim))


def create_interaction_matrix_p(vs, p_double, Upp):
    """
    创建相互作用矩阵
    :param vs:
    :param p_double:
    :param Upp:
    :return:
    """
    print("start create_interaction_matrix_p")
    print("==========================")
    dim = vs.dim
    data = []
    row = []
    col = []
    if Upp != 0:
        for tbl_idx, value in p_double:
            data.append(Upp * value); row.append(tbl_idx); col.append(tbl_idx)
    return sp.coo_matrix((data, (row, col)), shape=(dim, dim))
