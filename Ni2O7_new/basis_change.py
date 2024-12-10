import numpy as np
import scipy.sparse as sp

import parameters as pam
from variational_space import count_inversions, count_inversions_for_disorder
Ni_position = pam.Ni_position
holes = pam.holes


def get_double_type(states):
    double_type = {Ni_position[0]: [], Ni_position[1]: []}
    for hole in range(holes):
        x, y, orb, s = states[hole]
        if (x, y) in double_type:
            double_type[(x, y)] += [hole]
        else:
            double_type[(x, y)] = [hole]
    return double_type


def get_double_occ_list(vs):
    dim = vs.dim
    lookup_tbl = vs.lookup_tbl
    d_Ni_list = [[] for _ in Ni_position]
    d_part = [[] for _ in Ni_position]
    p_list = []
    for i in range(dim):
        states = lookup_tbl[i]
        # double_type = {(-1, 0): [hole_idx1, hole_idx2, ...], (1, 0): [hole_idx1, hole_idx2, ...],
        # O_position: [hole_idx, ...]}
        double_type = get_double_type(states)
        value = 0
        for position, part in double_type.items():
            if position in Ni_position:
                d_num = len(double_type[position])
                if d_num == 2:
                    index = Ni_position.index(position)
                    d_Ni_list[index] += [i]
                    d_part[index] += [part]
            else:
                p_num = len(double_type[position])
                if p_num > 1:
                    value += p_num * (p_num - 1) / 2
        if value > 0:
            p_list.append([i, value])
    return d_Ni_list, d_part, p_list


def create_single_triplet_matrix(vs, d_double, d_part):
    """
    创建单态三重态变换矩阵
    :param vs:
    :param d_double:
    :param d_part:
    :return:
    """
    dim = vs.dim
    lookup_tbl = vs.lookup_tbl
    data = []; row = []; col = []
    S_val = 2 * np.ones(dim, dtype=int)
    Sz_val = 2 * np.ones(dim, dtype=int)
    AorB_sym = np.zeros(dim, dtype=int)
    count_list = []
    for i in range(dim):
        if i not in d_double:
            data.append(1.0); row.append(i); col.append(i)
    for i1, tbl_idx in enumerate(d_double):
        if tbl_idx in count_list:
            continue
        states = lookup_tbl[tbl_idx]
        hole_idx1, hole_idx2 = d_part[i1]
        orb1, s1 = states[hole_idx1][-2:]
        orb2, s2 = states[hole_idx2][-2:]
        if s1 == s2:
            data.append(1.0); row.append(tbl_idx); col.append(tbl_idx)
            S_val[tbl_idx] = 1
            if s1 == 'up':
                Sz_val[tbl_idx] = 1
            else:
                Sz_val[tbl_idx] = -1
        else:
            if orb1 != orb2:
                partner_state = [list(state) for state in states]
                partner_state[hole_idx1][-1], partner_state[hole_idx2][-1] = (
                    partner_state[hole_idx2][-1], partner_state[hole_idx1][-1]
                )
                ph = 1 if count_inversions_for_disorder(partner_state) % 2 == 0 else -1
                partner_state = sorted(partner_state)
                partner_state = tuple(map(tuple, partner_state))
                partner_idx = vs.get_index(partner_state)
                count_list.append(partner_idx)

                data.append(1 / np.sqrt(2)); row.append(tbl_idx); col.append(tbl_idx)
                data.append(-ph / np.sqrt(2)); row.append(partner_idx); col.append(tbl_idx)
                S_val[tbl_idx] = 0; Sz_val[tbl_idx] = 0

                data.append(1 / np.sqrt(2)); row.append(tbl_idx); col.append(partner_idx)
                data.append(ph / np.sqrt(2)); row.append(partner_idx); col.append(partner_idx)
                S_val[partner_idx] = 1; Sz_val[partner_idx] = 0

            else:
                if orb1 == 'dxz':
                    partner_state = [list(state) for state in states]
                    partner_state[hole_idx1][-2], partner_state[hole_idx2][-2] = 'dyz', 'dyz'
                    partner_state = sorted(partner_state)
                    partner_state = tuple(map(tuple, partner_state))
                    partner_idx = vs.get_index(partner_state)
                    count_list.append(partner_idx)

                    data.append(1 / np.sqrt(2)); row.append(tbl_idx); col.append(tbl_idx)
                    data.append(1 / np.sqrt(2)); row.append(partner_idx); col.append(tbl_idx)
                    S_val[tbl_idx] = 0; Sz_val[tbl_idx] = 0
                    AorB_sym[tbl_idx] = 1

                    data.append(1 / np.sqrt(2)); row.append(tbl_idx); col.append(partner_idx)
                    data.append(-1 / np.sqrt(2)); row.append(partner_idx); col.append(partner_idx)
                    S_val[partner_idx] = 0; Sz_val[partner_idx] = 0
                    AorB_sym[partner_idx] = -1

                elif orb1 != 'dyz':
                    data.append(1.0); row.append(tbl_idx); col.append(tbl_idx)
                    S_val[tbl_idx] = 0; Sz_val[tbl_idx] = 0

    return sp.coo_matrix((data, (row, col)), shape=(dim, dim)), S_val, Sz_val, AorB_sym


def create_bounding_anti_bounding_basis_change_matrix(vs):
    """
    创建成键态和反成键态变换矩阵
    :param vs:
    :return:
    """
    dim = vs.dim
    data = []; row = []; col = []
    count_list = []
    bonding_val = np.zeros(dim, dtype=int)
    lookup_tbl = vs.lookup_tbl
    for tbl_idx in range(dim):
        if tbl_idx in count_list:
            continue
        pre_state = lookup_tbl[tbl_idx]
        sym_state = [(-x, y, orb, s) for x, y, orb, s in pre_state]
        ph = 1 if count_inversions_for_disorder(sym_state) % 2 == 0 else -1
        sym_state = sorted(sym_state)
        sym_state = tuple(sym_state)
        sym_idx = vs.get_index(sym_state)
        if sym_idx == tbl_idx:
            data.append(1.0); row.append(tbl_idx); col.append(tbl_idx)
        else:
            num_in_left_Ni = 0
            num_in_right_Ni = 0
            num_in_left_O = 0
            num_in_right_O = 0
            for x, y, _, _ in pre_state:
                if (x, y) == (-1, 0):
                    num_in_left_Ni += 1
                if (x, y) == (1, 0):
                    num_in_right_Ni += 1
                if x < 0 and (x, y) != (-1, 0):
                    num_in_left_O += 1
                if x > 0 and (x, y) != (1, 0):
                    num_in_right_O += 1
            if num_in_left_Ni < num_in_right_Ni:
                bound_idx = tbl_idx
                anti_idx = sym_idx
            elif num_in_left_Ni == num_in_right_Ni:
                if num_in_left_O < num_in_right_O:
                    bound_idx = tbl_idx
                    anti_idx = sym_idx
                else:
                    bound_idx = sym_idx
                    anti_idx = tbl_idx
            else:
                bound_idx = sym_idx
                anti_idx = tbl_idx

            data.append(1 / np.sqrt(2)); row.append(bound_idx); col.append(bound_idx)
            data.append(ph / np.sqrt(2)); row.append(anti_idx); col.append(bound_idx)
            bonding_val[bound_idx] = 1

            data.append(1 / np.sqrt(2)); row.append(bound_idx); col.append(anti_idx)
            data.append(-ph / np.sqrt(2)); row.append(anti_idx); col.append(anti_idx)
            bonding_val[anti_idx] = -1

            count_list.append(sym_idx)
    return sp.coo_matrix((data, (row, col)), shape=(dim, dim)), bonding_val
