import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg

import parameters as pam
Ni_position = pam.Ni_position
Ni_num = len(Ni_position)


def state_classification(state_param):
    """
    Classify the state into different types
    :param state_param: state_param = ((x1, y1, orb1, s1), (x2, y2, orb2, s2)...)
    :return: state_type: string, e.g. 'd10_L_d9_L2',
    means 0 hole in left O, 0 hole in Ni(-1, 0), 1 hole in middle O(0, 0), 1 hole in Ni(1, 0), 2 holes in right O
    """
    num_in_left_O = 0
    num_in_middle_O = 0
    num_in_right_O = 0
    num_in_left_Ni = 0
    num_in_right_Ni = 0
    for x, y, _, _ in state_param:
        if (x, y) == (-1, 0):
            num_in_left_Ni += 1
        if (x, y) == (1, 0):
            num_in_right_Ni += 1
        if x < 0 and (x, y) != (-1, 0):
            num_in_left_O += 1
        if x > 0 and (x, y) != (1, 0):
            num_in_right_O += 1
        if (x, y) == (0, 0):
            num_in_middle_O += 1
    O_num_to_str = {1: 'L', 2: 'L2', 3: 'L3', 4: 'L4', 5: 'L5', 6: 'L6'}
    Ni_num_to_str = {0: 'd10', 1: 'd9', 2: 'd8', 3: 'd7', 4: 'd6'}
    if num_in_left_Ni == 0 and num_in_right_Ni == 0:
        state_type = 'Lm_Ln'
    else:
        if num_in_left_O == 0:
            state_type = f'{Ni_num_to_str[num_in_left_Ni]}'
        else:
            state_type = f'{O_num_to_str[num_in_left_O]}_{Ni_num_to_str[num_in_left_Ni]}'
        if num_in_middle_O == 0:
            state_type += f'_{Ni_num_to_str[num_in_right_Ni]}'
        else:
            state_type += f'_{O_num_to_str[num_in_middle_O]}_{Ni_num_to_str[num_in_right_Ni]}'
        if num_in_right_O != 0:
            state_type += f'_{O_num_to_str[num_in_right_O]}'
    return state_type


def get_ground_state(matrix, vs, S_Ni_val, Sz_Ni_val, bonding_val):
    """
    Get the ground state of the system
    :param matrix:
    :param vs:
    :param S_Ni_val:
    :param Sz_Ni_val:
    :param bonding_val:
    :return:
    """
    lookup_tbl = vs.lookup_tbl
    print('start getting ground state')
    vals, vecs = sps.linalg.eigsh(matrix, k=pam.num_vals, which='SA')
    print('lowest eigenvalue of H from np.linalg.eigsh = ')
    print(vals)
    number = 1
    for i in range(1, pam.num_vals):
        if abs(vals[i] - vals[0]) > 1e-4:
            number = i
            break
    print('Degeneracy of ground state is ', number)
    weight_average = np.average(abs(vecs[:, :number]) ** 2, axis=1)
    for k in range(1):
        print('k = ', k)
        print('eigenvalue = ', vals[k])
        with open('data/ground_energy.txt', 'a') as f:
            f.write(f'{vals[k]}\n')
        # all_weight = abs(vecs[:, k]) ** 2
        all_weight = weight_average
        sorted_index = np.argsort(-all_weight)
        total = 0
        state_type_weight = {}
        specific_weight = {}
        print("Compute the weights in GS (lowest Aw peak)")
        for tbl_idx in sorted_index:
            weight = all_weight[tbl_idx]
            total += weight
            states = lookup_tbl[tbl_idx]
            state_type = state_classification(states)
            state_type += f'({bonding_val[tbl_idx]})'
            if state_type in state_type_weight:
                state_type_weight[state_type] += weight
                specific_weight[state_type] += [tbl_idx]
            else:
                state_type_weight[state_type] = weight
                specific_weight[state_type] = [tbl_idx]
        state_type_weight = sorted(state_type_weight.items(), key=lambda item: item[1], reverse=True)
        with open('data/val.txt', 'a') as f:
            f.write(f'{str(vals)}\n\n')
        weight_file = open('data/weight.txt', 'a')
        weight_file.write(f'total weight = {total}\n')
        type_file = open('data/type.txt', 'a')
        for state_type, type_weight in state_type_weight:
            type_file.write(f'{state_type}: {type_weight}\n')
            if type_weight > 0.005:
                print(f'{state_type}: {type_weight}', end='\n')
                weight_file.write(f'{state_type}: {type_weight}\n')
                for tbl_idx in specific_weight[state_type]:
                    states = lookup_tbl[tbl_idx]
                    weight = all_weight[tbl_idx]
                    if weight > 0.002:
                        print(end='\t')
                        weight_file.write('\t')
                        state_num = 0
                        for state in states:
                            x, y, orb, s = state
                            print(f'({x}, {y}, {orb}, {s})', end='  ')
                            weight_file.write(f'({x}, {y}, {orb}, {s})  ')
                            state_num += 1
                            if state_num == 3:
                                weight_file.write('\n\t')
                        print(end='\n\t')
                        weight_file.write('\n\t')
                        for i in range(Ni_num):
                            if S_Ni_val[i][tbl_idx] > 1:
                                continue
                            print(f'Ni{i + 1}: S = {S_Ni_val[i][tbl_idx]}, Sz = {Sz_Ni_val[i][tbl_idx]}', end=', ')
                            weight_file.write(f'Ni{i + 1}: S = {S_Ni_val[i][tbl_idx]}, Sz = {Sz_Ni_val[i][tbl_idx]}, ')
                        weight_file.write(f'bonding = {bonding_val[tbl_idx]}\n\tweight = {weight}\n\n')
                        print(f'bonding = {bonding_val[tbl_idx]}', end=', ')
                        print(f'weight = {weight}\n')
        weight_file.write('\n')
        weight_file.close()
        type_file.write('\n')
        type_file.close()
        print(f'total weight = {total}')
