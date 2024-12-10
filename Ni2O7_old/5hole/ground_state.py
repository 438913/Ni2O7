import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg
import time
import parameters as pam
model = pam.model

def state_classification(state_param):
    """
    Classify the state into different types
    :param state_param: state_param = [(orb1, x1, y1), (orb2, x2, y2), (orb3, x3, y3), (orb4, x4, y4), (orb5, x5, y5)]
    :return: state_type: string, e.g. 'd10_L_d9_L3',
    means 0 hole in left O, 0 hole in Ni(-1, 0), 1 hole in middle O(0, 0), 1 hole in Ni(1, 0), 3 holes in right O
    """
    num_in_left_O = 0
    num_in_middle_O = 0
    num_in_right_O = 0
    num_in_left_Ni = 0
    num_in_right_Ni = 0
    for (orb, x, y) in state_param:
        if orb in pam.Ni_orbs:
            if x == -1:
                num_in_left_Ni += 1
            if x == 1:
                num_in_right_Ni += 1
        if orb in pam.O_orbs:
            if (x, y) in [(-2, 0), (-1, 1), (-1, -1)]:
                num_in_left_O += 1
            if (x, y) == (0, 0):
                num_in_middle_O += 1
            if (x, y) in [(2, 0), (1, 1), (1, -1)]:
                num_in_right_O += 1
    O_num_to_str = {1: 'L', 2: 'L2', 3: 'L3', 4: 'L4'}
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


def get_ground_state(matrix, VS, S_Ni1_val, Sz_Ni1_val, S_Ni2_val, Sz_Ni2_val, bonding_val):
    """
    Obtain the ground state info, namely the lowest peak in Aw_dd's component
    in particular how much weight of various d8 channels: a1^2, b1^2, b2^2, e^2
    """
    t1 = time.time()
    print ('start getting ground state')
    # in case eigsh works:
    Neval = pam.Neval
    vals, vecs = sps.linalg.eigsh(matrix, k=Neval, which='SA')

    print('lowest eigenvalue of H from np.linalg.eigsh = ')
    print(vals)
    print(vals[0])

    if abs(vals[0] - vals[3]) < 10 ** (-5):
        number = 4
    elif abs(vals[0] - vals[2]) < 10 ** (-5):
        number = 3
    elif abs(vals[0] - vals[1]) < 10 ** (-5):
        number = 2
    else:
        number = 1
    print('Degeneracy of  ground state is ', number)
    txt = open('./data1/lowest_eigenvalue.txt', 'a')
    txt.write('lowest eigenvalue of H = ' + '\n')
    txt.write(str(vals) + '\n')
    txt.write(f'Degeneracy of  ground state is {number}\n\n\n')
    txt.close()
    dim = vecs.shape[0]
    weights = abs(vecs[:, :number]) ** 2
    weights_average = np.average(weights, axis=1)
    # weights_average = weights
    ilead = np.argsort(-weights_average)
    state_type_weight = {}
    specific_type_weight = {}
    total = 0
    print("Compute the weights in GS (original state before bonding-antibonding)")
    for istate in ilead:
        weight = weights_average[istate]
        total += weight
        state = VS.get_state(VS.lookup_tbl[istate])
        orb1 = state['hole1_orb']
        orb2 = state['hole2_orb']
        orb3 = state['hole3_orb']
        orb4 = state['hole4_orb']
        orb5 = state['hole5_orb']
        x1, y1, z1 = state['hole1_coord']
        x2, y2, z2 = state['hole2_coord']
        x3, y3, z3 = state['hole3_coord']
        x4, y4, z4 = state['hole4_coord']
        x5, y5, z5 = state['hole5_coord']
        bonding = bonding_val[istate]


        input_state = [(orb1, x1, y1), (orb2, x2, y2), (orb3, x3, y3), (orb4, x4, y4), (orb5, x5, y5)]
        state_type_key = state_classification(input_state)
        state_type_key += f'({bonding})'
        state_type_value = weight
        if state_type_key in state_type_weight:
            state_type_weight[state_type_key] += state_type_value
            specific_type_weight[state_type_key] += [istate]
        else:
            state_type_weight[state_type_key] = state_type_value
            specific_type_weight[state_type_key] = [istate]

    weight_list = sorted(state_type_weight.items(), key=lambda item: item[1], reverse=True)
    type_distribution = {'0.015 ~ 0.020': 0, '0.010 ~ 0.015': 0, '0.006 ~ 0.010': 0, '0.002 ~ 0.006': 0,  '< 0.002': 0}
    other_type_weight = 0
    txt = open('./data1/simplified_state_weight.txt', 'a')
    txt1 = open('./data1/weight.txt', 'a')
    txt1.write(f'total weight = {total}\n')
    for state_type, type_weight in weight_list:
        if type_weight < 0.020:
            if type_weight > 0.015:
                type_distribution['0.015 ~ 0.020'] += 1
            elif type_weight > 0.010:
                type_distribution['0.010 ~ 0.015'] += 1
            elif type_weight > 0.006:
                type_distribution['0.006 ~ 0.010'] += 1
            elif type_weight > 0.002:
                type_distribution['0.002 ~ 0.006'] += 1
            else:
                type_distribution['< 0.002'] += 1
            other_type_weight += type_weight
            continue
        print(f'{state_type}: {type_weight}')
        txt.write(f'{state_type}: {type_weight}\n')
        txt1.write(f'{state_type}: {type_weight}\n')
        other_distribution = {'0.003 ~ 0.005': 0, '0.001 ~ 0.003': 0, '< 0.001': 0}
        other_weight = 0
        for istate in specific_type_weight[state_type]:
            weight = weights_average[istate]
            state = VS.get_state(VS.lookup_tbl[istate])
            s1 = state['hole1_spin']
            s2 = state['hole2_spin']
            s3 = state['hole3_spin']
            s4 = state['hole4_spin']
            s5 = state['hole5_spin']
            orb1 = state['hole1_orb']
            orb2 = state['hole2_orb']
            orb3 = state['hole3_orb']
            orb4 = state['hole4_orb']
            orb5 = state['hole5_orb']
            x1, y1, z1 = state['hole1_coord']
            x2, y2, z2 = state['hole2_coord']
            x3, y3, z3 = state['hole3_coord']
            x4, y4, z4 = state['hole4_coord']
            x5, y5, z5 = state['hole5_coord']
            bonding = bonding_val[istate]
            if weight > 0.005:
                print('\tstate ', istate, ' ', orb1, s1, x1, y1, z1, ' ', orb2, s2, x2, y2, z2, ' ', orb3, s3, x3, y3,
                      z3, ' ', orb4, s4, x4, y4, z4, ' ', orb5, s5, x5, y5, z5,
                      '\n\tS_Ni1=', S_Ni1_val[istate], ',  Sz_Ni1=', Sz_Ni1_val[istate],
                      ',  S_Ni2=', S_Ni2_val[istate], ',  Sz_Ni2=', Sz_Ni2_val[istate], ',bonding=', bonding,
                      ", weight = \n\t", weight, '\n')
                txt1.write(f'\t({orb1} {s1} {x1} {y1} {z1})  ({orb2} {s2} {x2} {y2} {z2})  ({orb3} {s3} {x3} {y3} {z3})\n\t'
                           f'({orb4} {s4} {x4} {y4} {z4})  ({orb5} {s5} {x5} {y5} {z5})\n'
                           f'\tS_Ni1 = {S_Ni1_val[istate]},  Sz_Ni1 = {Sz_Ni1_val[istate]},  S_Ni2 = {S_Ni2_val[istate]},  Sz_Ni2 = {Sz_Ni2_val[istate]}'
                           f',  bonding = {bonding},  weight = \n\t{weight}\n\n')
            else:
                other_weight += weight
                if weight > 0.003:
                    other_distribution['0.003 ~ 0.005'] += 1
                elif weight > 0.001:
                    other_distribution['0.001 ~ 0.003'] += 1
                else:
                    other_distribution['< 0.001'] += 1
        txt1.write(f'\tother_distribution:  {other_distribution}\n\tother weight = {other_weight}\n\n')

    txt1.write(f'other_type_distribution: {type_distribution}\rother_type_weight: {other_type_weight}\n\n')
    txt1.close()
    txt.write('\n\n')
    txt.close()
    print(f'total weight = {total}')
