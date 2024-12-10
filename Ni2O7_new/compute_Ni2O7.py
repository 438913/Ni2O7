import time
import os

import parameters as pam
import variational_space
import basis_change as basis
import hamiltonian as ham
import ground_state as gs

start_time = time.time()


def compute_main(A, Upp, ed, ep, tpd, tpp):
    """
    Compute the matrix A_w in the main equation.
    :param A:
    :param Upp:
    :param ed:
    :param ep:
    :param tpd:
    :param tpp:
    :return:
    """
    Norb = pam.Norb
    Ni_position = pam.Ni_position
    Ni_num = len(Ni_position)
    vs = variational_space.VariationalSpace()

    tpd_nn_hop_dir, tpd_orbs, tpd_nn_hop_fac, tpp_nn_hop_fac \
        = ham.set_tpd_tpp(Norb, tpd, tpp)
    T_pd = ham.create_tpd_nn_matrix(vs, tpd_nn_hop_dir, tpd_nn_hop_fac)
    T_pp = ham.create_tpp_nn_matrix(vs, tpp_nn_hop_fac)
    Esite = ham.create_Esite_matrix(vs, A, ed, ep)
    H = T_pd + T_pp + Esite
    H0 = T_pd + T_pp + Esite
    d_double, d_part, p_double = basis.get_double_occ_list(vs)
    S_Ni_val_set = []
    Sz_Ni_val_set = []
    for i in range(Ni_num):
        U_Ni, S_Ni_val, Sz_Ni_val, AorB_Ni_sym \
            = basis.create_single_triplet_matrix(vs, d_double[i], d_part[i])
        S_Ni_val_set.append(S_Ni_val)
        Sz_Ni_val_set.append(Sz_Ni_val)
        U_Ni_d = (U_Ni.conjugate()).transpose()
        H_new = U_Ni_d @ H @ U_Ni
        Hint = ham.create_interaction_matrix_d(vs, d_double[i], d_part[i],
                                               S_Ni_val, Sz_Ni_val, AorB_Ni_sym, A)
        H = H_new + Hint
    Hint_p = ham.create_interaction_matrix_p(vs, p_double, Upp)
    H = H + Hint_p
    U_bond, bonding_val = basis.create_bounding_anti_bounding_basis_change_matrix(vs)
    U_bond_d = (U_bond.conjugate()).transpose()
    H_bond = U_bond_d @ H @ U_bond
    gs.get_ground_state(H_bond, vs, S_Ni_val_set, Sz_Ni_val_set, bonding_val)


if __name__ == '__main__':
    print(f'Start computation of Ni2O7 {pam.holes} holes')
    print('==========================')
    folder_path = 'data'
    file_paths = ['./data/type.txt', './data/val.txt', './data/weight.txt']
    os.makedirs(folder_path, exist_ok=True)
    A = pam.A
    Upp = pam.Upp
    print(f'{pam.holes} holes, A = {A}, B = {pam.B}, C = {pam.C}, Upp = {Upp}')
    for file_path in file_paths:
        with open(file_path, 'w') as f:
            f.write(f'{pam.holes} holes, A = {A}, B = {pam.B}, C = {pam.C}, Upp = {Upp}\n')
    with open('data/ground_energy.txt', 'w') as f:
        f.write(f'Sz = {pam.Sz_set[0]}\n')
    for tpd in pam.tpds:
        i = 0
        pressure = pam.pressures[i]
        print(f'pressure = {pressure} GPa, Sz_set = {pam.Sz_set}')
        ed = {key: item[i] for key, item in pam.eds.items()}
        ep = pam.eps[i]
        tpp = pam.tpps[i]
        print(f'ed = {ed}\nep = {ep}, tpd = {tpd}, tpp = {tpp}')
        for file_path in file_paths:
            with open(file_path, 'a') as f:
                f.write(f'pressure = {pressure} GPa, Sz_set = {pam.Sz_set}\n')
                f.write(f'ed = {ed}\nep = {ep}, tpd = {tpd}, tpp = {tpp}\n\n')
        compute_main(A, Upp, ed, ep, tpd, tpp)

    over_time = time.time()
    print("time cost:  ", over_time - start_time)
