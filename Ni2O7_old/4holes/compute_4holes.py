import math
# from numpy.linalg import inv
import sys

sys.path.append('../../src/')
from pylab import *

import parameters as pam
import variational_space as vs

# import hamiltonian as ham
import hamiltonian as ham  # convention of putting U/2 to d8 and d10 separately

import basis_change as basis
import utility as util
import ground_state as gs
# import ground_state_lanczos as gs
import time

import diff_pressure
import importlib

start_time = time.time()
M_PI = math.pi


#####################################
def compute_Aw_main(A, ep, tpd, tpp, tz_a1a1, tz_b1b1, pds, pdp, pps, ppp, Upp, \
                    d_Ni1_double, d_Ni2_double, p_double, double_Ni1_part, idx_Ni1, hole34_Ni1_part, \
                    double_Ni2_part, idx_Ni2, hole34_Ni2_part,
                    U_Ni1, S_Ni1_val, Sz_Ni1_val, U_Ni2, S_Ni2_val, Sz_Ni2_val, AorB_Ni1_sym, AorB_Ni2_sym):
    if Norb == 8:
        fname = 'ep' + str(ep) + '_epbilayer' + str(pam.epbilayer) + '_tpd' + str(tpd) + '_tpp' + str(tpp) \
                + '_tpzd' + str(tpzd) + '_tz_a1a1' + str(tz_a1a1) + '_Mc' + str(Mc) + '_Norb' + str(
            Norb) + '_eta' + str(eta)
        flowpeak = 'Norb' + str(Norb) + '_tpp' + str(tpp) + '_Mc' + str(Mc) + '_eta' + str(eta)
    elif Norb == 10 or Norb == 11 or Norb == 12:
        fname = 'ep' + str(ep) + '_epbilayer' + str(pam.epbilayer) + '_pdp' + str(pdp) + '_pps' + str(
            pps) + '_ppp' + str(ppp) \
                + '_tz_a1a1' + str(tz_a1a1) + '_tpzd' + str(tpzd) + '_Mc' + str(Mc) + '_Norb' + str(
            Norb) + '_eta' + str(eta)
        flowpeak = 'Norb' + str(Norb) + '_pps' + str(pps) + '_ppp' + str(ppp) + '_Mc' + str(Mc) + '_eta' + str(eta)

    w_vals = np.arange(pam.wmin, pam.wmax, pam.eta)
    Aw = np.zeros(len(w_vals))
    Aw_dd_total = np.zeros(len(w_vals))
    Aw_d8_total = np.zeros(len(w_vals))

    # set up H0
    if Norb == 7 or Norb == 4:
        tpd_nn_hop_dir, tpd_orbs, tpd_nn_hop_fac, tpp_nn_hop_fac \
            = ham.set_tpd_tpp(Norb, tpd, tpp, 0, 0, 0, 0)
    elif Norb == 10 or Norb == 12:
        tpd_nn_hop_dir, tpd_orbs, tpd_nn_hop_fac, tpp_nn_hop_fac \
            = ham.set_tpd_tpp(Norb, 0, 0, pds, pdp, pps, ppp)

    tz_fac = ham.set_tz(Norb, if_tz_exist, tz_a1a1, tz_b1b1)

    T_pd = ham.create_tpd_nn_matrix(VS, tpd_nn_hop_dir, tpd_orbs, tpd_nn_hop_fac)
    T_pp = ham.create_tpp_nn_matrix(VS, tpp_nn_hop_fac)
    #     T_z    = ham.create_tz_matrix(VS,tz_fac)

    Esite = ham.create_edep_diag_matrix(VS, A, ep)

    H0 = T_pd + T_pp + Esite
    #     H0 = T_pd
    print("H0 %s seconds ---" % (time.time() - start_time))

    '''
    Below probably not necessary to do the rotation by multiplying U and U_d
    the basis_change.py is only for label the state as singlet or triplet
    and assign the interaction matrix
    '''
    if pam.if_H0_rotate_byU == 1:
        H0_Ni1_new = U_Ni1_d.dot(H0.dot(U_Ni1))

    clf()

    if Norb == 4 or Norb == 7 or Norb == 10 or Norb == 11 or Norb == 12:
        Hint_Ni1 = ham.create_interaction_matrix_ALL_syms(VS, d_Ni1_double, double_Ni1_part, idx_Ni1,
                                                          hole34_Ni1_part, \
                                                          S_Ni1_val, Sz_Ni1_val, AorB_Ni1_sym, A)
        Hint_Ni2 = ham.create_interaction_matrix_ALL_syms(VS, d_Ni2_double, double_Ni2_part, idx_Ni2,
                                                          hole34_Ni2_part, \
                                                          S_Ni2_val, Sz_Ni2_val, AorB_Ni2_sym, A)
        Hint_p = ham.create_interaction_matrix_p(VS, p_double, Upp)
        if pam.if_H0_rotate_byU == 1:
            H_Ni1 = H0_Ni1_new + Hint_Ni1

            # continue rotate the basis for setting Cu layer's interaction (d_Cu_double)
            H0_Ni2_new = U_Ni2_d.dot(H_Ni1.dot(U_Ni2))
            H = H0_Ni2_new + Hint_Ni2 + Hint_p
        else:
            H = H0 + Hint_Ni2 + Hint_Ni1
        H_bond = U_bond_d.dot(H.dot(U_bond))
        H_bond.tocsr()

        ####################################################################################
        # compute GS only for turning on full interactions
        if pam.if_get_ground_state == 1:
            gs.get_ground_state(H_bond, VS, S_Ni1_val, Sz_Ni1_val, S_Ni2_val, Sz_Ni2_val, bonding_val, U_Ni1, U_Ni2)
        #             if Norb==8:
        #                 util.write_GS('Egs_'+flowpeak+'.txt',A,ep,tpd,vals[0])
        #                 #util.write_GS_components('GS_weights_'+flowpeak+'.txt',A,ep,tpd,wgt_d8, wgt_d9L, wgt_d10L2)
        #             elif Norb==10 or Norb==11 or Norb==12:
        #                 util.write_GS2('Egs_'+flowpeak+'.txt',A,ep,pds,pdp,vals[0])
        #                 #util.write_GS_components2('GS_weights_'+flowpeak+'.txt',A,ep,pds,pdp,wgt_d8, wgt_d9L, wgt_d10L2)

        #########################################################################
        '''
        Compute A(w) for various states
        '''


#         if pam.if_compute_Aw==1:
#             # compute d8
#             fig.compute_Aw_d8_sym(H, VS, d_double_no_eh, S_val, Sz_val, AorB_sym, A, w_vals, "Aw_d8_sym_", fname)

#             # compute d9L
#             b1L_state_indices, a1L_state_indices, b1L_state_labels, a1L_state_labels \
#                     = getstate.get_d9L_state_indices(VS, S_val, Sz_val)
#             fig.compute_Aw1(H, VS, w_vals, b1L_state_indices, b1L_state_labels, "Aw_b1L_", fname)
#             fig.compute_Aw1(H, VS, w_vals, a1L_state_indices, a1L_state_labels, "Aw_a1L_", fname)

#             # compute d10L2
#             d10L2_state_indices, d10L2_state_labels = getstate.get_d10L2_state_indices(VS, S_val, Sz_val)
#             fig.compute_Aw1(H, VS, w_vals, d10L2_state_indices, d10L2_state_labels, "Aw_d10L2_", fname)

#             # compute d8Ls for some special states
#             a1b1Ls_S0_state_indices, a1b1Ls_S0_state_labels, \
#             a1b1Ls_S1_state_indices, a1b1Ls_S1_state_labels, \
#             a1a1Ls_state_indices, a1a1Ls_state_labels \
#                                             = getstate.get_d8Ls_state_indices(VS, d_double_one_eh, S_val, Sz_val)
#             fig.compute_Aw1(H, VS, w_vals, a1b1Ls_S0_state_indices, a1b1Ls_S0_state_labels, "Aw_a1b1Ls_S0_", fname)
#             fig.compute_Aw1(H, VS, w_vals, a1b1Ls_S1_state_indices, a1b1Ls_S1_state_labels, "Aw_a1b1Ls_S1_", fname)
#             fig.compute_Aw1(H, VS, w_vals, a1a1Ls_state_indices, a1a1Ls_state_labels, "Aw_a1a1Ls_", fname)

#             # compute d9L2s
#             d9L2s_state_indices, d9L2s_state_labels = getstate.get_d9L2s_state_indices(VS)
#             fig.compute_Aw1(H, VS, w_vals, d9L2s_state_indices, d9L2s_state_labels, "Aw_d9L2s_", fname)


##########################################################################
if __name__ == '__main__':
    for pressure in [4]:
        diff_pressure.pressure = pressure
        importlib.reload(pam)
        Mc = pam.Mc
        print('Mc=', Mc)

        Norb = pam.Norb
        eta = pam.eta
        ed = pam.ed

        As = pam.As
        B = pam.B
        C = pam.C

        tz_a1a1 = pam.tz_a1a1
        tz_b1b1 = pam.tz_b1b1

        if_tz_exist = pam.if_tz_exist

        # set up VS
        VS = vs.VariationalSpace(Mc)

        d_Ni1_double, idx_Ni1, hole34_Ni1_part, double_Ni1_part, \
            d_Ni2_double, idx_Ni2, hole34_Ni2_part, double_Ni2_part, \
            p_double = ham.get_double_occu_list(VS)

        # change the basis for d_double states to be singlet/triplet

        if pam.basis_change_type == 'd_double':
            U_Ni1, S_Ni1_val, Sz_Ni1_val, AorB_Ni1_sym, \
                = basis.create_singlet_triplet_basis_change_matrix_d_double \
                (VS, d_Ni1_double, double_Ni1_part, idx_Ni1, hole34_Ni1_part)
            U_Ni2, S_Ni2_val, Sz_Ni2_val, AorB_Ni2_sym, \
                = basis.create_singlet_triplet_basis_change_matrix_d_double \
                (VS, d_Ni2_double, double_Ni2_part, idx_Ni2, hole34_Ni2_part)

        U_bond, bonding_val = basis.create_bonding_anti_bonding_basis_change_matrix(VS)
        U_Ni1_d = (U_Ni1.conjugate()).transpose()
        U_Ni2_d = (U_Ni2.conjugate()).transpose()
        U_bond_d = (U_bond.conjugate()).transpose()
        # check if U if unitary
        # checkU_unitary(U,U_d)

        if Norb == 7 or Norb == 4:
            for tpd in pam.tpds:
                for ep in pam.eps:
                    for A in pam.As:
                        util.get_atomic_d8_energy(A, B, C)
                        for tpp in pam.tpps:
                            for Upp in pam.Upps:
                                print('===================================================')
                                print('A=', A, 'ep=', ep, ' tpd=', tpd, ' tpp=', tpp, \
                                      ' Upp=', Upp)
                                Sz = pam.Sz
                                model = pam.model
                                pressure = pam.pressure
                                txt1 = open('./data1/lowest_eigenvalue.txt', model)
                                txt2 = open('./data1/weight.txt', model)
                                txt3 = open('./data1/simplified_state_weight.txt', model)
                                txt1.write(f'pressure = {pressure} Gpa  Sz = {Sz}   tpp = {tpp}\n')
                                txt2.write(f'pressure = {pressure} Gpa  Sz = {Sz}   tpp = {tpp}\n\n')
                                txt3.write(f'pressure = {pressure} Gpa  Sz = {Sz}   tpp = {tpp}\n')
                                txt1.close()
                                txt2.close()
                                txt3.close()
                                compute_Aw_main(A, ep, tpd, tpp, tz_a1a1, tz_b1b1, 0, 0, 0, 0, Upp, \
                                                d_Ni1_double, d_Ni2_double, p_double, double_Ni1_part, idx_Ni1,
                                                hole34_Ni1_part, \
                                                double_Ni2_part, idx_Ni2, hole34_Ni2_part, \
                                                U_Ni1, S_Ni1_val, Sz_Ni1_val, U_Ni2, S_Ni2_val, Sz_Ni2_val, AorB_Ni1_sym,
                                                AorB_Ni2_sym)

    print("--- %s seconds ---" % (time.time() - start_time))