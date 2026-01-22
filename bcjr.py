import numpy as np
from fw_value           import fw_value_prior
from bw_value           import bw_value_prior
from update_priors      import update_priors

def bcjr(N, section_instance, prior_s1, prior_s2_ins, prior_s2_oth, prior_s3):

    ### Forward/Backward recursion
        # Each trellis section consists of 3 stages between 4 vertex sets. 
        # During a full simulation round (N trellis sections), we construct 4 matrices each correspond to one vertex set:
        # rows correspond to the index of vertices, and columns index the trellis section.


    # Initial Values

    v_set1_fw_mat = np.zeros((section_instance.pointer_card, N))
    v_set2_fw_mat = np.zeros((section_instance.pointer_card * section_instance.alph_card, N))
    v_set3_fw_mat = np.zeros((section_instance.pointer_card * section_instance.alph_card, N))
    v_set4_fw_mat = np.zeros((section_instance.pointer_card, N))

    v_set1_bw_mat = np.zeros((section_instance.pointer_card, N))
    v_set2_bw_mat = np.zeros((section_instance.pointer_card * section_instance.alph_card, N))
    v_set3_bw_mat = np.zeros((section_instance.pointer_card * section_instance.alph_card, N))
    v_set4_bw_mat = np.zeros((section_instance.pointer_card, N))

    v_set1_fw_mat[0,0] = 1
    v_set4_bw_mat[-1,-1] = 1
    for fwDex in range(N):
        bwDex = N-1 - fwDex

        v_set1_fw_mat[:,fwDex] = v_set4_fw_mat[:,fwDex-1] if fwDex != 0 else v_set1_fw_mat[:,fwDex]
        v_set2_fw_mat[:,fwDex],\
        v_set3_fw_mat[:,fwDex],\
        v_set4_fw_mat[:,fwDex] = fw_value_prior(section_instance, v_set1_fw_mat[:,fwDex],\
                                                prior_s1[:,fwDex], prior_s2_ins[:,fwDex], prior_s2_oth[:,fwDex], prior_s3[:,fwDex])

        v_set4_bw_mat[:,bwDex] = v_set1_bw_mat[:,bwDex+1] if bwDex != N-1 else v_set4_bw_mat[:,bwDex]
        v_set1_bw_mat[:,bwDex],\
        v_set2_bw_mat[:,bwDex],\
        v_set3_bw_mat[:,bwDex] = bw_value_prior(section_instance, v_set4_bw_mat[:,bwDex],\
                                                prior_s1[:,bwDex], prior_s2_ins[:,bwDex], prior_s2_oth[:,bwDex], prior_s3[:,bwDex])


    ### Calculating Priors

    prior_s1, prior_s2_ins, prior_s2_oth, prior_s3\
        = update_priors(N, section_instance,\
                        prior_s1, prior_s2_ins, prior_s2_oth, prior_s3,\
                        v_set1_fw_mat, v_set2_fw_mat, v_set3_fw_mat,\
                        v_set2_bw_mat, v_set3_bw_mat, v_set4_bw_mat) 
    

    return v_set1_fw_mat, v_set2_fw_mat, v_set3_fw_mat, v_set4_fw_mat,\
           v_set1_bw_mat, v_set2_bw_mat, v_set3_bw_mat, v_set4_bw_mat,\
           prior_s1, prior_s2_ins, prior_s2_oth, prior_s3