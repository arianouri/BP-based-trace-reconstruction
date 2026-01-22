import numpy as np
def update_priors(N, section_instance,\
                  prior_s1, prior_s2_ins, prior_s2_oth, prior_s3,\
                  v_set1_fw_mat, v_set2_fw_mat, v_set3_fw_mat,\
                  v_set2_bw_mat, v_set3_bw_mat, v_set4_bw_mat):


    prior_new_s1     = np.zeros_like(prior_s1)
    prior_new_s2_ins = np.zeros_like(prior_s2_ins)
    prior_new_s2_oth = np.zeros_like(prior_s2_oth)
    prior_new_s3     = np.zeros_like(prior_s3)

    for tDex in range(N):
        
        for bDex_1 in range(prior_new_s1.shape[0]):
            prior_new_s1[bDex_1, tDex] \
                = v_set1_fw_mat[section_instance.s1[bDex_1, 0].astype(int), tDex]\
                * v_set2_bw_mat[section_instance.s1[bDex_1, 1].astype(int), tDex]\
                * section_instance.s1[bDex_1, 4] * prior_s1[bDex_1, tDex]
            
        # normalize + update priors
        group_sums = np.zeros(section_instance.pointer_card-1) # -1 because the last state of first stage has no output
        np.add.at(group_sums, section_instance.s1[:, 0].astype(int), prior_new_s1[:, tDex])
        prior_s1[:, tDex] = np.nan_to_num(prior_new_s1[:, tDex] / group_sums[section_instance.s1[:, 0].astype(int)], nan=0.0)
            

        for bDex_2_ins in range(prior_new_s2_ins.shape[0]):
            prior_new_s2_ins[bDex_2_ins, tDex] \
                = v_set2_fw_mat[section_instance.s2_ins_filt[bDex_2_ins, 0].astype(int), tDex]\
                * v_set2_bw_mat[section_instance.s2_ins_filt[bDex_2_ins, 1].astype(int), tDex]\
                * section_instance.s2_ins_filt[bDex_2_ins, 4] * prior_s2_ins[bDex_2_ins, tDex]      

        for bDex_2_oth in range(prior_new_s2_oth.shape[0]):
            prior_new_s2_oth[bDex_2_oth, tDex] \
                = v_set2_fw_mat[section_instance.s2_oth_filt[bDex_2_oth, 0].astype(int), tDex]\
                * v_set3_bw_mat[section_instance.s2_oth_filt[bDex_2_oth, 1].astype(int), tDex]\
                * section_instance.s2_oth_filt[bDex_2_oth, 4] * prior_s2_oth[bDex_2_oth, tDex]

        # normalize + update priors
        group_sums = np.zeros(section_instance.pointer_card * section_instance.alph_card)
        np.add.at(group_sums, section_instance.s2_ins_filt[:, 0].astype(int), prior_new_s2_ins[:, tDex])
        np.add.at(group_sums, section_instance.s2_oth_filt[:, 0].astype(int), prior_new_s2_oth[:, tDex])
        prior_s2_ins[:, tDex] = np.nan_to_num(prior_new_s2_ins[:, tDex] / group_sums[section_instance.s2_ins_filt[:, 0].astype(int)], nan=0.0)
        prior_s2_oth[:, tDex] = np.nan_to_num(prior_new_s2_oth[:, tDex] / group_sums[section_instance.s2_oth_filt[:, 0].astype(int)], nan=0.0)


        for bDex_3 in range(prior_new_s3.shape[0]):
            prior_new_s3[bDex_3, tDex] \
                = v_set3_fw_mat[section_instance.s3[bDex_3, 0].astype(int), tDex]\
                * v_set4_bw_mat[section_instance.s3[bDex_3, 1].astype(int), tDex]\
                * section_instance.s3[bDex_3, 4] * prior_s3[bDex_3, tDex]
            
        # normalize + update priors
        group_sums = np.zeros(section_instance.pointer_card * section_instance.alph_card)
        np.add.at(group_sums, section_instance.s3[:, 0].astype(int), prior_new_s3[:, tDex])
        prior_s3[:, tDex] = np.nan_to_num(prior_new_s3[:, tDex] / group_sums[section_instance.s3[:, 0].astype(int)], nan=0.0) 

    return prior_s1, prior_s2_ins, prior_s2_oth, prior_s3