import numpy as np
def fw_value(section_instance, v_set1):
    ### Forward value of vertices

    v_set2 = np.zeros((section_instance.pointer_card * section_instance.alph_card))
    v_set3 = np.zeros((section_instance.pointer_card * section_instance.alph_card))
    v_set4 = np.zeros((section_instance.pointer_card))

    ### V set NO. 2
    for v2_Dex in range(v_set2.shape[0]):
    # for v2_Dex in range(1):
        # this part corresponds to emptying the buffer (1st stage of the trellis section)
        prevDex_1 = section_instance.s1[section_instance.s1[:,1] == v2_Dex, 0]
        if prevDex_1.size != 0:
            branch_weight_1 = section_instance.s1[(section_instance.s1[:,0] == prevDex_1) \
                                                 &(section_instance.s1[:,1] == v2_Dex), 3]
            
            incoming_branch_1 = v_set1[prevDex_1.astype(int)] * branch_weight_1
        else:
            incoming_branch_1 = 0

        # this part corresponds to insertion event (parallel branches in the 1st stage of the trellis section)
        prevDex_2 = section_instance.s2_ins_filt[section_instance.s2_ins_filt[:,1] == v2_Dex, 0]
        if prevDex_2.size != 0:
            branch_weight_2 = section_instance.s2_ins_filt[(section_instance.s2_ins_filt[:,0] == prevDex_2) \
                                                          &(section_instance.s2_ins_filt[:,1] == v2_Dex), 3]
            
            incoming_branch_2 = v_set2[prevDex_2.astype(int)] * branch_weight_2
        else:
            incoming_branch_2 = 0

        v_set2[v2_Dex] = (incoming_branch_1 + incoming_branch_2).item()

    v_set2[:] = np.nan_to_num(v_set2[:]/np.sum(v_set2[:]), nan=0.0)

    ### V set NO. 3
    for v3_Dex in range(v_set3.shape[0]):
        
        prevDex = section_instance.s2_oth_filt[section_instance.s2_oth_filt[:,1] == v3_Dex, 0]
        for pDex in prevDex.astype(int):
            
            branch_weight = section_instance.s2_oth_filt[(section_instance.s2_oth_filt[:,0] == pDex) \
                                                        &(section_instance.s2_oth_filt[:,1] == v3_Dex), 3]
            v_set3[v3_Dex] = (v_set3[v3_Dex] \
                + branch_weight * v_set2[pDex]).item()
            
    v_set3[:] = np.nan_to_num(v_set3[:]/np.sum(v_set3[:]), nan=0.0)

    ### V set NO. 4

    for v4_Dex in range(v_set4.shape[0]):
        prevDex = section_instance.s3[section_instance.s3[:,1] == v4_Dex, 0]
        v_set4[v4_Dex] = np.sum(v_set3[prevDex.astype(int)])

    return v_set2, v_set3, v_set4


def fw_value_prior(section_instance, v_set1, prior_s1, prior_s2_ins, prior_s2_oth, prior_s3):
    ### Forward value of vertices

    v_set2 = np.zeros((section_instance.pointer_card * section_instance.alph_card))
    v_set3 = np.zeros((section_instance.pointer_card * section_instance.alph_card))
    v_set4 = np.zeros((section_instance.pointer_card))

    ### V set NO. 2
    for v2_Dex in range(v_set2.shape[0]):
    # for v2_Dex in range(1):
        # this part corresponds to preparing the buffer (1st stage of the trellis section)
        prevDex_1 = section_instance.s1[section_instance.s1[:,1] == v2_Dex, 0]
        if prevDex_1.size != 0:

            maskDex = ((section_instance.s1[:, 0] == prevDex_1) &
                       (section_instance.s1[:, 1] == v2_Dex))
            
            branch_weight_1 = section_instance.s1[maskDex, 4] * prior_s1[maskDex]
            
            incoming_branch_1 = v_set1[prevDex_1.astype(int)] * branch_weight_1
        else:
            incoming_branch_1 = 0

        # this part corresponds to insertion event (parallel branches in the 1st stage of the trellis section)
        prevDex_2 = section_instance.s2_ins_filt[section_instance.s2_ins_filt[:,1] == v2_Dex, 0]
        if prevDex_2.size != 0:

            maskDex = ((section_instance.s2_ins_filt[:, 0] == prevDex_2) &
                       (section_instance.s2_ins_filt[:, 1] == v2_Dex))

            branch_weight_2 = section_instance.s2_ins_filt[maskDex, 4] * prior_s2_ins[maskDex]
            
            incoming_branch_2 = v_set2[prevDex_2.astype(int)] * branch_weight_2
        else:
            incoming_branch_2 = 0

        v_set2[v2_Dex] = (incoming_branch_1 + incoming_branch_2).item()

    v_set2[:] = np.nan_to_num(v_set2[:]/np.sum(v_set2[:]), nan=0.0)

    
    


    ### V set NO. 3
    for v3_Dex in range(v_set3.shape[0]):
        
        prevDex = section_instance.s2_oth_filt[section_instance.s2_oth_filt[:,1] == v3_Dex, 0]
        for pDex in prevDex.astype(int):

            maskDex = ((section_instance.s2_oth_filt[:, 0] == pDex) &
                       (section_instance.s2_oth_filt[:, 1] == v3_Dex))
            
            branch_weight = section_instance.s2_oth_filt[maskDex, 4] * prior_s2_oth[maskDex]
            v_set3[v3_Dex] = (v_set3[v3_Dex] + branch_weight * v_set2[pDex]).item()
            
    v_set3[:] = np.nan_to_num(v_set3[:]/np.sum(v_set3[:]), nan=0.0)

    ### V set NO. 4
    for v4_Dex in range(v_set4.shape[0]):
        prevDex = section_instance.s3[section_instance.s3[:,1] == v4_Dex, 0]
        for pDex in prevDex.astype(int):

            maskDex = ((section_instance.s3[:, 0] == pDex) &
                       (section_instance.s3[:, 1] == v4_Dex))
            
            branch_weight = section_instance.s3[maskDex, 4] * prior_s3[maskDex]
            v_set4[v4_Dex] = (v_set4[v4_Dex] + branch_weight * v_set3[pDex]).item()
            
    return v_set2, v_set3, v_set4