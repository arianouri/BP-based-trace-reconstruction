import numpy as np
def bw_value(section_instance, v_set4):
    ### Backward value of vertices

    v_set3 = np.zeros((section_instance.pointer_card * section_instance.alph_card))
    v_set2 = np.zeros((section_instance.pointer_card * section_instance.alph_card))
    v_set1 = np.zeros((section_instance.pointer_card))


    ### V set NO. 3
    for v3_Dex in reversed(range(v_set3.shape[0])):

        postDex = section_instance.s3[section_instance.s3[:,0] == v3_Dex, 1]
        v_set3[v3_Dex] = (v_set4[postDex.astype(int)]).item()
            
    v_set3[:] = np.nan_to_num(v_set3[:]/np.sum(v_set3[:]), nan=0.0)


    ### V set NO. 2
    for v2_Dex in reversed(range(v_set2.shape[0])):

        postDex_ins = section_instance.s2_ins_filt[section_instance.s2_ins_filt[:,0] == v2_Dex, 1]
        postDex_oth = section_instance.s2_oth_filt[section_instance.s2_oth_filt[:,0] == v2_Dex, 1]
        
        if postDex_ins.size != 0:
            branch_weight_ins = section_instance.s2_ins_filt[(section_instance.s2_ins_filt[:, 0] == v2_Dex)\
                                                            &(section_instance.s2_ins_filt[:, 1] == postDex_ins), 3]
                
            v_set2[v2_Dex] = v_set2[v2_Dex] + branch_weight_ins * v_set2[postDex_ins.astype(int)]
        
        for pDex in postDex_oth:
            
            branch_weight_oth = section_instance.s2_oth_filt[(section_instance.s2_oth_filt[:, 0] == v2_Dex)\
                                                            &(section_instance.s2_oth_filt[:, 1] == pDex), 3]
            
            v_set2[v2_Dex] = v_set2[v2_Dex] + branch_weight_oth * v_set3[pDex.astype(int)]

    v_set2[:] = np.nan_to_num(v_set2[:]/np.sum(v_set2[:]), nan=0.0)


    ### V set NO. 1
    for v1_Dex in range(v_set1.shape[0]):
        # This loop is **not reversed**.
        # The last vertex has no outgoing branch, so its backward value cannot be computed
        # and there is no initial value for it. 
        # We therefore set its backward value equal to that of the previous vertex in the same stage.

        postDex = section_instance.s1[section_instance.s1[:,0] == v1_Dex, 1]

        if postDex.size != 0:
            for pDex in postDex:
                branch_weight = section_instance.s1[(section_instance.s1[:,0] == v1_Dex)\
                                                   &(section_instance.s1[:,1] == pDex), 3]
                
                v_set1[v1_Dex] = v_set1[v1_Dex] + branch_weight * v_set2[pDex.astype(int)]
        else:
            v_set1[v1_Dex] = v_set1[v1_Dex-1]

    v_set1[:] = np.nan_to_num(v_set1[:]/np.sum(v_set1[:]), nan=0.0)


    return v_set1, v_set2, v_set3


def bw_value_prior(section_instance, v_set4, prior_s1, prior_s2_ins, prior_s2_oth, prior_s3):

    ### Backward value of vertices

    v_set3 = np.zeros((section_instance.pointer_card * section_instance.alph_card))
    v_set2 = np.zeros((section_instance.pointer_card * section_instance.alph_card))
    v_set1 = np.zeros((section_instance.pointer_card))


    ### V set NO. 3
    for v3_Dex in reversed(range(v_set3.shape[0])):

        postDex = section_instance.s3[section_instance.s3[:,0] == v3_Dex, 1]

        maskDex = ((section_instance.s3[:, 0] == v3_Dex) &
                   (section_instance.s3[:, 1] == postDex))

        branch_weight = section_instance.s3[maskDex, 4] * prior_s3[maskDex]

        v_set3[v3_Dex] = (branch_weight * v_set4[postDex.astype(int)]).item()
            
    v_set3[:] = np.nan_to_num(v_set3[:]/np.sum(v_set3[:]), nan=0.0)


    ### V set NO. 2
    for v2_Dex in reversed(range(v_set2.shape[0])):

        postDex_ins = section_instance.s2_ins_filt[section_instance.s2_ins_filt[:,0] == v2_Dex, 1]
        postDex_oth = section_instance.s2_oth_filt[section_instance.s2_oth_filt[:,0] == v2_Dex, 1]
        
        if postDex_ins.size != 0:

            maskDex = (section_instance.s2_ins_filt[:, 0] == v2_Dex) &\
                      (section_instance.s2_ins_filt[:, 1] == postDex_ins)

            branch_weight_ins = section_instance.s2_ins_filt[maskDex, 4] * prior_s2_ins[maskDex]
                
            v_set2[v2_Dex] = v_set2[v2_Dex] + branch_weight_ins * v_set2[postDex_ins.astype(int)]
        
        for pDex in postDex_oth:

            maskDex = (section_instance.s2_oth_filt[:, 0] == v2_Dex)&\
                      (section_instance.s2_oth_filt[:, 1] == pDex)
            
            branch_weight_oth = section_instance.s2_oth_filt[maskDex, 4] * prior_s2_oth[maskDex]
            
            v_set2[v2_Dex] = v_set2[v2_Dex] + branch_weight_oth * v_set3[pDex.astype(int)]

    v_set2[:] = np.nan_to_num(v_set2[:]/np.sum(v_set2[:]), nan=0.0)

        
    ### V set NO. 1
    for v1_Dex in range(v_set1.shape[0]):
        # This loop is **not reversed**.
        # The last vertex has no outgoing branch, so its backward value cannot be computed
        # and there is no initial value for it. 
        # We therefore set its backward value equal to that of the previous vertex in the same stage.

        postDex = section_instance.s1[section_instance.s1[:,0] == v1_Dex, 1]

        if postDex.size != 0:
            for pDex in postDex:

                maskDex = (section_instance.s1[:,0] == v1_Dex)&\
                          (section_instance.s1[:,1] == pDex)

                branch_weight = section_instance.s1[maskDex, 4] * prior_s1[maskDex]
                
                v_set1[v1_Dex] = v_set1[v1_Dex] + branch_weight * v_set2[pDex.astype(int)]
        else:
            v_set1[v1_Dex] = v_set1[v1_Dex-1]

    v_set1[:] = np.nan_to_num(v_set1[:]/np.sum(v_set1[:]), nan=0.0)

    return v_set1, v_set2, v_set3