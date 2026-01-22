import numpy            as np
import editdistance

from cls_TrellisStage   import TrellisStage
from ids_channel        import ids_channel
from bcjr               import bcjr

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

######
K = 10
N = 100

max_ITER = K

p_ins = 0.1
p_del = 0.1
p_sub = 0.1
###########

alph_card = 4

linter = 0
cum_mean_num = 0
while True:

    linter += 1    
    str_Dex = f"Center #{linter}"
    print("=" * len(str_Dex))
    print(str_Dex)
    print("-" * len(str_Dex))

    ondeck_center = np.random.randint(0, alph_card, size=N)

    input_app_array = np.zeros((K, alph_card, N))

    app_up_array = np.ones((K, alph_card, N))
    app_dw_array = np.ones((K, alph_card, N))
    
    NO_error_array_cov_anl = np.zeros((K, 2))
    conv_flag = 0
    for ITER in range(max_ITER):
        print(f'ITER {ITER+1}:')
        trace_Dex = 0
        for _ in range(K):
        
            ondeck_trace = ids_channel(ondeck_center, alph_card, p_ins, p_del, p_sub)

            ####################################################################################
            ### Building a Trellis Section ###
            ##################################

            inp_prior_dist = np.ones((alph_card))/alph_card

            section_instance = TrellisStage(ondeck_trace, alph_card, inp_prior_dist, p_ins, p_del, p_sub)

            ### Removing branches corresponding to Y \neq y ###

            section_instance.filter(ondeck_trace)
            # section_instance_temp = copy.copy(section_instance)

            ### Initial prior Pr(S_i | S_{i^-}) for all three stages.
                # These priors are updated iteratively to obtain
                # Pr(S_i | S_{i^-}, Y^(1)), Pr(S_i | S_{i^-}, Y^(2)), ..., Pr(S_i | S_{i^-}, Y^(K)).
            
            # Pr(S_i | S_{i^-}) = Pr(Y, S_i | S_{i^-})/Pr(Y | S_{i^-}, S_i)
            prior_s1_init = np.zeros((section_instance.s1.shape[0], N))
            prior_s1_init[:,:] = np.nan_to_num((section_instance.s1[:,3] / section_instance.s1[:,4])[:, None], nan=0.0)
            if ITER == 0:
                prior_s1 = prior_s1_init
            else:
                app_up = np.nan_to_num(input_app_array_prev[trace_Dex-1,:,:] / app_dw_array_prev[trace_Dex-1,:,:], nan=0.0)
                app_up = np.nan_to_num(app_up / app_up.sum(axis=0, keepdims=True), nan=0.0)

                app_dw = np.nan_to_num(input_app_array_prev[(trace_Dex+1) % K,:,:] / app_up_array_prev[(trace_Dex+1) % K,:,:], nan=0.0)
                app_dw = np.nan_to_num(app_dw / app_dw.sum(axis=0, keepdims=True), nan=0.0)

                app_up_array[trace_Dex:,:] = app_up
                app_dw_array[trace_Dex:,:] = app_dw

                app_matr_3 = np.nan_to_num((app_up * app_dw) / (np.tile(inp_prior_dist, (N, 1)).T), nan=0.0)
                app_matr_3 = np.nan_to_num(app_matr_3 / app_matr_3.sum(axis=0, keepdims=True), nan=0.0)

                prior_s1 = np.tile(app_matr_3, (section_instance.trace_length, 1))

            prior_s2_ins = np.zeros((section_instance.s2_ins_filt.shape[0], N))
            prior_s2_ins[:,:] = np.nan_to_num((section_instance.s2_ins_filt[:,3]/section_instance.s2_ins_filt[:,4])[:, None], nan=0.0)
            
            prior_s2_oth = np.zeros((section_instance.s2_oth_filt.shape[0], N))
            prior_s2_oth[:,:] = np.nan_to_num((section_instance.s2_oth_filt[:,3]/section_instance.s2_oth_filt[:,4])[:, None], nan=0.0)

            prior_s3 = np.zeros((section_instance.s3.shape[0], N))
            prior_s3[:,:] = np.nan_to_num((section_instance.s3[:,3]/section_instance.s3[:,4])[:, None], nan=0.0)


            ### Forward/Backward recursion

            v_set1_fw_mat, v_set2_fw_mat, v_set3_fw_mat, v_set4_fw_mat,\
            v_set1_bw_mat, v_set2_bw_mat, v_set3_bw_mat, v_set4_bw_mat,\
            _, _, _, _ = bcjr(N, section_instance,\
                            prior_s1, prior_s2_ins, prior_s2_oth, prior_s3)
            

            #######################################################
            # # W_mtr_2 = np.multiply(v_set2_fw_mat, v_set2_bw_mat)
            # # W_mtr_2 = W_mtr_2/W_mtr_2.sum(axis=0)
            # # 
            # # app_matr_2 = np.zeros((section_instance.alph_card, N))
            # # for xDex in range(section_instance.alph_card):
            # #     app_matr_2[xDex,:] = np.sum(W_mtr_2[np.arange(W_mtr_2.shape[0]) % section_instance.alph_card == xDex,:], axis=0)
            # #
            # ### # # note: max(|W_mtr_3-W_mtr_2|) = 0.0208
            ###############################################

            W_mtr_3 = np.multiply(v_set3_fw_mat, v_set3_bw_mat)
            W_mtr_3 = np.nan_to_num(W_mtr_3/W_mtr_3.sum(axis=0), nan=0.0)
            

            for xDex in range(section_instance.alph_card):
                input_app_array[trace_Dex, xDex, :] = np.sum(W_mtr_3[np.arange(W_mtr_3.shape[0]) % section_instance.alph_card == xDex,:], axis=0)

            hard_est = np.argmax(input_app_array[trace_Dex,:,:], axis=0)

            
            NO_error = editdistance.eval(ondeck_center, hard_est)
            NO_error_array_cov_anl[trace_Dex,0] = NO_error

            trace_Dex = trace_Dex+1
            print(f"\tTrace #{trace_Dex} | NO. of errors: {NO_error}")
            # print(editdistance.eval(ondeck_center, ondeck_trace))

        app_up_array_prev = app_up_array
        app_dw_array_prev = app_dw_array
        input_app_array_prev = input_app_array

        if np.all(NO_error_array_cov_anl == NO_error_array_cov_anl[0, 0]): #consensus check
            conv_flag = 1

        if NO_error==0 or conv_flag==1:
            break

        NO_error_array_cov_anl[:,1] = NO_error_array_cov_anl[:,0]

    print(f'\nError fraction (ondeck center): {NO_error/N}')
    cum_mean_num = cum_mean_num + NO_error/N
    print(f'Cumulative error fraction: {cum_mean_num/(linter)}\n')