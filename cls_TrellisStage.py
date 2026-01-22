import numpy as np
class TrellisStage():
    
    def __init__(self, ondeck_trace, alph_card, inp_prior_dist, p_ins, p_del, p_sub):

        self.trace_length = len(ondeck_trace)
        self.pointer_card = self.trace_length + 1
        self.alph_card = alph_card
        p_cor = 1 - (p_ins + p_del + p_sub)


        ########################################
        ## 1st stage : (P_{i}) -> (P_{i}, X_{i})

        ## stage_1_trellis:
            # 1st col: S_0^{i} =  (P_{i})
            # 2nd col: S_1^{i} == (P_{i},   X_{i})
            # 3rd col: (X_{i})
            # 4th col: Pr(X_{i})
            # 5th col: Pr(Phi|S_0, S_1) (allways = 1 as we don't have parallel branches at this stage)
            
            # this variable also acts like a state map for S_1^{i} <-> (P_{i},   X_{i})

        stage_1_trellis = np.zeros(((self.pointer_card-1)*alph_card, 5)) # number of pointers: pointer_card-1 -> the last pointer (=pointer_card) has no outgoing branches
        for iDex in range(self.pointer_card-1):
            stage_1_trellis[alph_card*iDex : alph_card*(iDex+1), 0] = iDex*np.ones((alph_card)) # S_0^{i} =  (P_{i})
            stage_1_trellis[alph_card*iDex : alph_card*(iDex+1), 1] = np.array(range(alph_card*iDex, alph_card*(iDex+1))) # S_1^{i} == (P_{i},   X_{i})
            stage_1_trellis[alph_card*iDex : alph_card*(iDex+1), 2] = np.array(range(alph_card)) # (X_{i})
            stage_1_trellis[alph_card*iDex : alph_card*(iDex+1), 3] = inp_prior_dist # Pr(X_{i})
            stage_1_trellis[alph_card*iDex : alph_card*(iDex+1), 4] = np.ones((alph_card))

            
        #####################################
        ## 2nd stage: modeling the IDS events

        # total number of edges leaving each vertice is 2|setX|+1

        #   > Insertion (|setX| edges): 
        #       (P_{i} = p, X_{i}) -> (P_{i} = p+1, X_{i})
        
        ## stage_2_op_ins_trellis:
            # 1st col: S_1^{i} == (P_{i} = p,   X_{i})
            # 2nd col: S_1^{i} == (P_{i} = p+1, X_{i})
            # 3rd col: (Y_{p})
            # 4th col: Pr(Y_{p}|insertion) . Pr(insertion) = p_ins/alph_card
            # 5th col: Pr(Y_{p}|S_0, S_1) = Pr(Y_{p}|insertion)
            
        stage_2_op_ins_trellis = np.zeros(( (self.pointer_card*alph_card-alph_card)*alph_card, 5 ))
        for iDex in range(self.pointer_card*alph_card-alph_card):
            stage_2_op_ins_trellis[alph_card*iDex : alph_card*(iDex+1), 0] = iDex*np.ones((alph_card)) # S_1^i == (P_{i} = p,   X_i)
            stage_2_op_ins_trellis[alph_card*iDex : alph_card*(iDex+1), 1] = (iDex+alph_card)*np.ones((alph_card)) # S_1^i == (P_{i} = p+1, X_i)
            stage_2_op_ins_trellis[alph_card*iDex : alph_card*(iDex+1), 2] = np.array(range(alph_card)) # (Y_{p})
            stage_2_op_ins_trellis[alph_card*iDex : alph_card*(iDex+1), 3] = p_ins * np.ones((alph_card))/alph_card # Pr(Y_{p}, insertion)
            stage_2_op_ins_trellis[alph_card*iDex : alph_card*(iDex+1), 4] = (p_ins * np.ones((alph_card))/alph_card)/np.sum(p_ins * np.ones((alph_card))/alph_card)

        #   > Deletion/Substitution/Replication (|setX|+1 edges): 
        #       (P_{i}, X_{i}) -> (P_{i+1}, X_{i}) 
        
        ## stage_2_op_other_trellis:
            # 1st col: S_1^{i} == (P_{i} = p,   X_{i})
            # 2nd col: S_2^{i} == (P_{i+1},   X_{i})
            # 3rd col: (Y_{p})
            # 4th col: Pr(Y_{p}|channel DSR) . Pr(Channel DSR)
            # 5th col: Pr(Y_{p}|S_0, S_1) = Pr(Y_{p}|Channel DSR)

        stage_2_op_other_trellis = np.zeros(( (self.pointer_card*alph_card-alph_card)*alph_card + (self.pointer_card*alph_card), 5 ))
        
        # Deletion branches at buttom of the trellis
        stage_2_op_other_trellis[-1*alph_card:, 0] = np.array(range(self.pointer_card*alph_card-alph_card, self.pointer_card*alph_card))
        stage_2_op_other_trellis[-1*alph_card:, 1] = np.array(range(self.pointer_card*alph_card-alph_card, self.pointer_card*alph_card))
        stage_2_op_other_trellis[-1*alph_card:, 2] = -1*np.ones((alph_card))
        stage_2_op_other_trellis[-1*alph_card:, 3] = np.ones((alph_card)) # single outgoing branch w.p. 1
        stage_2_op_other_trellis[-1*alph_card:, 4] = np.ones((alph_card))

        # Deletion/Substitution/Replication
        for iDex in range(self.pointer_card*alph_card-alph_card):

            # Deletion
            stage_2_op_other_trellis[(alph_card+1)*iDex, 0:4] = [iDex, iDex, -1, p_del]
            stage_2_op_other_trellis[(alph_card+1)*iDex, 4] = 1

            # Substitution/Replication
            stage_2_op_other_trellis[(alph_card+1)*iDex+1 : (alph_card+1)*(iDex+1), 0] = iDex*np.ones((alph_card))
            stage_2_op_other_trellis[(alph_card+1)*iDex+1 : (alph_card+1)*(iDex+1), 1] = (iDex+alph_card)*np.ones((alph_card))
            stage_2_op_other_trellis[(alph_card+1)*iDex+1 : (alph_card+1)*(iDex+1), 2] = np.array(range(alph_card))

            stage_2_op_other_trellis[(alph_card+1)*iDex+1 : (alph_card+1)*(iDex+1), 3] =  p_sub*np.ones((alph_card))/(alph_card-1)
            stage_2_op_other_trellis[(alph_card+1)*iDex+1 + iDex % alph_card, 3] = p_cor

            stage_2_op_other_trellis[(alph_card+1)*iDex+1 : (alph_card+1)*(iDex+1), 4] =  (p_sub*np.ones((alph_card))/(alph_card-1))/(p_cor + np.sum(p_sub*np.ones((alph_card-1))/(alph_card-1)))
            stage_2_op_other_trellis[(alph_card+1)*iDex+1 + iDex % alph_card, 4] = p_cor/(p_cor + np.sum(p_sub*np.ones((alph_card-1))/(alph_card-1)))


        ############################################
        ## 3rd stage : (P_{i+1}, X_{i}) -> (P_{i+1})
        
        ## stage_1_trellis:
            # 1st col: S_2^{i} == (P_{i+1},   X_{i})
            # 2nd col: S_3^{i} == (P_{i+1})
            # 3rd col: (X_{i})
            # 4th col: Pr(P_{i+1}|S_2^{i} == (P_{i+1},   X_{i})) \in {0, 1}
            # 5th col: Pr(Phi|S_0, S_1) (=1)

        stage_3_trellis = np.zeros((self.pointer_card*alph_card, 5)) # number of pointers: pointer_card
        for iDex in range(self.pointer_card):
            stage_3_trellis[alph_card*iDex : alph_card*(iDex+1), 0] = np.array(range(alph_card*iDex, alph_card*(iDex+1)))
            stage_3_trellis[alph_card*iDex : alph_card*(iDex+1), 1] = iDex*np.ones((alph_card))
            stage_3_trellis[alph_card*iDex : alph_card*(iDex+1), 2] = np.array(range(alph_card))
            stage_3_trellis[alph_card*iDex : alph_card*(iDex+1), 3] = np.ones((alph_card))
            stage_3_trellis[alph_card*iDex : alph_card*(iDex+1), 4] = np.ones((alph_card))

        self.s1 = stage_1_trellis
        self.s2_ins = stage_2_op_ins_trellis
        self.s2_oth = stage_2_op_other_trellis
        self.s3 = stage_3_trellis

    

    def filter(self, ondeck_trace):
        # Removing branches corresponding to Y \neq y #

        # 'filter' is a class method that add the attributes .stage_2_ins_filtered and .stage_2_oth_filtered 

        # When the trellis traverse from P_{i} = p to either P_{i} = p+1 (insertion) or P_{i+1} = p (+1) (DSR)...
        # ... an output is generated y_p; so the branches traversing vertically (every non-deletion branch)...
        # ... should be consistent with ondeck_trace (y_0 ... y_p ...)


        stage_2_ins_filtered = np.empty((0, self.s2_ins.shape[1]))
        stage_2_oth_filtered = np.empty((0, self.s2_oth.shape[1]))
        aux_ondeck_trace = np.append(ondeck_trace, -1)
        for pDex in range(self.trace_length+1):

            stage_2_ins_filtered = np.vstack((stage_2_ins_filtered,
            self.s2_ins[ (self.s2_ins[:,0]//self.alph_card == pDex) \
                       & (self.s2_ins[:,2] == aux_ondeck_trace[pDex]) ]))
            
            stage_2_oth_filtered = np.vstack((stage_2_oth_filtered,
            self.s2_oth[ (self.s2_oth[:,0]//self.alph_card == pDex) \
                     & ( (self.s2_oth[:,2] == aux_ondeck_trace[pDex])
                         |(self.s2_oth[:,2] == -1)
                       ) ]                   ))
        
        self.s2_ins_filt = stage_2_ins_filtered
        self.s2_oth_filt = stage_2_oth_filtered

        return stage_2_ins_filtered, stage_2_oth_filtered