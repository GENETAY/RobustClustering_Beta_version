# Date : 2020-09-15 14:09

#!/usr/bin/python
# -*- coding: utf-8 -*-
#"""
# K-bMOM algorithm
#"""

__author__ = "Camille Saumard, Edouard GENETAY"
__copyright__ = "Copyright 2020"
__version__ = "8.4"
__maintainer__ = "Edouard GENETAY"
__email__  = "genetay.edouard@gmail.com"
__status__ = "version beta"

# modifications of october 2019 by Camille BRUNET SAUMARD:
# add a sampling scheme : uniform without conditioning to the cluster-membership
# Add stopping criterion based on Atkaike's criterion
#
# modifications of november 2019 by Edouard GENETAY:
# I realized that the simulation done with particular initialization procedures
# did not applied for K-bMOM because not initialization centers can be given in argument.
# I correct that and launch simu for K-bMOM.
#
# In addition in this version, dictonaries have been changed into lists

# import time
import numpy as np
import random
from math import modf, log, inf
from scipy.spatial.distance import cdist
from .kmedianpp import euclidean_distances, kmedianpp_init
from .KbMOM_v8_5_utils import block_size_max, minimal_number_of_blocks,flatten_list

class KbMOM:
    
    def __init__(
        self,
        X,
        K,
        nbr_blocks,
        coef_ech = 6,
        max_iter = 10,
        strategy = 'centroids',
        quantile = 0.5,
        outliers = None,
        confidence = 0.95,
        threshold = 0.00001,
        initial_centers = None,
        init_by_kmeanspp = True,
        nb_min_repr_by_cluster = None,
        score_strategy="block_members_distances",
        random_state = 1
        ):
        '''
        param X : data we want to cluster. Given at instanciation to know their number and their dimensionality.      
        param K : number of clusters
        param nbr_blocks : number of blocks to create in init and loop
        param coef_ech : NUMBER of data in each block and cluster
        param max_iter : number of iterations of the algorithm
        param strategy :
            take distance to centroids of the median block or to representants of the median block.
            This variable has no effects in this version.
        param quantile : quantile to keep for the empirical risk; by default the median
        param outliers : number of supposed outliers
        param confidence : probability that the median block is corrupted is majorated be 'confidence'
        param threshold : threshold used with the aitkens criterion
        param init_by_kmeanspp : will the initialisation centers randomly pick according kmeans++ or kmedians++ procedure.
        param nb_min_repr_by_cluster : how many representant of each cluster at leasat should there be in each block
        param score_strategy:
            this variable controls how outlyingness is computed. It can take three values "denumber",
            "centroids_distances" and "block_members_distances"
        param random_state: set the random seed
        
        type X : array
        type K : int
        type nbr_blocks : 
        type coef_ech : 
        type max_iter : 
        type strategy : 
        type quantile : 
        type outliers : 
        type confidence :
        type threshold : 
        type init_by_kmeanspp : 
        type nb_min_repr_by_cluster :
        type score_strategy :
        type random_state: int
        
        return :
        '''
        
        # given element
        self.K = K
        self.max_iter = max_iter
        self.n, self.p = X.shape
        self.quantile = quantile
        self.strategy = strategy
        self.coef_ech = coef_ech
        self.B = nbr_blocks
        self.alpha = 1 - confidence
        self.threshold = threshold
        self.init_by_kmeanspp = init_by_kmeanspp
        self.nb_min_repr_by_cluster = nb_min_repr_by_cluster
        self.score_strategy = score_strategy
        self.random_state = random_state
        
        self.initial_centers = initial_centers
        
        # Test some given values
        if outliers is not None:
            self.outliers = outliers
            t_sup = minimal_number_of_blocks(
                n_sample=self.n,
                n_outliers=self.outliers,
                b_size=self.coef_ech,
                alpha=self.alpha
            )
            if self.coef_ech > t_sup :
                self.coef_ech = max((t_sup-5),1)
                self.coef_ech = int(round(self.coef_ech))
                print('warning:: the size of blocks has been computed according to the breakdown point theory (from 1)')

            B_sup = block_size_max(
                n_sample=self.n,
                n_outliers=self.outliers
            )
            if self.B < B_sup :
                self.B     = round(B_sup) + 10
                self.B     = int(self.B)
                print('warning:: the number of blocks has been computed according to the breakdown point theory (from 2)')
        
        # Deal with exceptions:
        if self.n < 2*self.K:
            raise Exception("K is too big. K must be less than 2n, with n the size of the data sample")
        if self.coef_ech < 2*self.K:
            self.coef_ech = 2*self.K
        
        # internal element initialization
        self.cluster_size = [round(self.n/self.K)]*self.K
        self.weights_list = [1/self.n for i in range(self.n)] # dictionary
        self.score = np.zeros((self.n,))
        self.centers = 0
        self.iter = 0
    
    def sampling_init(self,X):
        '''
        # Initialisation function: create nbr_blocks blocks, initialize with a kmeans++, compute a partition
        and the empirical risk per block // Select the block that corresponds to the quantile "quantile" of the risk and update all the parameters
        '''
        init_blocks = [0]*self.B
        
        # instanciation of kmeans++
        x_squared = X**2
        x_squared_norms = x_squared.sum(axis=1)
        
        # Blocks creation
        size_of_blocks = self.coef_ech
        for i in range(self.B):
            idx = random.choices(np.arange(self.n),k = int(size_of_blocks))
            
            cent_kmedian = kmedianpp_init(
                X[idx,:], 
                self.K, 
                x_squared_norms[idx],
                random_state=2*self.random_state, 
                n_local_trials=None,
                square=self.init_by_kmeanspp
            )
            km_dist = cdist(X[idx,:],cent_kmedian)
            km_labels_ = km_dist.argmin(axis=1)
            
            init_blocks[i] = [[j for j,x in zip(idx,km_labels_) if x==k] for k in range(self.K)]
            
        return init_blocks
            
        
    def sampling_all_blocks(self,X):#,nbr_blocks,weighted_point,cluster_sizes):
        '''
        # Function which creates nbr_blocks blocks conditionally to the partition_dict and return a dictionary 
        # per block of the ids of selected  data per cluster and per bloc
        #  
        #     nbr_blocks      : number of blocks to be created
        #     weighted_point : a dictonnary of clusters containing the lists of weights
        #     cluster_sizes   : vector containing the size of each clusters
        '''
        dict_blocks = [0]*self.B
        for i in range(self.B):
            # to ensure there are K categories in each block run that part, one can go through the first piece of code, else run 'else'
            list_of_points = []
            if self.nb_min_repr_by_cluster is not None:
                for k in range(self.K):
                    list_of_points = list_of_points + random.choices(np.array(self.partition_dict[k]), k=self.nb_min_repr_by_cluster)
                idx_indep = random.choices(np.arange(self.n), k=self.coef_ech-self.nb_min_repr_by_cluster*self.K)
            else:
                idx_indep = random.choices(np.arange(self.n), k=self.coef_ech)
            idx = list_of_points + idx_indep
            
#            # to release the constrain "ensure each category has got at least one representant" run this code
#             idx = random.choices(np.arange(self.n), k=self.coef_ech)


            km_dist    = cdist(X[idx,:],self.centers)
            km_labels_ = km_dist.argmin(axis=1)         
            dict_blocks[i] = [[j for j,x in zip(idx,km_labels_) if x==k] for k in range(self.K)]
            
        return dict_blocks
    
    def within_var(self,one_block,X):
        '''
        # Function which returns a list of within inertia per cluster for the given block
        #      one_block : dictionnary of the subsample of one block according to the clusters
        '''
        n_b     = 0
        var_b   = [0]*self.K
#         print("one_block = "+str(one_block))
#         for key , val in enumerate(one_block):
# #             # if block can lack one or more category representant and you want it to work, run that :
# #             if len(val)>0:
# #                 X_block    = X[val,:]
# #                 var_b[key] = len(val)*np.sum(np.var(X_block,axis=0))
# #                 n_b       += len(val)
# #             else:
# # #                 var_b[key] = -1
# # #         return([x/n_b if x>= 0 else -1 for x in var_b])
# #                 var_b[key] = 0
    
# #             # otherwise, if the generation of blocks make sure all categories are represented in block (occur after init), then run that :
# #             X_block = X[val,:]
# # #             print("val = "+str(val))
# # #             var_b[key] = len(val)*np.sum(np.var(X_block,axis=0))
# # #             n_b += len(val)
# # #         return([x/n_b for x in var_b])
    
# #             var_b[key] = len(val)*np.sum(np.var(X_block,axis=0))/self.coef_ech
# # #             n_b += len(val)

        # finally, if you want to make sure that the clusters in block contains a given amount of data even during init, run that :
        # test is: if all clusters have enough data, don't disqualify the block (enough means at least self.nb_min_repr_by_cluster)
        if sum(np.array(list(map(len,one_block))) >= self.nb_min_repr_by_cluster) == self.K :
            for key , val in enumerate(one_block):
                X_block = X[val,:]
    #             print("val = "+str(val))
    #             var_b[key] = len(val)*np.sum(np.var(X_block,axis=0))
    #             n_b += len(val)
    #         return([x/n_b for x in var_b])

                var_b[key] = len(val)*np.sum(np.var(X_block,axis=0))/self.coef_ech
    #             n_b += len(val)
        else: # disqualify the block
            var_b = [-1]
        return var_b
    
            
    def Qrisk(self,all_blocks,X):
        '''
        # Function which computes the sum of all within variances and return:
                - Q_risk               = the q-quantile block risk
                - Q_id                 =  id of the associated q-quantile block
                - dict_within[Q_id]    = list of the within variances of the selected bloc
        ```parameters ```       
            . all_blocks : output of sampling_all_blocks
            . X          : matrix of datapoints
        '''
        
        dict_within = [0]*self.B
        Qb_risks = [0]*self.B # np.zeros((1,self.B))
        
        # case: computation of kmeans loss
        for key, one_block_ in enumerate(all_blocks):
            dict_within[key] = self.within_var(one_block_,X)
#             # if it may happen that block do not have all category represented and one wants to take into account only those full representative run that, otherwise, don't run that
#             if -1 not in dict_within[key]:
#                 Qb_risks[key] = sum(dict_within[key])
            # otherwise, run that :

#             if key == 0:
            Qb_risks[key] = sum(dict_within[key])

#         Compute and get the Q-quantile bloc
        # if it may happen that block do not have all category represented and one wants to take into account only those full representative run that, otherwise, don't run that
        if -1 in Qb_risks:
            qb_risks = [x for x in Qb_risks if x>=0]
            len_     = len(qb_risks)
            Q_risk = sorted(qb_risks)[round(self.quantile*len_)]
        else:
            Q_risk = sorted(Qb_risks)[round(self.quantile*self.B)]
        
        # otherwise, run that :
        risks_argsort = np.argsort(np.array(Qb_risks),kind="mergesort")
        quantile_index = risks_argsort[round(self.quantile*self.B)]
        quantile_value = Qb_risks[quantile_index]
        self.res_Qb_risk.append(quantile_value)
        return dict_within[quantile_index],all_blocks[quantile_index]
#         # anterior version
#         Q_risk = np.sort(a=np.array(Qb_risks),kind='mergesort')[round(self.quantile*self.B)]
#         self.res_Qb_risk.append(Q_risk)
#         # Select the Q-quantile bloc
#         Q_id = Qb_risks.index(Q_risk)
#         return dict_within[Q_id],all_blocks[Q_id]

    
    def Qb_centers(self,Q_block,X):
        '''
        #compute the mean vector of each cluster in the q-quantile block and return it
        '''
        centers_ = []
        for k in range(self.K):
            center_k = X[Q_block[k],:].mean(axis=0)
            centers_.append(center_k)
        return np.array(centers_)
    
    def shapen(self,partition_list):
        '''
        Function which shapes the partition and weigths as dictionaries
        
        ``` parameters ```
        
              partition_list : list of cluster affectations sorted according to id datapoint
              
        '''
        partition_d = {k:[] for k in range(self.K)}
        
        for i, x in enumerate(partition_list):
            partition_d[x].append(i)
            
        self.partition_dict = partition_d
    
    def weigthingscheme(self,X,Qblock,Qb_within,D_nk_min_rows,partition_array):
        '''
        Function which computes data depth
        ``` prms ```
            param Qblock: Q-quantile block
            param Qb_within:
            param D_nk:
            param D_nk_min_rows:
        ''' 
        if self.score_strategy == 'denumber':
            for clus, idk in enumerate(Qblock):
                self.score[idk] += 1
                
        elif self.score_strategy == 'centroids_distances':
            for clus, idk in enumerate(Qblock):
                if Qb_within[clus] == 0:
                    # not to divide by zero
                    self.score[idk] += 1
                else:
                    self.score[idk] += np.exp(-D_nk_min_rows[idk]/Qb_within[clus]) # boils down to one 1 when Qb_within[clus] tends to 0
        
        elif self.score_strategy == 'block_members_distances':
            elements_in_Qblock = X[flatten_list(Qblock)]
            D_nk = cdist(XA=X,XB=elements_in_Qblock) # distance between datapoint and those in Qblock
#             print(D_nk)
            D_nk_min_rows = np.amin(D_nk,axis=1) # distance between datapoint and the nearest within Qblock
#             print(D_nk_min_rows)
            for clus in range(len(Qblock)):
#                 print(f'Qb_within = {Qb_within}')
#                 print(f'Qb_within[clus] = {Qb_within[clus]}')
                mask_loc_ = partition_array == clus
                if Qb_within[clus] == 0:
                    # not to divide by zero
                    self.score[mask_loc_] += 1
                else:
#                     print(f'D_nk_min_rows[mask_loc_] = {D_nk_min_rows[mask_loc_]}')
#                     print(np.exp(-D_nk_min_rows[mask_loc_]/Qb_within[clus]))
                    self.score[mask_loc_] += np.exp(-D_nk_min_rows[mask_loc_]/Qb_within[clus])
            
    
    
    def update_loop(self,X,Qb,Qb_within):
        '''
        # Function which updates scores, weights_list, partition and size of clusters
        #
        # Qb : Q-quantile block
        # Q_block_list_of_within_var : within inertia of the qq block
        '''
        # updates centers
        self.centers = self.Qb_centers(Qb,X)
        self.res_Qb_centers.append(self.centers)
        
        # retrieve partition of data
        D_nk = cdist(XA=X,XB=self.centers) # take distances from centroids
        partition_array = D_nk.argmin(axis=1)
        
        # compute empirical risk
        D_nk_min_rows = np.amin(D_nk,axis=1)
        empirical_risk = (D_nk_min_rows**2).mean()
        self.res_emp_risk.append(empirical_risk)
        
        # update size of clusters:
        self.cluster_size = np.bincount(partition_array)
        
        # compute the weights of each point:
        self.weigthingscheme(X,Qb,Qb_within,D_nk_min_rows,partition_array)
        
        # change format of affectation vector (from array to list)
        self.shapen(partition_array)
    
    def initialisation_without_init_centers(self,X):
        # Initialisation step :
        # initialisation per block: sampling M blocks and init via kmeans++
        init_blocks = self.sampling_init(X)
        
        # compute empirical risk among blocks and select the Q-quantile-block
        Q_within, Q_b = self.Qrisk(init_blocks,X)
        
        # update all the global variables
        self.update_loop(X,Q_b,Q_within)
        
        # save results
        self.id_Qblock_init = Q_b
#         self.res_Qb_risk.append(Q_risk)
    
    def initialisation_with_init_centers(self,X):
        # take initial centers given as parameter
        self.centers = self.initial_centers
        
        # retrieve partition
        D_nk_init = cdist(XA=X,XB=self.centers)
        partition_array_init = D_nk_init.argmin(axis=1)
        
        # compute empirical risk
        D_nk_init_min_rows = D_nk_init[[np.arange(self.n).tolist(),partition_array_init.tolist()]]
        empirical_risk = (D_nk_init_min_rows**2).mean()
        
        # compute cluster sizes
        self.cluster_size = np.bincount(partition_array_init)
        
        # change format of affectation vector (from array to list)
        self.shapen(partition_array_init)
        
        # save results
        self.res_Qb_risk.append(inf) # variable whose value cannot exist when init centers are given
        self.res_emp_risk.append(empirical_risk)
        self.res_Qb_centers.append(self.centers)
    
    def fit(self,X):
        '''
        Main loop of the K-bMOM algorithm:
        
        param X : numpy array = contains the data we eant to cluster      
        type X : array
        '''
        self.id_Qblock_init = None
        self.res_Qb_risk    = []
        self.res_emp_risk   = []
        self.res_Qb_centers = []
        Aitkens        = [None,None]
         
        # initialisation step
        if self.initial_centers is not None:
            self.initialisation_with_init_centers(X)
        else:
            self.initialisation_without_init_centers(X)
        
        # Main Loop - fitting process
        if (self.max_iter == 0):
            condition = False
        else:
            condition = True
        while condition:
            # sampling
            all_blocks = self.sampling_all_blocks(X)
            
            # Compute empirical risk for all blocks and select the Q-block
#             print(all_blocks)
            Qb_within_var, Q_block = self.Qrisk(all_blocks,X)      
            
            # updates
            self.update_loop(X,Q_block,Qb_within_var)
            
            self.iter += 1
            if self.iter>1:
                Aitkens_ = self.stopping_crit(self.res_Qb_risk)
                Aitkens.append(Aitkens_)
                if Aitkens_ < self.threshold:
                    condition = False
                if self.iter >= self.max_iter:
                    condition = False
           
        return {'centroids':self.centers
                ,'labels':self.predict(X)
                ,'id_Qblock_init':self.id_Qblock_init
                ,'convergence': Aitkens
                ,'scores':self.score
                ,'risk': self.res_emp_risk
                ,'Qb_risk': self.res_Qb_risk
                ,'Qb_centers': self.res_Qb_centers
                ,'B':self.B
                ,'t':self.coef_ech
#                 ,'n_iter':self.iter
               }
    
    
    def predict(self,X):
        '''
        Function which computes the partition based on the centroids of Median Block
        '''
        D_nk = cdist(X,self.centers)
        return D_nk.argmin(axis=1)
    
    def get_centers(self):
        return self.centers
    

    def bloc_size(self,n_sample,n_outliers):
        '''
        Function which fits the maximum size of blocks before a the breakpoint
        
        these bounds are obtained by chernoff inequality. Proof may not be provided,
        for more info about them, contact me by email.
        ```prms```
        n_sample: nb of data
        n_outlier: nb of outliers
        '''
        return (log(2.)/log(1/(1- (n_outliers/n_sample))))


    def bloc_nb(self,n_sample,n_outliers,b_size=None,alpha=0.05):
        '''
        Function which fits the minimum nb of blocks for a given size t before a the breakpoint
        
        these bounds are obtained by chernoff inequality. Proof may not be provided,
        for more info about them, contact me by email.
        ```prms```
        n_sample: nb of data
        n_outlier: nb of outliers
        b_size = bloc_size
        alpha : threshold confiance
        '''
        if n_outliers/n_sample >= 0.5:
            print('too much noise')
            return()
        elif b_size is None:
            t = bloc_size(n_sample,n_outliers)
            return log(1/alpha) / (2* ((1-n_outliers/n_sample)**t - 1/2)**2)
        else:
            t = b_size
            return log(1/alpha) / (2* ((1-n_outliers/n_sample)**t - 1/2)**2)
   
    def stopping_crit(self,risk_median):
        risk_ = risk_median[::-1][:3]
        den = (risk_[2]-risk_[1])-(risk_[1]-risk_[0])
        Ax = risk_[2] - (risk_[2]-risk_[1])**2/den
#         print(Ax)
        return Ax
    
    def stopping_crit_GMM(self,risk_median):
        risk_ = risk_median[::-1][:3]
        Aq = (risk_[0] - risk_[1])/(risk_[1] - risk_[2])
        
        Rinf = risk_[1] + 1/(1-Aq)*(risk_[0] - risk_[1])
        return Rinf