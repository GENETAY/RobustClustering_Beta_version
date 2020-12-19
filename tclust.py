''' 
LIST of differences between the R program and the python one :
1)  the initialization of the cluster sizes are forced to have non zero components in the python code
    not in the R one. It happened to me during the tests that the R program initialized a cluster with
    size 0. The problem is that the design of the algorithm enables initially-non-zero cluster to
    degenerate if it is better but not an initially-zero cluster to become non degenerated.
    
2)  in the tclust function, the return function encountered in the body of the code may not work
    identically as in R. The associated cases were indeed not tested.
    
3)  One cannot given the initialiazed centers, sigma, cluster_sizes and cluster_weights in the R version
    One can instead in the python one.

4)  The R version does not support alpha=0 as input, but the python one does.

5)  The python version does not work identically to the R one when p==1

6)  The R version has got a few more arguments than the python one. The
    corresponding features in R are unavailable in python
    > fuzzy (nevertheless result of the R code is not satisfying)
    > trace
    > restr_deter
    > f_hook_iter
    > f_hook_model


LIST of differences between the tclust that is implemented and the version in the scientific article:
1)  the argument m that is used in the fuzzy estimation does not appear in the article.
    implemented .^(m/(m-1)) , in the article .^(1)
2)  the eigenvalues restriction gets respected through a thresholding in the implemented version and
    through the dykstra projection algorithm in the article
'''

import numpy as np
from numpy import linalg as LA 
# from numpy.random import multivariate_normal
import copy
# from sys import exit
from math import floor,log,inf # modf,sqrt,exp,cos,sin
import pandas as pd
# from scipy.spatial.distance import cdist,mahalanobis
from scipy.stats import multivariate_normal #,cauchy,chi2,mode,


def get_init_cluster_size(dict_iter,dict_param):
    # this function initializes the size of the clusters so that it sums up to no_trim=floor((1-alpha)n)
    # moreover, and contrarily to what is coded in R, we add a test to insure that no cluster size is initilized to 0
    # if K was taken too big, then the model selection will take indeed some cluster sizes equal to 0
    
    # change variable names for conciseness
    K=dict_param["K"]
    not_trimmed_number=dict_param["not_trimmed_number"]
    
    if K==1:
        # if only one cluster, then it contains all not trimmed points
        res=not_trimmed_number
    else:
        crit=True
        while crit:
            # drawn randomly (but not uniformly) proportions as follows :
            # drawn randomly uniformly K values between 0 and 1
            K_random_values=np.random.uniform(low=0,high=1,size=K)#;print(K_random_values)
            # normalize to have proportion/probability vector
            initial_proportions=K_random_values/K_random_values.sum()#;print(initial_proportions)
            # then drawn with this vector 'not_trimmed_number' values in (1,2,...,K)
            drawn_values=np.random.choice(np.arange(K),p=initial_proportions,replace=True,size=not_trimmed_number)#;print(drawn_values)
            # finally output the number of 1s, of 2s, etc, in the vector 'drawn_values'
            res=np.bincount(drawn_values)#;print(res)
            # check occurence of "K-1" as integer in res
            length_res_OK=len(res)!=K # True if "K-1" were in res, then, res has got K components counting occurence of numbers from 0 to K-1
            # check that all integers between 0 and "K-1" has been drawn at least once
            not_all_were_drawn_OK=any(res==0)
            # if one of the boolean hereabove is True, forget the result and go through the while_loop once again
            crit = length_res_OK or not_all_were_drawn_OK
    return(res)


def init_clusters(x,dict_iter,dict_param):
    # this function outputs K centers and K covariances matrices
    # method used depends on which inputs were given
    # it also updates the parameter equal_weights, and the variable cluster_sizes and cluster_weights
    
    # change variable name for consiseness
    K=dict_param["K"]
    p=dict_param["p"]
    n=dict_param["n"]
    not_trimmed_number=dict_param["not_trimmed_number"]
    
    # fetch or compute first values of centers and sigma
    if np.any(dict_param["init_centers"])==None and np.any(dict_param["init_sigma"])==None:
        # if NO initial centers and NO covariance matrices have been given as input
        # then do like R_TClust : make K groups of p+1 points out of K*(p+1) points randomly drawn
        # and compute the emprical mean and the covariance matrix of each groups
        if n<=K*(p+1):
            raise Exception("there must be at least K*(p+1) point for the method to work")
        idx=np.transpose(np.random.choice(np.arange(n),replace=False,size=K*(p+1)).reshape(p+1,K))#;print(idx)
        Ksubsamples=x[idx,:]#;print(Ksubsamples)
        init_centers=Ksubsamples.mean(axis=1)#;print(init_centers)
        centers=init_centers#;print(dict_iter)
        sigma=np.zeros((K,p,p))
        for k in range(K):
            sigma[k,:,:]=np.cov(np.transpose(Ksubsamples[k,:,:]))
    elif np.any(dict_param["init_centers"])!=None and np.any(dict_param["init_sigma"])==None:
        # if only initial centers have been given as input
        # then take them and initialize covariance matrices as identity matrices
        centers=dict_param["init_centers"]
        sigma=np.tile(A=np.eye(p,p),reps=(K,1)).reshape(K,p,p)
    elif np.any(dict_param["init_centers"])==None and np.any(dict_param["init_sigma"])!=None:
        # if only covariance matrices have been given as input
        # then take them and drawn K points uniformly randomly
        idx=np.random.choice(np.arange(n),replace=False,size=K)#;print(idx)
        centers=x[idx,:]#;print(dict_iter["centers"])
        sigma=dict_param["init_sigma"]
    elif np.any(dict_param["init_centers"])!=None and np.any(dict_param["init_sigma"])!=None:
        centers=dict_param["init_centers"]
        sigma=dict_param["init_sigma"]
    
    # store in the dictionary dict_iter the centers and sigma
    dict_iter["centers"]=centers
    dict_iter["sigma"]  =sigma
    # update the cluster sizes depending on K and the boolean equal_wieghts
    # and update the weights of each cluster, used in the log likelihood
    if dict_param['equal_weights']:
        cluster_sizes=np.tile(A=not_trimmed_number/K,reps=K)
    else:
        cluster_sizes=get_init_cluster_size(dict_iter=dict_iter,dict_param=dict_param)
        
    dict_iter['cluster_sizes']=cluster_sizes
    dict_iter['cluster_weights']=cluster_sizes/not_trimmed_number
    return(dict_iter)


def division_by_zero(x,y):
    # this function divides itemwise x by y.
    # if 0 is encountered in y, then the corresponding division outputs 0
    
    # x and y must be numpy.ndarray and have the same length
    if type(x)!=type(np.array([0])):
        raise Exception("x must be an numpy.ndarray")
    if type(y)!=type(np.array([0])):
        raise Exception("y must be an numpy.ndarray")
    if len(x)!=len(y):
        raise Exception("x and y must have the same length")
        
    # change variable names
    zero_tol=dict_param["zero_tol"]
    y_loc=copy.deepcopy(y)      #;print(y_loc)
    
    mask_y_loc=y_loc<zero_tol   #;print(mask_y_loc)
    # change zeros in y_loc by 1 to use universel division of numpy
    y_loc[mask_y_loc]=1         #;print(y_loc)
    res=x/y_loc                 #;print(res)
    # as post-division task, switch to 0 the values in res where a division by zero would have occured
    res[mask_y_loc]=0           #;print(res)
    return(res)
# # unitary tests
# temp_x=np.array([0,1,2,3,4,5]);print(temp_x)
# # test the initial test
# temp_y=np.array([5,1,4,2,0])
# res_test=division_by_zero(x=temp_x,y=temp_y,dict_iter=dict_iter,dict_param=dict_param);print(res_test)
# # test the program
# temp_z=np.array([5,1,4,3,2,0])
# res_prog=division_by_zero(x=temp_x,y=temp_z,dict_iter=dict_iter,dict_param=dict_param);print(res_prog)


def estim_cluster_param(x,dict_iter,dict_param):
    # this function updates the values of empirical centers and covariance matrix
    # according to the z_ij values (assignment to cluster values)
    
    # change variable name
    n        =dict_param["n"]
    p        =dict_param["p"]
    zero_tol =dict_param["zero_tol"]
    z        =dict_iter["z"]
    c_sizes  =dict_iter["cluster_sizes"]
    
    # K may not be constant
    K        =len(dict_iter["cluster_sizes"])
    
    for k in range(K):
        if c_sizes[k]<=zero_tol:
            # in that case, the cluster is empty
            # one ignores the value of the center. It won't be used anymore.
            # update the covariance matrix
            dict_iter["sigma"][k,:,:]=np.zeros((p,p))
        else:
            # update empirical centers
            new_center_k=np.dot(np.transpose(z[:,k]),x)/c_sizes[k]
            dict_iter['centers'][k,:]=new_center_k
            # update empirical covariance matrices
            x_k_centered=x-new_center_k #;print(x_k_centered)
            # here I first divided by "c_sizes[k]-1" believing that it would give us the empirical covariance matrix, but "c_sizes[k]" is not comparable to n but rather to 0.5.
            # I corrected then the code to do exactly the same as in the R code
            new_sigma_k=np.dot(np.multiply(np.transpose(x_k_centered),z[:,k]),x_k_centered)/c_sizes[k] 
            dict_iter["sigma"][k,:,:]=new_sigma_k
    return(dict_iter)

def log_extended(arg): # we create a log function extended to 0 to simplify the code
    if arg == 0:
        image = -inf
    else:
        image = log(arg)
    return(image)

vec_log=np.vectorize(pyfunc=log_extended) # we create a vectorizable logarithm

def calc_fuzzy_row(z_i,dict_iter,dict_param):
    # This function computes exactly the same affectation row z_i_ as in the code R.
    # I did not understood though why they do what they do.
    # I spend then time to compute the two versions (R and python) step by step to check that they output the same result
    
    ## comment Edouard Genetay 20191011 : this function does the following calculation :
    ## from a row-vector z_i extracted from z where z_i = (z_i1,z_i2,...,z_iK)
    ## in fuzzy clustering, one return the row-vector v as follows
    ##                                       _                                  _
    ##                    1                 |                   ^[-m/(m-1)]      |
    ## v = ______________________________ . |  ..., [-log(z_ij)]            ,... |
    ##      __K                             |_                                  _|_{j from 1 to K}
    ##      \               ^[-m/(m-1)]
    ##      /_  [-log(z_ik)]
    ##      k=1
    ##
    ## That is indeed the quantity to compute when z_ik= r_j.f(x_i ; mu_j, sigma_j). But I don't know where the m comes from.
    ## That also explains why rows of z sums up to 1. That is not obvious in the code.
    ##
    ## in hard clustering case, it's simpler : v_i = 0 for all i except the one corresponding to the assigned cluster i0, v_i0=1
   
    # change variable names
    K        =dict_param["K"];K
    m        =dict_param["m"];m
    zero_tol =dict_param["zero_tol"];zero_tol
    z_i_     =copy.deepcopy(z_i);z_i_
    
    idx_max=z_i_.argmax()                                  # give the position of the greatest component of z_i_
    if z_i_[idx_max]>=1:
        res=(np.arange(K)==idx_max)+0                      # res is a p-tuple. '+0' converts boolean tuple in numerical one.
    else:        
        copy_z_i_=copy.deepcopy(z_i_)#;print(copy_z_i_)    # we make a copy of z_i_ that we call log_z_i_ because that is what we get at the end
        copy_z_i_[copy_z_i_<zero_tol]=1#;print(copy_z_i_)  # here we want to prevent math error (i.e. log(0)), so 0 is switch to 1 and then log(1)=0, these component will not interfere anymore
        log_z_i_=-vec_log(copy_z_i_)#;print(log_z_i_)      # desired result
        la3=np.tile(A=log_z_i_,reps=(K,1))#;print(la3)
        la2=np.transpose(la3)#;print(la2)
        z2_3d=division_by_zero(x=la3,y=la2)#;print(z2_3d)

        mat_modif=np.power(z2_3d,1/(m-1))#;print(mat_modif)
        z2_ij=mat_modif.sum(axis=0)#;print(z2_ij)

        array_final=division_by_zero(x=np.ones(K),y=z2_ij)#;print(array_final)
        res=array_final
    
    if res.sum()<=zero_tol:
        res=np.tile(A=1/K,reps=K)
    
    final_res=np.power(res,m)
    return(final_res)

# useful in the function find_clust_assignment and calcobj
def equivalent_dmvnorm_R(k,x,centers,sigma):
    # computes the values of the density function of a multinormal distribution of mean "mean" and covariance "cov" at the data points x
    res=multivariate_normal.pdf(x=x[:,:],mean=centers[k,:],cov=sigma[k,:,:],allow_singular=True)
    return(res)

def find_clust_assignment(x,dict_iter,dict_param):
    # This function update 4 quantities :
    # the z matrix : the matrix of the weighted likelihood to have been drawn from the k-th cluster
    # the assignment vector : whose i-th component is the cluster index assigned to the i-th data point at this iteration
    # the "code" : which is a boolean that is True only when z (resp assignment) remained the same from an iteration to another in the fuzzy case (resp non fuzzy case).
    # the cluster sizes : a vector with K components whose i-th component is the number of data finally affected to the i-th cluster

    n      =dict_param['n']
    K      =dict_param['K']
    fuzzy  =dict_param['fuzzy']
    trim   =dict_param['trimmed_number']
    eq_w   =dict_param['equal_weights']
    cw     =dict_iter['cluster_weights']
    centers=dict_iter['centers'] #;print(centers)
    sigma  =dict_iter['sigma']   #;print(sigma)

    # LIKELIHOOD
    # here one computes the values of each density function of the gaussian mixture at the data points x.
    # one wants to have the likelihood (ll) weighted by the mixting coefficient :
    # ll_{i,j} = f(x_i ; mu_j, sigma_j) which is the value of the density function of the j_th cluster at x_i
    # where f(.;mu_j,sigma_j) is the pdf of a multinormale distribution with mean mu_j and covariance matrix sigma_j
    # the weighted likelihood (wll) is then wll_{i,j} = r_j.f(x_i ; mu_j, sigma_j) where r_j is the probability for a point to be drawn according to f(.; mu_j, sigma_j)
    ll=np.transpose(np.array([equivalent_dmvnorm_R(k=k,x=x,centers=centers,sigma=sigma) for k in range(K)])) #; print("ll= "+str(ll[:10,:]))
    wll=np.multiply(ll,cw) #;print("wll="+str(wll[:10,:]))

    old_assign=dict_iter['assign'] #;print("old_assign="+str(old_assign))    # keep in memory the previous assignment vector
    old_z=dict_iter['z'] #;print("old_z="+str(old_z))       # keep in memory the previous assignment vector
    assignment=wll.argmax(axis=1)#;print(assignment)        # assignment according to the argmax of the wll
        
    if fuzzy > 0:
        # ASSIGNMENT
        # Here, the clustering is "Fuzzy". Then, z contains values in [0,1] and its rows sum up to 1.
        # It is the index of the greatest component of the wll
        
        # COMPUTATION FOR z (fuzzy clustering matrix)
        ## commented by Edouard Genetay 20191011 : this function does the following calculation :
        ## if wll is the weighted likelihood at the beginning of the step n, then the fuzzy clustering matrix z will
        ## be updated as follows :
        ##                                          _                                       _
        ##                    1                    |                        ^[-m/(m-1)]      |
        ## z = _________________________________ . |  ..., [-log(wll_{i,j})]            ,... |
        ##      __K                                |_                                       _|_{i from 1 to n / j from 1 to K}
        ##      \                    ^[-m/(m-1)]
        ##      /_  [-log(wll_{i,k})]
        ##      k=1
        ##
        ## I don't know what the m mean and where it comes from. According to the scientific article available at
        ## https://projecteuclid.org/download/pdfview_1/euclid.aos/1211819566 page n°1329, there should not be such a m.
        
        # the code lines hereunder do the same. it is just requiring the function calc_fuzzy_row if the boolean fuzzy>=2
        # or computing the desired quantity in the 'else' part if fuzzy<2 (i.e. when fuzzy=T, the 'else' part is computed)
        if fuzzy >=2:
            # a first way to compute the z array, that can be chosen with fuzzy>=2
            z=np.apply_along_axis(arr=wll,axis=1,func1d=calc_fuzzy_row,dict_iter=dict_iter,dict_param=dict_param)#;print(z) # a first way to compute the new z array
        else:
            # a second way to compute the z array
            # intanciation of 5 variables (one needs them to use masks)
            z=np.zeros((n,K))
            log_wll=copy.deepcopy(z)
            pre_z=copy.deepcopy(z)
            z2=copy.deepcopy(z)
            z1=copy.deepcopy(z)

            # search where z exceeds 1 and create masks
            max_wll=wll.max(axis=1)#;print(max_wll)
            max_wll_greater1=max_wll>=1#;print(max_wll_greater1)
            wll_bzo=((wll>0)*(wll<1))#;print(wll_bzo)

            # compute z1
            # z1 is the "z" matrix corresponding to the assignment in the Hard-clustering case without trimming
            # its rows corresponding to max_wll_greater1 will be used
            assig_=np.concatenate([np.arange(n).reshape(n,1),assignment.reshape(n,1)],axis=1)#;print(assig_)
            z1[np.transpose(assig_).tolist()]=1#;print(z1)

            #compute z2
            # z2 is the soft/fuzzy clustering matrix without trimming
            # its rows corresponding to not(max_wll_greater1) will be used
            log_wll[wll_bzo]=-vec_log(wll[wll_bzo])#;print(log_wll)
            la2=np.tile(A=log_wll,reps=(K,1,1))#;print(la2)
            la3=np.transpose(la2)#;print(la3)
            z2_3d=division_by_zero(la3,la2)#;print(z2_3d)
            z2 = np.transpose(np.power(z2_3d,1/(m-1)).sum(axis=2))#;print(z2)
            z2=division_by_zero(np.ones((n,K)),z2)#;print(z2)
            sum_z2=z2.sum(axis=1)#;print(sum_z2)
            z2[sum_z2<=zero_tol,:]=1/K#;print(z2)

            # merge the two quantities z1 and z2 in z through boolean vector max_wll_greater1
            pre_z[max_wll_greater1,:]=z1[max_wll_greater1,:]#;print(pre_z)
            pre_z[np.invert(max_wll_greater1),:]=z2[np.invert(max_wll_greater1),:]#;print(pre_z)
            z=np.power(pre_z,m)#;print(z)
        
        # TRIMMING
        # put trimming in the picture (dict_iter['assign'] and dict_iter['z'])
        log_wll_=copy.deepcopy(wll)
        
        log_wll_=vec_log(wll)
        log_wll_[z==0]=0;#print(log_wll_)
        disc=np.multiply(z,log_wll_).sum(axis=1)#;print(disc)
        idx_out=np.argsort(disc)[:trim]#;print(idx_out)
        assignment[idx_out]=-1#;print(assignment) # replace the assignment of the trimmed data points by -1
        z[idx_out,:]=0#;print(z)

        # Check equality between old_z and z
        code=all((old_z==z).flatten())#;print(code)
        c_sizes=z.sum(axis=0)#;print(c_sizes)

    else:
        # ASSIGNMENT
        # Here, the clustering is "Hard". Then, z contains only 0's and 1's and its rows still sum up to 1.
        # The assignment vector gives the position of the only 1 of the rows of z.
        
        # TRIMMING
        # 1) Then one wants to get the index of the data that will be trimmed "idx.out"
        mask=np.array([np.arange(n),assignment]).tolist()                        # mask of the entries of the matrix to change
        disc=wll[mask]#;print(disc)                                              # only take the greatest values a each row in ll (computationnal parcimony)
        idx_out=np.argsort(disc)[:trim]#;print(idx_out)
        # 2) setting trimmed data points to class 0
        assignment[idx_out]=-1#;print(assignment) # replace the assignment of the trimmed data points by -1
        
        # UPDATES
        # update the other parameters of the algorithm : code, csize, z_ij
        # does the assignment vector remain unchange? True or False
        code=all((old_assign==assignment).flatten())#;print(code)
        
        # build the z matrix that corresponds to the assignment vector
        z=np.zeros((n,K))#;print(z)
        pre_mask=np.array([np.arange(n),assignment])#;print("pre_mask="+str(pre_mask))  # mask of the entries of the matrix z to change to 1, one still has to get rid of the -1
        mask=pre_mask[:,pre_mask[1,:]>=0].tolist()#;print("mask="+str(mask))
        z[mask]=1#;print("z="+str(z))
        
        # count the number of data points per cluster
        c_sizes=np.sum(z,axis=0)#;print(c_sizes)  # there must be no -1 in 'assignment' to use np.bincount(), so we add 1. Counting the -1 is counting the 0. To be sure that c_sizes will always have the right length, we ask bincount to count up to K+1
    
    # compute new proportions of cluster among all data when it's worth it (i.e. when eq_w=equal_weights == False)
    if eq_w==False:
        cw=c_sizes/np.sum(c_sizes)#;print(cw)

    # update phase
    dict_iter_loc=copy.deepcopy(dict_iter)
    dict_iter_loc['z']=z
    dict_iter_loc['assign']=assignment
    dict_iter_loc['code']=code
    dict_iter_loc['cluster_sizes']=c_sizes #;print("c_sizes in findAssig="+str(c_sizes))
    dict_iter_loc['cluster_weights']=cw
    return(dict_iter_loc)

def calcobj(x,dict_iter,dict_param):
    # this function computes the objective criterion of TCLUST. According to the scientific article available at
    # https://projecteuclid.org/download/pdfview_1/euclid.aos/1211819566 page n°1329, given a probability measure P
    # (P=P_n empirical measure in practice) one wants to maximize the following expectation
    #          
    #                                _   _K                                                  _
    #                               |   \                                                     |
    #   Objective criterion = E_{P} |   /_   z_j(.) [ log(r_j) + log( f(.;mu_j,sigma_j) ) ]   |
    #                               |_  j=1                                                  _|
    #         
    #  where the notation of the article "z_j(.)" is related to our fuzzy clustering matrix z through :
    #  for all i, z_j(x_i)=z_ij
    # moreover, for those that are more used to sums notations, when P=P_{n} is an empirical distribution
    # one can rewrite E_{P} as a mean over the i varying from 1 to n. Up to a multiplicative constant (1/n),
    # one aims to maximize:
    #           _n   _K                     _n   _K
    #          \    \                      \    \
    #   obj =  /_   /_   z_ij log(r_j)  +  /_   /_  z_ij.log( f(x_i;mu_j,sigma_j) )
    #          i=1  j=1                    i=1  j=1
    #
    #          \_______  ___________/                         \_____   ________/
    #                  \/                                           \/
    #           they initialize obj with this                       w (notation used below)
    #           and then iteratively add other
    #           terms
    #
    # more rigorously, they initialize obj with the sum of the s_j*log(r_j) from j from 1 to K
    # where s_j is the size of the clusters. It is the same because s_j is the sum of the z_ij
    # over the i's. (It is written in the function find_clust_assignment : s_j=c_sizes=z.sum(axis=0))
    #
    
    # change variable name
    K=dict_param["K"]
    zero_tol=dict_param["zero_tol"]
    eq_w=dict_param['equal_weights']
    z=dict_iter["z"]
    c_sizes=dict_iter["cluster_sizes"]
    cw=dict_iter['cluster_weights']
    centers=dict_iter['centers']
    sigma=dict_iter['sigma']

    if eq_w:
        obj=0
    else:
        mask_nz=c_sizes > zero_tol # mask_nz stands for mask_non_zero
        obj=np.sum((c_sizes[mask_nz]*vec_log(cw[mask_nz])))
        
    for k in range(K):
        w=equivalent_dmvnorm_R(k=k,centers=centers,sigma=sigma,x=x)#;print(w)
        if z[:,k].sum()>zero_tol:
            if z[w<=0,k].sum()<=zero_tol:
                obj=obj+sum(z[w>0,k]*vec_log(w[w>0]))#;print(obj)
            else:
                obj=obj-inf
                dict_iter["obj"]=obj
                return(dict_iter) # we can now return a value, no need to do more loops
    
    dict_iter["obj"]=obj
    return(dict_iter)

def TreatSingularity(dict_iter,dict_param):
    raise Exception("After trimming, all points in the data set are concentrated in k subspaces.") ##  a single point is a subspace too.
    code=2  # indicating the data's concentration in either k subspaces or points.
    dict_iter['code']=code
    return(dict_iter)

def parsetclust_res(x,dict_iter,dict_param):
    # this function is used to give the right format to the output of the tclust algorithm
    # print(dict_iter)
    # change variable name
#     n        =dict_param["n"]
    K        =dict_param["K"]
#     p        =dict_param["p"]
#     zero_tol =dict_param["zero_tol"]
#     z        =dict_iter["z"]
    c_sizes  =dict_iter["cluster_sizes"]

    # INDEX sorted according decreasing order of the cluster sizes
    idx_clust=np.argsort(c_sizes)[::-1]#;print(idx_clust)
    idx_nz=c_sizes[idx_clust]!=0#;print(idx_nz)
    idx_clust=idx_clust[idx_nz]#;print(idx_clust)   # here are the indexes of the clusters whose csize is strictly positive
    # in python one can not affect values at unexisting position of an array but R does. that is why we do one more step than in R
    id_clust=np.arange(K+1)-1#;print(id_clust)
    id_clust[idx_clust+1]=np.arange(len(idx_clust))#;print(id_clust)

    ############ comment of Edouard GENETAY 21/10/2019:
    ############ The piece of code hereunder is in the R code but is unuseful. So a comment it. 
    ############ Note also that the argument 'x' in parsetclust_res is only used in that commented piece of code. 
    ### # ITERATIONS
    ### info={"iter_successful": 0,"iter_converged":0,"dim":x.shape}#;print(info)
    ############
    
    # RETURNABLE DICTIONARY
#     raise Exception("warning, in Parsetclust_Res the element 'par' of the dictionary is commented and ret has not got the status of class")
    ret={
        "centers": np.transpose(dict_iter['centers'][idx_clust,:])
        ,"cov" : dict_iter["sigma"][idx_clust,:,:]
        ,"cluster" : id_clust [dict_iter["assign"] + 1]
        ,"tclust_input" : dict_param
        ,"k" : len(idx_clust)
        ,"obj" : dict_iter["obj"]
        ,"size" : dict_iter['cluster_sizes'][idx_clust]
        ,"weights" : dict_iter['cluster_weights'][idx_clust]
        ,"ret_orig" : dict_iter
        #,"int" : info
    }#;print(ret)

    # The dictionary "ret" has got the status of " tclust class"
    return(ret)

def restr2_eigenv_preliminary_tests(eigenvalues,c_sizes):
    # checks for right structure of autovalues itself and comparatively to c_sizes
    if not(type(eigenvalues) == type(np.array([0,0]))):
        raise Exception("in restr2_eigenv 'autovalues' must be a numpy.array. here autovalues is unfortunately not a numpy.array")
    elif not(len(eigenvalues.shape) == 2):
        raise Exception("in restr2_eigenv 'autovalues' must be a 2d numpy.array. here autovalues is unfortunately not a 2d numpy.array")
    elif not(all(np.array(eigenvalues.shape) > 0)):
        raise Exception("in restr2_eigenv 'autovalues' must be a 2d numpy.array with all dimensions greater than 1. here dimensions constrain is not verified")
#     elif not(eigenvalues.shape[1]==len(c_sizes)):
#             raise Exception("in restr2_eigenv 'autovalues' must have as many colomns as the number of clusters")

def restr2_eigenv(eigenvalues,dict_param,dict_iter):
    # change variable names
    ev=eigenvalues#;print("ev="+str(ev))
    restr_fact=dict_param['restr_fact']#;print('restr_fact='+str(restr_fact))
    zero_tol=dict_param['zero_tol']#;print('zero_tol='+str(zero_tol))
    c_sizes=dict_iter['cluster_sizes']#;print("c_sizes="+str(c_sizes))

    # check for right structure of autovalues
    restr2_eigenv_preliminary_tests(ev,c_sizes)

    dd=np.transpose(ev)#;print("dd="+str(dd))
    p,K=ev.shape#;print("p="+str(p)+"  K="+str(K))
    nn=np.sum(c_sizes)#;print("nn="+str(nn))
    nis=np.transpose(np.tile(A=c_sizes,reps=(p,1)))#;print(nis)
    idx_nis_gr_0 = nis > zero_tol#;print("idx_nis_gr_0="+str(idx_nis_gr_0))
    #print("c_sizes="+str(c_sizes))
    used_ev = c_sizes > zero_tol#;print("used_ev="+str(used_ev))
    #print("ev="+str(ev))
    ev_nz = ev[:,used_ev]#;print("ev_nz="+str(ev_nz))

    if ev_nz.max() <= zero_tol:
        return (np.zeros((p,K)))

    # we check if the  eigenvalues verify the restrictions if True then eigenvalues are ok, return them
    # the eigenvalues that are not used are affected to be the mean value of the 'used_ev' values.
    # Why?? and why shouldn't it be ev_nz.min() (better algorithmic complexity)
    if ev_nz.min()>zero_tol:
#         print("track 2.1")
        if ev_nz.max() / ev_nz.min() <= restr_fact:
#             print("track 2.2")
            ev[:,np.invert(used_ev)] = ev_nz.mean()#;print("ev="+str(ev))
            return (ev)

    # print()
    # comments of R :
    # d_ is the ordered set of values in which the restriction objective function change the definition
    # points in d_ correspond to  the frontiers for the intervals in which this objective function has the same definition
    # ed is a set with the middle points of these intervals
    d_= np.sort(np.array([ev,ev/restr_fact]).flatten())#;print("d_="+str(d_))
    dim = len(d_)#;print("dim="+str(dim))
    d_1 = np.array(d_.tolist()+[d_[-1]*2])#;print("d_1="+str(d_1)) #here, d_1 is the array d_ with 1 spare component at the end that is the doble of its last component
    d_2 = np.array([0]+d_.tolist())#;print("d_2="+str(d_2)) #here, d_1 is the array d_ with 1 spare component at the end that is the doble of its last component
    ed = (d_1+d_2)/2#;print("ed="+str(ed))
    dim = dim+1#;print("dim="+str(dim))

    # comments of R :
    # the only relevant eigenvalues are those belong to a clusters with sample size greater than 0.
    # eigenvalues corresponding to a clusters with 0 individuals has no influence in the objective function
    # if all the eigenvalues are 0 during the smart initialization we assign to all the eigenvalues the value 1
    # we build the sol array
    # sol[1],sol[2],.... this array contains the critical values of the interval functions which defines the m objective function
    # we use the centers of the interval to get a definition for the function in each interval
    # this set with the critical values (in the array sol) contains the optimum m value

    # instanciation of 5 variables
    rr = np.zeros((K,dim))#;print("rr="+str(rr))
    ss = np.zeros((K,dim))#;print("ss="+str(ss))
    tt = np.zeros((K,dim))#;print("tt="+str(tt))
    sol = np.zeros((dim))#;print("sol="+str(sol))
    sal = np.zeros((dim))#;print("sal="+str(sal))

    for mp in range(dim):
        # values of rr, ss and tt
        for i in range(K):
            rr[i,mp] = np.sum((dd[i,] < ed[mp])) + sum((dd[i,:] > ed[mp]*restr_fact))#;print("rr="+str(rr))
            ss[i,mp] = np.sum(dd[i,]*(dd[i,:] < ed[mp]))#;print("ss="+str(ss))
            tt[i,mp] = np.sum(dd[i,]*(dd[i,:] > ed[mp] * restr_fact))#;print("tt="+str(tt))
        # sol    
        sol[mp]=np.sum(c_sizes/nn*(ss[:,mp]+tt[:,mp]/restr_fact))/(np.sum(c_sizes/nn*(rr[:,mp])))#;print("sol="+str(sol))
        # e and o
        e=sol[mp]*(dd<sol[mp])+dd*(dd>=sol[mp])*(dd<=restr_fact*sol[mp])+(restr_fact*sol[mp])*(dd>restr_fact*sol[mp])#;print("e="+str(e))
        o = -1/2* nis / nn *(vec_log(e)+dd/e)#;print("o="+str(o))
        # sal
        sal[mp]=np.sum(o)#;print("sal="+str(sal))

    # comments of R :
    # mm is the optimum value for the eigenvalues procedure
    mm = sol[sal.argmax()]#;print("mm="+str(mm))
    # based on the mm value we get the restricted eigenvalues

    ret_restr2_eigenv=np.transpose(mm*(dd<mm)+dd*(dd>=mm)*(dd<=restr_fact*mm)+(restr_fact*mm)*(dd>restr_fact*mm))#;print("ret_restr2_eigenv="+str(ret_restr2_eigenv)) ## the return value
    return(ret_restr2_eigenv)

def restr_diffax(dict_iter,dict_param,f_restr_eigen = restr2_eigenv):
    # change variable names
    p=dict_param["p"]#;print("p="+str(p))
    K=dict_param["K"]#;print("K="+str(K))
    restr_fact=dict_param['restr_fact']#;print('restr_fact='+str(restr_fact))
    zero_tol=dict_param['zero_tol']#;print('zero_tol='+str(zero_tol))
    sigma=dict_iter['sigma']#;print("sigma="+str(sigma))
    c_sizes=dict_iter['cluster_sizes']#;print("c_sizes="+str(c_sizes))

    # my python TClust equivalent does not enables all cases that the R code does.
    # This test prevents using this python code in issuing cases
    if p==1:
        raise Exception("the R code runs well when p==1 but it is not yet available in the python code")

    # there were these tests in the R code. It could be simpler to understand with the same performances
    if p==1:
        print("track 1.1")
        if restr_fact == 1:
            print("track 1.2")
            True
    #       # all variances are imposed to be equal -> use the simpler restr.avgcov function instead
    #       return (restr.avgcov (iter, pa))
    else:
        # else - if p == 1 always use the simpler eigen - restriction 
        restr_deter = False

    u=np.array([0.]*K*p*p).reshape((K,p,p))#;print("u="+str(u))
    d=np.array([0.]*K*p).reshape((p,K))#;print("d="+str(d))
    for k in range(K):
        dk,uk = LA.eig(sigma[k,:,:])
        # a small difference between the two codes appears here.
        # Indeed, numpy.linalg.eig does not outputs eigenvalues in the same order as the 'eigen' function in R.
        # (the eigen R function outputs the eigenvalues in the absolute value decreasing order,
        # if eigenvalues are complex, they are ordered in module decreasing order)
        # maybe I could do without ordering but it is easier for me to compare the outputs of the two codes.
        decreasing_sorting_map=np.argsort(dk)[::-1]
        u[k,:,:] = uk[:,decreasing_sorting_map]
        d[:,k] = dk[decreasing_sorting_map]
    d [d < 0] = 0 #;print("d>=0  =",d)   ## all eigenvalue < 0 are restricted to 0 BECAUSE: min (d) < 0 -> max (d) / min (d) < 0 which would always fit our eigenvalue - criterion!

    ret_restr2_eigenv = f_restr_eigen (eigenvalues=d,dict_iter=dict_iter,dict_param=dict_param) #;ret_restr2_eigenv
    #print("ret_restr2_eigenv="+str(ret_restr2_eigenv))
    
    ## checking for singularity in all clusters. True if ret_restr2_eigenv is zero-matrix
    code = np.max(ret_restr2_eigenv) > zero_tol#;print("code=",code)

    if not code:
#         print("track 2")
#         print(dict_iter)
        return(dict_iter)

    for k in range(K):
        ## re-composing the sigmas through formula : now cov = v.Diag.t(v), where v is an orthonormal matrix
        v=u[k,:,:]
        Diag_new_ev=np.diag (ret_restr2_eigenv[:,k]) # [::-1] <-(adjustment tests) # if one does not reverse the positions of the egenvalue it does not work... I don't catch that yet
        t_v=np.transpose(v)
        # store the matrix product in sigma
        sigma[k,:,:] = v.dot(Diag_new_ev).dot(t_v)#;print("sigma[:,:,k]=",sigma[k,:,:])

    # UPDATE
    dict_iter['sigma']=sigma
    dict_iter['code']=code
    return(dict_iter)

def test_format_X_tclust(X):
    df_test=pd.DataFrame(np.eye(5))
    if type(X)==type(df_test):
        x = X.values
        return(x)
    elif type(X)!=type(np.array([0])):
        raise Exception("in tclust 'x' must be a numpy.array")
    elif not(len(X.shape)==2):
        raise Exception("in tclust 'x' must be a 2d numpy.array. here 'x' is unfortunately not a 2d numpy.array")
    elif not(all(np.array(X.shape)>0)):
        raise Exception("in tclust 'x' must be a 2d numpy.array with all dimensions greater than 1. here dimensions constrain is not verified")
    elif not (X.dtype==np.array([2.]).dtype or X.dtype==np.array([2]).dtype):
        raise Exception("in tclut 'x' must contain floats or integers")
    x=X
    return(x)

def tclust_python_v1(x, # data
                  k=3,alpha = 0.05,restr_fact=12,m = 5,equal_weights = False,fuzzy = False, # necessary arguments for computation
                  init_centers=None,init_sigma=None,init_cluster_size=None,init_cluster_weights=None, # control of the initialization
                  nstart = 1,iter_max = 20,zero_tol = 1e-16,store_x = True,trace = None, # optional arguments to control some parameters, output and loops
                  f_restr = restr_diffax,restr_deter=None,f_hook_iter=None,f_hook_model=None # intern functions that could be used. Only f_restr was compulsory to implement to run tclust
                 ):
    
    # test for x
    x=test_format_X_tclust(X=x)#;print("x="+str(x))
    # test on variable that are not yet supported in this version
    if trace!=None:
        raise Exception("please, do not give a value for the variable 'trace', this version does not support it yet")
    if restr_deter!=None:
        raise Exception("please, do not give a value for the variable 'restr_deter', this version does not support it yet")
    if f_hook_iter!=None:
        raise Exception("please, do not give a value for the variable 'f_hook_iter', this version does not support it yet")
    if f_hook_model!=None:
        raise Exception("please, do not give a value for the variable 'f_hook_model', this version does not support it yet")    

    n,p=x.shape#;print("n="+str(n));print("p="+str(p))
    trimmed_proportion=alpha#;print("trimmed_proportion="+str(trimmed_proportion))
    not_trimmed_number=floor(n*(1-trimmed_proportion))#;print("not_trimmed_number="+str(not_trimmed_number))
    trimmed_number=n-not_trimmed_number#;print("trimmed_number="+str(trimmed_number))

    # The two transversal dictionaries
    dict_iter={"obj":- inf
               ,'nb_iter':None
               ,'assign':None
               ,"cluster_sizes":None
               ,'cluster_weights':None
               ,"sigma":None
               ,"centers":None
               ,"code":None
               ,"z":None
              }
    dict_param={"n":n
                ,"p":p
                ,"K":k
                ,'trimmed_proportion':trimmed_proportion
                ,"alpha":alpha # I duplicate the names for the variable "trimmed_proportion" because the code R names it "alpha"
                ,"m":m 
                ,"trimmed_number":trimmed_number
                ,"not_trimmed_number":not_trimmed_number
                ,'fuzzy':fuzzy
                ,"equal_weights":equal_weights
                ,"zero_tol":zero_tol
                ,"restr_deter":restr_deter
                ,"restr_fact":restr_fact
                ,"nstart":nstart
                ,"iter_max":iter_max
                ,"trace":trace
                ,"store_x":store_x
                ,"init_centers":init_centers
                ,"init_sigma":init_sigma
                ,"f_restr":f_restr
                ,"f_hook_iter":f_hook_iter
                ,"f_hook_model":f_hook_model
               }

    if store_x==True:
        dict_param["x"]=x

    # will be used to store the best iteration among the nstart ones
    best_dict_iter = dict_iter
    array_nb_iter_pro_start=np.array([0]*nstart)
    dict_iter['idx_best_start']=None
    
    for j in range(nstart):
#         print("j="+str(j))
        dict_iter['idx_best_start']=j
        dict_iter = init_clusters(x=x, dict_iter=dict_iter, dict_param=dict_param)#;print(dict_iter)

        # this part of the code is not supported : f_hook_iter is not available.
        # One cannot pass the test because Exception tests were lead at the beginning
        if f_hook_iter!=None:
            f_hook_iter(x,find_clust_assignment(dict_iter=dict_iter,dict_param=dict_param),dict_param=dict_param,which_iter=j,constant=0)

        lastobj=-inf
        for i in range(iter_max):
            # print("i="+str(i))
            # if i==0:
            dict_iter=f_restr(dict_iter=dict_iter,dict_param=dict_param)#;print(dict_iter)

            if not dict_iter["code"]:
                # all eigenvalues are zero or a singularity has been detected..
                if i==0:
                    to_return = Parsetclust.Res (x, TreatSingularity (calcobj (x, iter, pa), pa), dict_param)
                    return to_return
                else:
                    for k in range(dict_param['K']):
                        dict_iter['sigma'][k,:,:]=np.eye(M=p,N=p)

            dict_iter = find_clust_assignment(x=x, dict_iter=dict_iter, dict_param=dict_param) ## finding the cluster assignment on behalf of the current sigma & center information
            #print("dict_iter="+str(dict_iter))

            # this part of the code is not supported : f_hook_iter is not available.
            # One cannot pass the test because Exception tests were lead at the beginning
            if i==0 and f_hook_iter!=None:
                # print("path 2.1")
                f_hook_iter (x, dict_iter, dict_param, dict_param, j, i)

            if dict_iter["code"] or i == iter_max:
                # print("i="+str(i))
                array_nb_iter_pro_start[j]=i+1 # one is about to break the loop, one store how many iteration were needed
                # print("stored="+str(array_nb_iter_pro_start[j]))
                break # break the for-loop. return the result and don't re-estimate cluster parameters this time

            dict_iter=estim_cluster_param(x=x,dict_iter=dict_iter,dict_param=dict_param)
            # print("center in tclust_py="+str(dict_iter['centers']))

            #############################################
            ## Here should be inserted a piece of code that would be executed if "trace" is True.
            ## in this version, "trace" is not supported
            #############################################

        dict_iter = calcobj(x=x, dict_iter=dict_iter, dict_param=dict_param)
        # print("obj="+str(dict_iter['obj']))

        dict_iter['code']= (i == iter_max-1) # the "-1" is here to fit with the fact that range(iter_max) ranges from 0 to iter_max-1

        if f_hook_model!=None:
            f_hook_model (x, dict_iter, dict_param, dict_param, j)

        if j == 0 or dict_iter["obj"] > best_dict_iter["obj"]:
            # print("best_iter")
            best_dict_iter = dict_iter
            # print(best_dict_iter['obj'])
            # print(best_dict_iter['sigma'])
    
    best_dict_iter['nb_iter_pro_start']=array_nb_iter_pro_start
    to_return=parsetclust_res(x=x,dict_iter=best_dict_iter,dict_param=dict_param);to_return
    return(to_return)