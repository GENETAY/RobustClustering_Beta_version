from sklearn.neighbors import LocalOutlierFactor
import numpy as np
from scipy.spatial.distance import cdist

def dist_to_t(XA,rr):
    '''
    XA = np.array 1D
    rr = np.array 1D
    to fit the shape that arguments will have when using np.apply_along_axis
    '''
    XAA=XA.reshape(1,-1) # convert to 2D array to fit cdist() format
    rrr=rr.reshape(1,-1) # convert to 2D array to fit cdist() format
    res=cdist(XA=XAA,XB=rrr)[0]
    return res

def dist_to_centers(vector,centers):
    '''
    vector  = 1D array
    centers = 1D or 2D array
    returns a 1D array with all distances between the vector and the centers
    '''
    res=np.apply_along_axis(arr=centers,axis=1,func1d=dist_to_t,rr=vector).reshape(1,-1)[0]
    return res

def dist_vec_to_centers(X,centers):
    '''
    X       = 1D or 2D array
    centers = 1D or 2D array
    returns a 1D or 2D array with all distances between each rows of X and the centers
    the rows of the output corresponds to rows of X and column to the centers
    '''
    res=np.apply_along_axis(arr=X,axis=1,func1d=dist_to_centers,centers=centers)
    return(res)

def ROBIN( X,k,
            method="optimum",epsilon=0.05,determinist=True,
            n_neighbors=20, algorithm='auto', leaf_size=30,
            metric='minkowski', p=2, metric_params=None,
            contamination="legacy", novelty=False, n_jobs=None):
    
    if not(any(method==np.array(["optimum","minimum","approx"]))):
        print("'method' should be either 'optimum', 'minimum' or 'approx', default 'optimum'. When 'approx' is chosen, give a value for epsilon, default 0.05")
    
    
    # initialization of variables
    n,p=X.shape
    if determinist==True:
        origine=np.tile(A=0,reps=p).reshape(1,p);origine
#         centers=np.array([[]]).reshape(0,p).shape
        centers_index=[] # we will return the indexes for more convenience
        m=0;m
    else:
        index_first_center=np.random.choice(np.arange(n),size=1,replace=False)[0]
        centers_index=[index_first_center] # we will return the indexes for more convenience
        centers=X[index_first_center,:].reshape(1,p)
        m=1;m
        
    
    # compute the LOF as first step
    init__=LocalOutlierFactor(n_neighbors=n_neighbors, algorithm=algorithm, leaf_size=leaf_size,
                     metric=metric, p=p, metric_params=metric_params,
                     contamination=contamination, novelty=novelty, n_jobs=n_jobs);init__
    LOF_algo_run=init__.fit(X);LOF_algo_run
    LOF_values=-LOF_algo_run.negative_outlier_factor_
    # print("LOF_values="+str(LOF_values))
    
    # search then for k centers
    while m<k:
        #print("m=",m)
        # compute distances to centers except when there is no center yet
        if m==0:
            array_distance=dist_vec_to_centers(X=X,centers=origine).flatten();#print("path 1.1:")
        else:
            if m==1:
                array_distance=dist_vec_to_centers(X=X,centers=centers).flatten();#print("path 1.2:")
            if m >1:
                array_distance=dist_vec_to_centers(X=X,centers=centers).min(axis=1);#print("path 1.3:")
        # find the index order that would sort the distance array
        order=np.flip(m=np.argsort(array_distance),axis=0);order
        
        # search for the next center
        if method=='optimum':
            # Looks for the LOF that is the nearest to 1: one transforms the LOF_values with x->(x-1)^2 and then searchs the min
            position_ordered_LOF_next_nearest_to_1=np.argsort(((LOF_values[order]-1)**2))[m];position_ordered_LOF_next_nearest_to_1
            position_LOF_next_nearest_to_1=order[position_ordered_LOF_next_nearest_to_1];position_LOF_next_nearest_to_1
            centers_index.append(position_LOF_next_nearest_to_1);centers_index
            point_LOF_next_nearest_to_1=X[position_LOF_next_nearest_to_1,:];point_LOF_next_nearest_to_1
            if m==0:
                centers=point_LOF_next_nearest_to_1.reshape(1,p);#print("path 2.1 : ",centers)
            else:
                centers=np.concatenate([centers,point_LOF_next_nearest_to_1.reshape(1,p)]);#print("path 2.2 : ",centers)
        elif method=='minimum':
            # Looks for the LOF that is minimum
            position_ordered_LOF_next_min=np.argsort(LOF_values[order])[m];position_ordered_LOF_next_min
            position_LOF_next_min=order[position_ordered_LOF_next_min];position_LOF_next_min
            centers_index.append(position_LOF_next_min)
            point_LOF_next_min=X[position_LOF_next_min,:];point_LOF_next_min
            if m==0:
                centers=point_LOF_next_min.reshape(1,p)
            else:
                centers=np.concatenate([centers,point_LOF_next_min.reshape(1,p)])
        elif method=="approx":
            # Looks for the first LOF value that falls in ]1-eps,1+eps[
            i=0
            stopping_crit=True
            while stopping_crit:
                if i<n:
                    # it is indeed possible that one has pointed at all LOF values while centers are still missing.
                    # this situation means that the intervalle ]1-eps,1+eps[ contains too few points to be centers.
#                     print(i,end='')
                    if abs(LOF_values[order[i]]-1)<epsilon and not(any(order[i]==centers_index)):
                        centers_index.append(order[i])
                        if m==0:
                                centers=X[order[i],:].reshape(1,p)
                        else:
                                centers=np.concatenate([centers,X[order[i],:].reshape(1,p)])
                        stopping_crit=False
                    else:
                        i=i+1
                else:
                    # all values of LOF have been pointed by the loop and it still misses centers then one uses the 'optimum' method to complet centers 
                    stopping_crit=False
                    method="optimum"
        else:
            print("warnings:: unvalid value given for 'method'")
    
        # we now have m+1 centers, we need to increment m
        m=m+1;m
#         print()
    res=np.array(centers_index)
    return res