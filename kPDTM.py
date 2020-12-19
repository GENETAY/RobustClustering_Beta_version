import numpy as np
from sklearn.neighbors import KDTree
from scipy.spatial import distance # For the Mahalanobis distance

#cf: https://luzine.lumenai.fr/notebooks/Test_Demo/MOMK/Notebook_kPDTM_kPLM.ipynb
# modified by edouard on the 2019/10/29

def DTM(X,query_pts,q):
    '''
    Compute the values of the DTM of the point cloud X
    Require sklearn.neighbors.KDTree to search nearest neighbors
    
    Input:
    X: a nxd numpy array representing n points in R^d
    query_pts:  a sxd numpy array of query points
    q: parameter of the DTM in {1,2,...,n}
    
    Output: 
    DTM_result: a sx1 numpy array contaning the DTM of the 
    query points
    
    Example:
    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    Q = np.array([[0,0],[5,5]])
    DTM_values = DTM(X, Q, 3)
    '''
    n = X.shape[0]     
    if(q>0 and q<=n):
        kdt = KDTree(X, leaf_size=30, metric='euclidean')
        NN_Dist, NN = kdt.query(query_pts, q, return_distance=True)
        DTM_result = np.sqrt(np.sum(NN_Dist*NN_Dist,axis=1) / q)
    else:
        raise AssertionError("Error: q should be in {1,2,...,n}")
    return(DTM_result)

def mean_var(X,x,q,kdt):
    '''
    An auxiliary function.
    
    Input:
    X: an nxd numpy array representing n points in R^d
    x: an sxd numpy array representing s points, 
        for each of these points we compute the mean and variance of the nearest neighbors in X
    q: parameter of the DTM in {1,2,...,n} - number of nearest neighbors to consider
    kdt: a KDtree obtained from X via the expression KDTree(X, leaf_size=30, metric='euclidean')
    
    Output:
    Mean: an sxd numpy array containing the means of nearest neighbors
    Var: an sx1 numpy array containing the variances of nearest neighbors
    
    Example:
    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    x = np.array([[2,3],[0,0]])
    kdt = KDTree(X, leaf_size=30, metric='euclidean')
    Mean, Var = mean_var(X,x,2,kdt)
    '''
    NN = kdt.query(x, q, return_distance=False)
    Mean = np.zeros((x.shape[0],x.shape[1]))
    Var = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        Mean[i,:] = np.mean(X[NN[i],:],axis = 0)
        Var[i] = np.mean(np.sum((X[NN[i],:] - Mean[i,:])*(X[NN[i],:] - Mean[i,:]),axis = 1))
    return Mean, Var

import random # For the random centers from which the algorithm starts

# modifyed by Edouard on the 2019/10/29 to be able to initialise the centers
def optima_for_kPDTM(X,q,k,sig,iter_max = 10,nstart = 1,initial_centers=None):
    '''
    Compute local optimal centers for the k-PDTM-criterion $R$ for the point cloud X
    Require sklearn.neighbors.KDTree to search nearest neighbors
    
    Input:
    X: an nxd numpy array representing n points in R^d
    query_pts:  an sxd numpy array of query points
    q: parameter of the DTM in {1,2,...,n}
    k: number of centers
    sig: number of sample points that the algorithm keeps (the other ones are considered as outliers -- cf section "Detecting outliers")
    iter_max : maximum number of iterations for the optimisation algorithm
    nstart : number of starts for the optimisation algorithm
    initial_centers: k*d numpy array. algorithm starts with these centers instead of initialize them at random.
    
    Output: 
    centers: a kxd numpy array contaning the optimal centers c^*_i computed by the algorithm
    means: a kxd numpy array containing the local centers m(c^*_i,\mathbb X,q)
    variances: a kx1 numpy array containing the local variances v(c^*_i,\mathbb X,q)
    colors: a size n numpy array containing the colors of the sample points in X
        points in the same weighted Voronoi cell (with centers in opt_means and weights in opt_variances)
        have the same color
    cost: the mean, for the "sig" points X[j,] considered as signal, of their smallest weighted distance to a center in "centers"
        that is, min_i\|X[j,]-means[i,]\|^2+variances[i].
        
    
    Example:
    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    sig = X.shape[0] # There is no trimming, all sample points are assigned to a cluster
    centers, means, variances, colors, cost = optima_for_kPDTM(X, 3, 2, sig)
    '''
    n = X.shape[0]
    d = X.shape[1]
    opt_cost = np.inf
    opt_centers = np.zeros([k,d])
    opt_colors = np.zeros(n)
    opt_kept_centers = np.zeros(k)
    if(q<=0 or q>n):
        raise AssertionError("Error: q should be in {1,2,...,n}")
    elif(k<=0 or k>n):
        raise AssertionError("Error: k should be in {1,2,...,n}")
    else:
        kdt = KDTree(X, leaf_size=30, metric='euclidean')
        for starts in range(nstart):
            
            # Initialisation
            colors = np.zeros(n)
            min_distance = np.zeros(n) # Weighted distance between a point and its nearest center
            kept_centers = np.ones((k), dtype=bool)
            # instance of initial centers, whether given or not.
            if initial_centers is None:
                # code of brecheteau and tinarrage
                first_centers_ind = random.sample(range(n), k) # Indices of the centers from which the algorithm starts
                centers = X[first_centers_ind,:]
            else:
                # edouard's modification
                centers = initial_centers
                
            old_centers = np.ones([k,d])*np.inf
            mv = mean_var(X,centers,q,kdt)
            Nstep = 1
            while((np.sum(old_centers!=centers)>0) and (Nstep <= iter_max)):
                Nstep = Nstep + 1
                
                # Step 1: Update colors and min_distance
                for j in range(n):
                    cost = np.inf
                    best_ind = 0
                    for i in range(k):
                        if(kept_centers[i]):
                            newcost = np.sum((X[j,:] - mv[0][i,:])*(X[j,:] - mv[0][i,:])) + mv[1][i]
                            if(newcost < cost):
                                cost = newcost
                                best_ind = i
                    colors[j] = best_ind
                    min_distance[j] = cost
                    
                # Step 2: Trimming step - Put color -1 to the (n-sig) points with largest cost
                index = np.argsort(-min_distance)
                colors[index[range(n-sig)]] = -1
                ds = min_distance[index[range(n-sig,n)]]
                costt = np.mean(ds)
                
                # Step 3: Update Centers and mv
                old_centers = np.copy(centers)
                old_mv = mv
                for i in range(k):
                    pointcloud_size = np.sum(colors == i)
                    if(pointcloud_size>=1):
                        centers[i,] = np.mean(X[colors==i,],axis = 0)
                    else:
                        kept_centers[i] = False
                mv = mean_var(X,centers,q,kdt)
                
            if(costt <= opt_cost):
                opt_cost = costt
                opt_centers = np.copy(old_centers)
                opt_mv = old_mv
                opt_colors = np.copy(colors)
                opt_kept_centers = np.copy(kept_centers)
                
        centers = opt_centers[opt_kept_centers,]
        means = opt_mv[0][opt_kept_centers,]
        variances = opt_mv[1][opt_kept_centers]
        colors = np.zeros(n)
        for i in range(n):
            colors[i] = np.sum(opt_kept_centers[range(int(opt_colors[i]+1))])-1
        cost = opt_cost
        
    return(centers, means, variances, colors, cost)


def kPDTM(X,query_pts,q,k,sig,iter_max = 10,nstart = 1,initial_centers=None):
    '''
    Compute the values of the k-PDTM of the empirical measure of a point cloud X
    Require sklearn.neighbors.KDTree to search nearest neighbors
    
    Input:
    X: a nxd numpy array representing n points in R^d
    query_pts:  a sxd numpy array of query points
    q: parameter of the DTM in {1,2,...,n}
    k: number of centers
    sig: number of points considered as signal in the sample (other signal points are trimmed)
    initial_centers: k*d numpy array. algorithm starts with these centers instead of initialize them at random.
    
    Output: 
    kPDTM_result: a sx1 numpy array contaning the kPDTM of the 
    query points
    
    Example:
    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    Q = np.array([[0,0],[5,5]])
    kPDTM_values = kPDTM(X, Q, 3, 2,X.shape[0])
    '''
    n = X.shape[0]     
    if(q<=0 or q>n):
        raise AssertionError("Error: q should be in {1,2,...,n}")
    elif(k<=0 or k>n):
        raise AssertionError("Error: k should be in {1,2,...,n}")
    elif(X.shape[1]!=query_pts.shape[1]):
        raise AssertionError("Error: X and query_pts should contain points with the same number of coordinates.")
    else:
        centers, means, variances, colors, cost = optima_for_kPDTM(X,q,k,sig,iter_max = iter_max,nstart = nstart,
                                                                   initial_centers=initial_centers)
        kPDTM_result = np.zeros(query_pts.shape[0])
        for i in range(query_pts.shape[0]):
            kPDTM_result[i] = np.inf
            for j in range(means.shape[0]):
                aux = np.sqrt(np.sum((query_pts[i,]-means[j,])*(query_pts[i,]-means[j,]))+variances[j])
                if(aux<kPDTM_result[i]):
                    kPDTM_result[i] = aux 
                    
    return(kPDTM_result, centers, means, variances, colors, cost)

def optima_for_kPLM(X,q,k,sig,iter_max = 10,nstart = 1,initial_centers=None):
    '''
    Compute local optimal centers and matrices for the k-PLM-criterion $R'$ for the point cloud X    
    Input:
    X: an nxd numpy array representing n points in R^d
    query_pts:  an sxd numpy array of query points
    q: parameter of the DTM in {1,2,...,n}
    k: number of centers
    sig: number of sample points that the algorithm keeps (the other ones are considered as outliers -- cf section "Detecting outliers")
    iter_max : maximum number of iterations for the optimisation algorithm
    nstart : number of starts for the optimisation algorithm
    initial_centers: k*d numpy array. algorithm starts with these centers instead of initialize them at random.
    
    Output: 
    centers: a kxd numpy array contaning the optimal centers c^*_i computed by the algorithm
    Sigma: a list of dxd numpy arrays containing the covariance matrices associated to the centers
    means: a kxd numpy array containing the centers of ellipses that are the sublevels sets of the k-PLM
    weights: a size k numpy array containing the weights associated to the means
    colors: a size n numpy array containing the colors of the sample points in X
        points in the same weighted Voronoi cell (with centers in means and weights in weights)
        have the same color    
    cost: the mean, for the "sig" points X[j,] considered as signal, of their smallest weighted distance to a center in "centers"
        that is, min_i\|X[j,]-means[i,]\|_{Sigma[i]^(-1)}^2+weights[i].         
    
    Example:
    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    sig = X.shape[0] # There is no trimming, all sample points are assigned to a cluster
    centers, Sigma, means, weights, colors, cost = optima_for_kPLM(X, 3, 2, sig)
    '''
    n = X.shape[0]
    d = X.shape[1]
    opt_cost = np.inf
    opt_centers = np.zeros([k,d])
    opt_Sigma = []
    opt_means = np.zeros([k,d])
    opt_weights = np.zeros(k)
    opt_colors = np.zeros(n)
    opt_kept_centers = np.zeros(k)
    if(q<=0 or q>n):
        raise AssertionError("Error: q should be in {1,2,...,n}")
    elif(k<=0 or k>n):
        raise AssertionError("Error: k should be in {1,2,...,n}")
    else:
        #kdt = KDTree(X, leaf_size=30, metric='euclidean')
        for starts in range(nstart):
            
            # Initialisation
            colors = np.zeros(n)
            kept_centers = np.ones((k), dtype=bool)
            # instance of initial centers, whether given or not.
            if initial_centers is None:
                # code of brecheteau and tinarrage
                first_centers_ind = random.sample(range(n), k) # Indices of the centers from which the algorithm starts
                centers = X[first_centers_ind,:]
            else:
                # edouard's modification
                centers = initial_centers
            old_centers = np.ones([k,d])*np.inf
            Sigma = [np.identity(d)]*k
            old_Sigma = np.copy(Sigma)
            old_mi = np.zeros([k,d])
            old_weights = np.zeros(k)
            
            mi = np.zeros([k,d]) # means
            vi = np.zeros(k) # variances for the mahalanobis norms
            ci = np.zeros(k) # log(det(Sigma))
            
            Nstep = 1
            continue_Sigma = True
            
            while((continue_Sigma or (np.sum(old_centers!=centers)>0)) and (Nstep <= iter_max)):
                Nstep = Nstep + 1
                
                # Step 1: Update mi, vi and ci
                for i in range(k):
                    index = np.argsort([distance.mahalanobis(X[j,], centers[i,], np.linalg.inv(Sigma[i])) for j in range(X.shape[0])])
                    index = index[range(q)]
                    mi[i,] = np.mean(X[index,], axis = 0)
                    vect_aux = [distance.mahalanobis(X[index[j],], mi[i,], np.linalg.inv(Sigma[i])) for j in range(q)]
                    vi[i] = np.mean([val*val for val in vect_aux]) # The square of the Mahalanobis distance
                    sign, ci[i] = np.linalg.slogdet(Sigma[i]) # log(det(Sigma[i]))
                                     
                # Step 2: Update colors and min_distance
                min_distance = np.zeros(n) # Weighted distance between a point and its nearest center
                for j in range(n):
                    cost = np.inf
                    best_ind = 0
                    for i in range(k):
                        if(kept_centers[i]):
                            aux = distance.mahalanobis(X[j,],mi[i,],np.linalg.inv(Sigma[i]))
                            newcost = aux*aux + vi[i] + ci[i]
                            if (newcost < cost):
                                cost = newcost
                                best_ind = i
                    colors[j] = best_ind
                    min_distance[j] = cost
                    
                # Step 3: Trimming step - Put color -1 to the (n-sig) points with largest cost
                index = np.argsort(-min_distance)
                colors[index[range(n-sig)]] = -1
                ds = min_distance[index[range(n-sig,n)]]
                costt = np.mean(ds)
                
                # Step 4: Update Centers and mi and Sigma
                old_centers = np.copy(centers)
                old_mi = np.copy(mi)
                old_weights = vi+ci
                old_Sigma = np.copy(Sigma)
                for i in range(k):
                    pointcloud_size = np.sum(colors == i)
                    if(pointcloud_size>1):
                        centers[i,] = np.mean(X[colors==i,],axis = 0)  
                        index = np.argsort([distance.mahalanobis(X[j,], centers[i,], np.linalg.inv(Sigma[i])) for j in range(X.shape[0])])
                        index = index[range(q)]
                        mi[i,] = np.mean(X[index,], axis = 0)
                        aa = np.dot(np.array([mi[i,]-centers[i,]]).T,np.array([mi[i,]-centers[i,]]))
                        bb = (q-1)/q*np.cov(np.array([X[index[j],] for j in range(q)]).T)
                        cc = (pointcloud_size - 1)/(pointcloud_size)*np.cov(np.array(X[colors==i,]).T)
                        Sigma[i] = aa+bb+cc
                    elif(pointcloud_size==1):
                        centers[i,] = np.mean(X[colors==i,],axis = 0)  
                        index = np.argsort([distance.mahalanobis(X[j,], centers[i,], np.linalg.inv(Sigma[i])) for j in range(X.shape[0])])
                        index = index[range(q)]
                        mi[i,] = np.mean(X[index,], axis = 0)
                        aa = np.dot(np.array([mi[i,]-centers[i,]]).T,np.array([mi[i,]-centers[i,]]))
                        bb = (q-1)/q*np.cov(np.array([X[index[j],] for j in range(q)]).T)
                        Sigma[i] = aa + bb
                    else:
                        kept_centers[i] = False
                Stop_Sigma = True # True while old_Sigma = Sigma
                for i in range(k):
                    if(kept_centers[i]):
                        Stop_Sigma = (Stop_Sigma and (np.sum([old_Sigma[i]!=Sigma[i]])==0))
                continue_Sigma = not Stop_Sigma
                
            if(costt <= opt_cost):
                opt_cost = costt
                opt_centers = np.copy(old_centers)
                opt_means = np.copy(old_mi)
                opt_weigths = np.copy(old_weights)
                opt_Sigma = np.copy(old_Sigma)
                opt_colors = np.copy(colors)
                opt_kept_centers = np.copy(kept_centers)
                
        centers = opt_centers[opt_kept_centers,]
        Sigma = [opt_Sigma[i] for i in range(k) if opt_kept_centers[i]]#### ATTENTION !!!!
        means = opt_means[opt_kept_centers,]
        weights = opt_weigths[opt_kept_centers]
        colors = np.zeros(n)
        for i in range(n):
            colors[i] = np.sum(opt_kept_centers[range(int(opt_colors[i]+1))])-1
        cost = opt_cost
        
    return(centers, Sigma, means, weights, colors, cost)

# modifyed by Edouard on the 2019/10/29 to be able to initialise the centers
def kPLM(X,query_pts,q,k,sig,iter_max = 10,nstart = 1,initial_centers=None):
    '''
    Compute the values of the k-PDTM of the empirical measure of a point cloud X
    Require sklearn.neighbors.KDTree to search nearest neighbors
    
    Input:
    X: a nxd numpy array representing n points in R^d
    query_pts:  a sxd numpy array of query points
    q: parameter of the DTM in {1,2,...,n}
    k: number of centers
    sig: number of points considered as signal in the sample (other signal points are trimmed)
    initial_centers: k*d numpy array. algorithm starts with these centers instead of initialize them at random.
    
    Output: 
    kPDTM_result: a sx1 numpy array contaning the kPDTM of the 
    query points
    
    Example:
    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    Q = np.array([[0,0],[5,5]])
    kPLM_values = kPLM(X, Q, 3, 2,X.shape[0])
    '''
    n = X.shape[0]     
    if(q<=0 or q>n):
        raise AssertionError("Error: q should be in {1,2,...,n}")
    elif(k<=0 or k>n):
        raise AssertionError("Error: k should be in {1,2,...,n}")
    elif(X.shape[1]!=query_pts.shape[1]):
        raise AssertionError("Error: X and query_pts should contain points with the same number of coordinates.")
    else:
        centers, Sigma, means, weights, colors, cost = optima_for_kPLM(X,q,k,sig,iter_max = iter_max,nstart = nstart,
                                                                      initial_centers=initial_centers)
        kPLM_result = np.zeros(query_pts.shape[0])
        for i in range(query_pts.shape[0]):
            kPLM_result[i] = np.inf
            for j in range(means.shape[0]):
                aux0 = distance.mahalanobis(query_pts[i,],means[j,],np.linalg.inv(Sigma[j]))
                aux = aux0*aux0 + weights[j] # We don't take the squareroot, since aux could be negative
                if(aux<kPLM_result[i]):
                    kPLM_result[i] = aux 
                    
    return(kPLM_result, centers, Sigma, means, weights, colors, cost)