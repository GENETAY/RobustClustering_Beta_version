from math import log,inf,floor,ceil

def flatten_list(lols): # lols = list of listS
    res = []
    for l in lols:
        res += l
    return res

def block_size_max(n_sample,n_outliers):
    '''
    Function which fits the maximum size of blocks before a the breakpoint
    ```prms```
    n_sample: nb of data
    n_outlier: nb of outliers
    '''
    outlier_proportion = n_outliers/n_sample
    
    if n_outliers == 0:
        bloc_size_max = inf
    else:
        bloc_size_max = floor(log(2.)/log(1/(1-outlier_proportion)))
    return bloc_size_max


def minimal_number_of_blocks(n_sample,n_outliers,b_size=None,alpha=0.05):
    '''
    Function which fits the minimum nb of blocks for a given size t before a the breakpoint
    ```prms```
    n_sample: nb of data
    n_outlier: nb of outliers
    b_size = bloc_size
    alpha : threshold confiance
    '''
    outlier_proportion = n_outliers/n_sample
    
    if n_outliers/n_sample >= 0.5:
        raise Exception('Either the number of outliers is too big or there is too much noise in the data')
    elif b_size is None:
        if n_outliers == 0 :
            bloc_nb_min = 1
        else:
            b_size_loc_ = block_size_max(n_sample,n_outliers)
            bloc_nb_min = ceil(log(1/alpha) / (2* ((1-outlier_proportion)**b_size_loc_ - 1/2)**2))
    else:
        if n_outliers == 0 :
            bloc_nb_min = 1
        else:
            bloc_nb_min = ceil(log(1/alpha) / (2* ((1-outlier_proportion)**b_size - 1/2)**2))
            
    return bloc_nb_min