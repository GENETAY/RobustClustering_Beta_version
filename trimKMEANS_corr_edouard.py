from random import sample
import numpy as np

def trimmed_Kmeans(data,k,trim=0.1, runs=100, points= None,printcrit=False,maxit=None):
    '''
    data: np.array of dataset
    k   : nb of clusters
    trim: trimmed parameters
    runs: nb of iterations max
    points : initial datapoints. None by default
    '''
    if maxit is None:
        maxit = 2*len(data)
    
    countmode = runs+1
    data = np.array(data)
    n,p  = data.shape
    nin  = round((1-trim)*n)
    crit = np.Inf

    oldclass = np.zeros((n,))
    iclass   = np.zeros((n,))
    optclass = np.zeros((n,))
    disttom  = np.zeros((n,))
    
    for i in range(runs):
        #if i/countmode == round(i/countmode):
            #print("iteration",i)
        if points is None:
            means = data[sample(np.arange(n).tolist(),k),:]
        else:
            means = points.copy()
        wend = False
        itcounter = 0

        while not wend:
            itcounter += 1
            for j in range(n):
                dj = np.zeros((k,))
                for l in range(k):
                    #print(data[j,:],means[j,:])
                    dj_   = (data[j,:]-means[l,:])**2
                    dj[l] = dj_.sum()
                iclass[j] = dj.argmin()
                disttom[j]= dj.min()

            order_idx = np.argsort(disttom)[(nin+1):]
            iclass[order_idx] = -1 # t'es sur que c'est pas la classe d'outliers ici? -1, 0 ou K+1??
            
            if itcounter >= maxit or np.all(oldclass in iclass) :
                wend = True
            else:
                for l in range(k):
                    if sum(iclass==l)==0 : # j'ai l'impression que si ==0 alors toutes les donnees sont outliers
                        means[l,:] = data[iclass==0,:]
                    else:
                        if sum(iclass==l)>1 :
                            if means.shape[1] == 1:
                                means[l,:] = data[iclass==l,:].means()
                            else:
                                means[l,:] = data[iclass==l,:].means(axis=1)
                        else:
                            means[l,:] = data[iclass==l,:]
                oldclass = iclass # here i changed "<-" into '='
        
        newcrit = disttom[iclass>0].sum()
        if printcrit:
            print("Iteration ",i," criterion value ",newcrit/nin,"\n") # ah bon!? on calcul la distorsion moyenne sur les donnees non trimmmees...?!
        if newcrit <= crit :
            optclass = iclass.copy()
            crit = newcrit.copy()
            optmeans = means.copy()

#     optclass[optclass==0] = k+1 # ca suggere que les outliers sont les 0
    out = {'classification':optclass,'means':optmeans,'criterion':crit/nin,'disttom':disttom,
           'ropt':np.sort(disttom)[nin],'k':k,'trim':trim,"runs":runs}
    return(out)

'''
I changed the "=0" at line 49 into "=-1"
I changed the "<-" at line 65 into "="
I commented line 75 to keep iclass as classification array
'''