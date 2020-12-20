# GenetayEdouard

i am still working on that package.
There may be error or bug in code. It will be soon fully functionnal. Please, send me a message or an email for any questions or remark: genetay.edouard@gmail.com

# Content
It is a package of robust algorithms.
One can find:
- ROBIN (initialisation procedure, see https://rdrr.io/github/brodsa/wrsk/man/ROBIN.html)
- kmedianspp (the same as kmeans++ where remote data are less likely to be picked up than with kmeans++)
- TClust (gaussian mixture model with trimming, EM algo, see https://cran.r-project.org/web/packages/tclust/index.html)
- K-PDTM (robust EM algo based on trimming that computes distances between measures instead of between points, see https://arxiv.org/abs/1801.10346)
- trimKmeans (gaussian mixture model with trimming, EM algo, https://rdrr.io/cran/lowmemtkmeans/man/tkmeans.html)
- SMM (student mixture model, EM algo, I edited his code: https://github.com/luiscarlosgph/t-Student-Mixture-Models/blob/master/src/smm/smm.py)
- K-bMOM (a mix between kmeans and the estimator MOM (median-of-means))
- some HTML or notebooks

# Remark
We edited existing codes or created our own programs to be able to give initial centers in the procedure. For example, trimmed-kmeans (https://rdrr.io/cran/lowmemtkmeans/man/tkmeans.html) or SMM don't have such arguments. It did matter to be able to compare performances of algorithms. We recoded TClust because we did not find any python package to do it when we started our work. Thing may have change by now.

# how to use it
To use it. Download all file and put them in a folder named "my_folder", then, use in a notebook use either "from my_folder.KbMOM_v4_before_5 import KbMOM" or "from KbMOM_v4_before_5 import KbMOM" depending on the working directory of your notebook. It should work! otherwise email me! =)

# Further info
These files were the programs used to do the experiments of the article about K-bMOM, submitted by Adrien SAUMARD, Camille SAUMARD and Edoaurd GENETAY in 2020 in a journal.
- Preprint here: https://arxiv.org/abs/2002.03899
- Submitted to JASA on august 2020, rejected after review in november 2020
- Submitted to CSDA on december 2020, still in process

These files were also the programs used to prepare my talk to CMStat2020.
