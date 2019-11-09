'''
  File name: getCoefficientMatrix.py
  Author: Arnav Dhamija
  Date created: 9/29/2019
'''

from scipy.sparse import lil_matrix, csr_matrix
import numpy as np

def getCoefficientMatrix(indexes):
    n = np.max(indexes).astype(int)
    print(n)
    laplacian = lil_matrix((n,n))
    # print(laplacian.shape)
    for i in range(0, indexes.shape[0]):
        for j in range(0, indexes.shape[1]):
            if indexes[i][j].astype(int)!=0:
                val = indexes[i][j].astype(int)-1
                laplacian[val,val] = 4
                if j > 0:
                    laplacian[val,indexes[i][j-1].astype(int)-1] = -1
                if i > 0:
                    laplacian[val,indexes[i-1][j].astype(int)-1] = -1
                if j < indexes.shape[1]-1:
                    laplacian[val,indexes[i][j+1].astype(int)-1] = -1
                if i < indexes.shape[0]-1:
                    laplacian[val,indexes[i+1][j].astype(int)-1] = -1
    coeffA = csr_matrix(laplacian)
    return coeffA
