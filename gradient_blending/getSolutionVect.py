'''
  File name: getSolutionVect.py
  Author: Arnav Dhamija
  Date created: 9/29/2019
'''

import numpy as np
from scipy import signal

# although this isn't the most efficient approach, the slowest step is inverting the A matrix
def getSolutionVect(indexes, source, target, offsetX, offsetY):
    n = np.max(indexes).astype(int)
    lapSrc = np.zeros((n))
    SolVectorb = np.zeros((n))
    b = np.zeros((n))
    for i in range(offsetY, indexes.shape[0]):
        for j in range(offsetX, indexes.shape[1]):
            idx = indexes[i][j].astype('int')
            if idx != 0:
                val = idx - 1
                if i-1 > 0 and indexes[i-1][j] == 0:
                    b[val] += target[i-1][j]
                if i+1 < indexes.shape[0] and indexes[i+1][j] == 0:
                    b[val] += target[i+1][j]
                if j-1 > 0 and indexes[i][j-1] == 0:
                    b[val] += target[i][j-1]
                if j+1 < indexes.shape[1] and indexes[i][j+1] == 0:
                    b[val] += target[i][j+1]
    laplacian = signal.convolve2d(source, [[0,-1,0],[-1,4,-1],[0,-1,0]],'same')
    for i in range(0, source.shape[0]):
        for j in range(0, source.shape[1]):
            if indexes[i+offsetY][j+offsetX].astype(int)!=0:
                val = indexes[i+offsetY][j+offsetX].astype(int)-1
                lapSrc[val] = laplacian[i][j]
    SolVectorb = lapSrc + b
    return SolVectorb
