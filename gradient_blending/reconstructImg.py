'''
  File name: reconstructImg.py
  Author: Arnav Dhamija
  Date created: 9/29/2019
'''


from getCoefficientMatrix import *
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import *
import copy

def normalise(x, overallMin, overallMax):
    x = (x - overallMin)/(overallMax - overallMin)
    return x*255

def reconstructImg(indexes, red, green, blue, targetImg):
    coeff = getCoefficientMatrix(indexes)
    redX = spsolve(coeff, red)
    greenX = spsolve(coeff, green)
    blueX = spsolve(coeff, blue)

    overallMin = min(np.min(redX),np.min(greenX),np.min(blueX))
    overallMax = max(np.max(redX),np.max(greenX),np.max(blueX))

    print(overallMin)
    print(overallMax)

    redX = normalise(redX, overallMin, overallMax)
    greenX = normalise(greenX, overallMin, overallMax)
    blueX = normalise(blueX, overallMin, overallMax)

    # not really needed since we're already normalising all the values
    redX = np.clip(redX, 0, 255)
    greenX = np.clip(greenX, 0, 255)
    blueX = np.clip(blueX, 0, 255)

    resultImg = copy.deepcopy(targetImg)

    for i in range(0, indexes.shape[0]):
        for j in range(0, indexes.shape[1]):
            if (indexes[i][j] != 0):
                val = (indexes[i][j] - 1).astype('int')
                resultImg[i,j,0] = redX[val]
                resultImg[i,j,1] = greenX[val]
                resultImg[i,j,2] = blueX[val]
    return resultImg
