'''
  File name: seamlessCloningPoisson.py
  Author: Arnav Dhamija
  Date created: 9/29/2019
'''

from matplotlib.image import *
import matplotlib.pyplot as plt
from getIndexes import *
from getCoefficientMatrix import *
from reconstructImg import *
from getSolutionVect import *
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import *
import numpy as np

def seamlessCloningPoisson(sourceImg, targetImg, mask, offsetX, offsetY):
    idx = getIndexes(mask, targetImg.shape[0], targetImg.shape[1], offsetX, offsetY)
    red = getSolutionVect(idx, sourceImg[:,:,0], targetImg[:,:,0], offsetX, offsetY)
    green = getSolutionVect(idx, sourceImg[:,:,1], targetImg[:,:,1], offsetX, offsetY)
    blue = getSolutionVect(idx, sourceImg[:,:,2], targetImg[:,:,2], offsetX, offsetY)
    resultImg = reconstructImg(idx, red, green, blue, targetImg)
    return resultImg
