'''
  File name: getIndexes.py
  Author: Arnav Dhamija
  Date created: 9/29/2019
'''

import numpy as np
def getIndexes(mask, targetH, targetW, offsetX, offsetY):
    ctr = 1
    indexes = np.zeros((targetH, targetW))
    for i in range(0, mask.shape[0]):
        for j in range(0, mask.shape[1]):   
             if mask[i][j] != 0:
                indexes[i+offsetY][j+offsetX] = ctr
                ctr = ctr+1
    indexes.astype(int)
    return indexes