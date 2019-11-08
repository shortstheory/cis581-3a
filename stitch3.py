import numpy as np
import cv2
from scipy import signal
import math
from matplotlib import pyplot as plt

def stitch3(imgL, imgM, imgR, HLM, HMR):
    cornersL = np.asarray([[0,0,1],[0,imgL.shape[0],1],[imgL.shape[1],0,1],[imgL.shape[1],imgL.shape[0],1]]).T

    cornersT = np.matmul(HLM,cornersL)
    cornersT = cornersT/cornersT[-1,:]
    cornersT = cornersT.round()
    xmax = int(np.max(cornersT[0,:]))
    xmin = int(np.min(cornersT[0,:]))
    ymax = int(np.max(cornersT[1,:]))
    ymin = int(np.min(cornersT[1,:]))
    T = [[1, 0, max(0,-xmin)], [0, 1, max(0,-ymin)], [0, 0, 1]]
    canvas = cv2.warpPerspective(imgL, T@HLM,(int(imgL.shape[1]*2),int(imgL.shape[0]*1.5)))

    _x = abs(max(0,-xmin))
    _y = abs(max(0,-ymin))
    plt.imshow(canvas)
    plt.show()
    offsetX = int(_x)
    offsetY = int(_y)
    T = [[1, 0, offsetX], [0, 1, offsetY], [0, 0, 1]]

    canvasR = cv2.warpPerspective(imgR, T@np.linalg.inv(HMR),(int(imgL.shape[1]*2),int(imgL.shape[0]*1.5)))
    canvas = canvas+canvasR
    canvas[_y:_y+imgR.shape[0],_x:_x+imgR.shape[1]]=imgM
    plt.imshow(canvas)
    plt.show()
    return canvas




