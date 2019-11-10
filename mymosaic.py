import numpy as np
import cv2
from scipy import signal
import math
from matplotlib import pyplot as plt
import sys
from scipy.ndimage import gaussian_filter
sys.path.append('gradient_blending')
from seamlessCloningPoisson import *
from warp_image import *

def mymosaic(imgL, imgM, imgR, HLM, HRM):
    widthMultiplier = 3
    heightMultiplier = 3
    cornersL = np.asarray([[0,0,1],[0,imgL.shape[0],1],[imgL.shape[1],0,1],[imgL.shape[1],imgL.shape[0],1]]).T

    cornersLT = np.matmul(HLM,cornersL)
    cornersLT = cornersLT/cornersLT[-1,:]
    xmax = np.max(cornersLT[0,:])
    xmin = np.min(cornersLT[0,:])
    ymax = np.max(cornersLT[1,:])
    ymin = np.min(cornersLT[1,:])
    _x = int(abs(max(0,-xmin)))
    _y = int(abs(max(0,-ymin)))

    T = [[1, 0, _x], [0, 1, _y], [0, 0, 1]]

    # offsetX = int(_x)
    # offsetY = int(_y)
    # T = [[1, 0, offsetX], [0, 1, offsetY], [0, 0, 1]]

    cornersR = np.asarray([[0,0,1],[0,imgR.shape[0],1],[imgR.shape[1],0,1],[imgR.shape[1],imgR.shape[0],1]]).T
    cornersRT = np.matmul(T@HRM,cornersR)
    cornersRT = cornersRT/cornersRT[-1,:]

    canvasMaxWidth = int(np.max(cornersRT[0:]))
    canvasMaxHeight = int(max(np.max(cornersRT[1:]),ymax))
    print(canvasMaxWidth)
    print(canvasMaxHeight)

    imgLMask = np.ones((imgL.shape))
    imgRMask = np.ones((imgR.shape))

    imgOutline = np.zeros((imgL.shape))
    imgOutline[0,0:imgOutline.shape[1]-1] = 1
    imgOutline[imgOutline.shape[0]-1,0:imgOutline.shape[1]-1] = 1
    imgOutline[0:imgOutline.shape[0]-1,0] = 1
    imgOutline[0:imgOutline.shape[0]-1,imgOutline.shape[1]-1] = 1

    canvasL = warp_image(imgL, T@HLM,canvasMaxWidth,canvasMaxHeight)
    imgLMask = warp_image(imgLMask, T@HLM,canvasMaxWidth,canvasMaxHeight)
    outlineL = warp_image(imgOutline, T@HLM,canvasMaxWidth,canvasMaxHeight)

    # plt.imshow(cv2.cvtColor(canvasL,cv2.COLOR_BGR2RGB))
    # plt.show()
    plt.imshow(imgLMask)
    plt.show()
    plt.imshow(outlineL)
    plt.show()

    outlineL = outlineL.astype('bool')

    imgMMask = np.zeros((imgLMask.shape))

    canvasR = warp_image(imgR, T@HRM,canvasMaxWidth,canvasMaxHeight)
    imgRMask = warp_image(imgRMask, T@HRM,canvasMaxWidth,canvasMaxHeight)
    outlineR = warp_image(imgOutline, T@HRM,canvasMaxWidth,canvasMaxHeight)
    outlineR = outlineR.astype('bool')

    canvasM = np.zeros((canvasL.shape))
    canvasImgAlpha = np.zeros((canvasL.shape))
    canvasM[_y:_y+imgM.shape[0],_x:_x+imgM.shape[1]]=imgM   

    imgMMask = np.zeros((imgLMask.shape))
    imgMMask[_y:_y+imgM.shape[0],_x:_x+imgM.shape[1]]=1

    kernel = np.ones((3,3))

    imgLMMask = (imgLMask*imgMMask)
    imgRMMask = (imgRMask*(imgMMask+imgLMask))
    imgRMMask[np.nonzero(imgRMMask)] = 1

    # bounding box for LMMask
    LMMaskStartX = np.min(np.argwhere(imgLMMask).T[1])
    LMMaskEndX = np.max(np.argwhere(imgLMMask).T[1])+1
    LMMaskStartY = np.min(np.argwhere(imgLMMask).T[0])
    LMMaskEndY = np.max(np.argwhere(imgLMMask).T[0])+1

    LMMAlphas = np.linspace(0,1,LMMaskEndX-LMMaskStartX)
    LMMX, LMMY = np.meshgrid(LMMAlphas, np.arange(0,LMMaskEndY-LMMaskStartY))
    LMMmaskMultiplier = np.zeros((imgLMMask.shape))
    LMMmaskMultiplier[LMMaskStartY:LMMaskEndY,LMMaskStartX:LMMaskEndX,0] = LMMX
    LMMmaskMultiplier[LMMaskStartY:LMMaskEndY,LMMaskStartX:LMMaskEndX,1] = LMMX
    LMMmaskMultiplier[LMMaskStartY:LMMaskEndY,LMMaskStartX:LMMaskEndX,2] = LMMX

    # bounding box for RMM mask
    RMMaskStartX = np.min(np.argwhere(imgRMMask).T[1])
    RMMaskEndX = np.max(np.argwhere(imgRMMask).T[1])+1
    RMMaskStartY = np.min(np.argwhere(imgRMMask).T[0])
    RMMaskEndY = np.max(np.argwhere(imgRMMask).T[0])+1

    RMMAlphas = np.linspace(0,1,RMMaskEndX-RMMaskStartX)
    RMMX, RMMY = np.meshgrid(RMMAlphas, np.arange(0,RMMaskEndY-RMMaskStartY))
    RMMmaskMultiplier = np.zeros((imgRMMask.shape))
    RMMmaskMultiplier[RMMaskStartY:RMMaskEndY,RMMaskStartX:RMMaskEndX,0] = RMMX
    RMMmaskMultiplier[RMMaskStartY:RMMaskEndY,RMMaskStartX:RMMaskEndX,1] = RMMX
    RMMmaskMultiplier[RMMaskStartY:RMMaskEndY,RMMaskStartX:RMMaskEndX,2] = RMMX

    imgLMMask = imgLMMask.astype('bool')
    imgRMMask = imgRMMask.astype('bool')

    alpha = 0.5
    canvasImgAlpha = canvasL+canvasM

    plt.imshow((imgLMMask*LMMmaskMultiplier*canvasM).astype('int'))
    plt.show()
    plt.imshow((imgLMMask*(1-LMMmaskMultiplier)*canvasL).astype('int'))
    plt.show()
    plt.imshow((imgLMMask*LMMmaskMultiplier*canvasM).astype('int')+(imgLMMask*(1-LMMmaskMultiplier)*canvasL).astype('int'))
    plt.show()

    canvasImgAlpha[imgLMMask] = (imgLMMask*LMMmaskMultiplier*canvasM+imgLMMask*(1-LMMmaskMultiplier)*canvasL)[imgLMMask]
    canvasR[imgRMMask] = (imgRMMask*(RMMmaskMultiplier)*canvasR)[imgRMMask]

    canvasImgAlpha[imgRMMask] = (imgRMMask*(1-RMMmaskMultiplier)*canvasImgAlpha)[imgRMMask]
    canvasImgAlpha=canvasImgAlpha+canvasR
    return canvasImgAlpha

