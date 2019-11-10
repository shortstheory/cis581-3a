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

def mymosaic(imgL, imgM, imgR, HLM, HMR):
    widthMultiplier = 3
    heightMultiplier = 3
    cornersL = np.asarray([[0,0,1],[0,imgL.shape[0],1],[imgL.shape[1],0,1],[imgL.shape[1],imgL.shape[0],1]]).T

    cornersT = np.matmul(HLM,cornersL)
    cornersT = cornersT/cornersT[-1,:]
    cornersT = cornersT
    xmax = np.max(cornersT[0,:])
    xmin = np.min(cornersT[0,:])
    ymax = np.max(cornersT[1,:])
    ymin = np.min(cornersT[1,:])
    T = [[1, 0, max(0,-xmin)], [0, 1, max(0,-ymin)], [0, 0, 1]]
    imgLMask = np.ones((imgL.shape))
    imgRMask = np.ones((imgR.shape))

    imgOutline = np.zeros((imgL.shape))
    imgOutline[0,0:imgOutline.shape[1]-1] = 1
    imgOutline[imgOutline.shape[0]-1,0:imgOutline.shape[1]-1] = 1
    imgOutline[0:imgOutline.shape[0]-1,0] = 1
    imgOutline[0:imgOutline.shape[0]-1,imgOutline.shape[1]-1] = 1

    canvasL = warp_image(imgL, T@HLM,int(imgL.shape[1]*widthMultiplier),int(imgL.shape[0]*heightMultiplier))
    imgLMask = warp_image(imgLMask, T@HLM,int(imgL.shape[1]*widthMultiplier),int(imgL.shape[0]*heightMultiplier))
    outlineL = warp_image(imgOutline, T@HLM,int(imgL.shape[1]*widthMultiplier),int(imgL.shape[0]*heightMultiplier))

    outlineL = outlineL.astype('bool')

    imgMMask = np.zeros((imgLMask.shape))

    _x = int(abs(max(0,-xmin)))
    _y = int(abs(max(0,-ymin)))

    offsetX = int(_x)
    offsetY = int(_y)
    T = [[1, 0, offsetX], [0, 1, offsetY], [0, 0, 1]]

    canvasR = warp_image(imgR, T@np.linalg.inv(HMR),int(imgL.shape[1]*widthMultiplier),int(imgL.shape[0]*heightMultiplier))
    imgRMask = warp_image(imgRMask, T@np.linalg.inv(HMR),int(imgL.shape[1]*widthMultiplier),int(imgL.shape[0]*heightMultiplier))
    outlineR = warp_image(imgOutline, T@np.linalg.inv(HMR),int(imgL.shape[1]*widthMultiplier),int(imgL.shape[0]*heightMultiplier))
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
    print(LMMX.shape)
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
    print(RMMX.shape)
    RMMmaskMultiplier = np.zeros((imgRMMask.shape))
    RMMmaskMultiplier[RMMaskStartY:RMMaskEndY,RMMaskStartX:RMMaskEndX,0] = RMMX
    RMMmaskMultiplier[RMMaskStartY:RMMaskEndY,RMMaskStartX:RMMaskEndX,1] = RMMX
    RMMmaskMultiplier[RMMaskStartY:RMMaskEndY,RMMaskStartX:RMMaskEndX,2] = RMMX

    imgLMMask = imgLMMask.astype('bool')
    imgRMMask = imgRMMask.astype('bool')

    alpha = 0.5
    canvasImgAlpha = canvasL+canvasM

    canvasImgAlpha[imgLMMask] = (imgLMMask*LMMmaskMultiplier*canvasM+imgLMMask*(1-LMMmaskMultiplier)*canvasL)[imgLMMask]
    canvasR[imgRMMask] = (imgRMMask*(RMMmaskMultiplier)*canvasR)[imgRMMask]

    canvasImgAlpha[imgRMMask] = (imgRMMask*(1-RMMmaskMultiplier)*canvasImgAlpha)[imgRMMask]
    canvasImgAlpha=canvasImgAlpha+canvasR

    canvasImgAlpha[outlineL] = cv2.dilate(canvasImgAlpha.copy(),kernel,iterations=1)[outlineL.astype('bool')]
    canvasImgAlpha[outlineR] = cv2.dilate(canvasImgAlpha.copy(),kernel,iterations=1)[outlineR.astype('bool')]
    return canvasImgAlpha

