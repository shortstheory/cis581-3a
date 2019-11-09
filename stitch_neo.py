import numpy as np
import cv2
from scipy import signal
import math
from matplotlib import pyplot as plt
import sys
from scipy.ndimage import gaussian_filter
sys.path.append('gradient_blending')
from seamlessCloningPoisson import *
from getHomographedImg import *

def stitch_neo(imgL, imgM, imgR, HLM, HMR):
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
    print(T)
    print(HLM)
    print(T@HLM)

    canvasL = getHomographedImg(imgL, T@HLM,int(imgL.shape[1]*2),int(imgL.shape[0]*1.5))
    imgLMask = getHomographedImg(imgLMask, T@HLM,int(imgL.shape[1]*2),int(imgL.shape[0]*1.5))
    outlineL = getHomographedImg(imgOutline, T@HLM,int(imgL.shape[1]*2),int(imgL.shape[0]*1.5))


    plt.imshow(outlineL)
    plt.show()
    outlineL = outlineL.astype('bool')

    # keep this here!
    imgMMask = np.zeros((imgLMask.shape))

    _x = int(abs(max(0,-xmin)))
    _y = int(abs(max(0,-ymin)))

    offsetX = int(_x)
    offsetY = int(_y)
    T = [[1, 0, offsetX], [0, 1, offsetY], [0, 0, 1]]

    canvasR = getHomographedImg(imgR, T@np.linalg.inv(HMR),int(imgL.shape[1]*2),int(imgL.shape[0]*1.5))
    imgRMask = getHomographedImg(imgRMask, T@np.linalg.inv(HMR),int(imgL.shape[1]*2),int(imgL.shape[0]*1.5))
    outlineR = getHomographedImg(imgOutline, T@np.linalg.inv(HMR),int(imgL.shape[1]*2),int(imgL.shape[0]*1.5))

    plt.imshow(outlineR)
    plt.show()
    outlineR = outlineR.astype('bool')

    canvasL = canvasL#+canvasR
    canvasM = np.zeros((canvasL.shape))
    canvasImgAlpha = np.zeros((canvasL.shape))
    canvasM[_y:_y+imgM.shape[0],_x:_x+imgM.shape[1]]=imgM   

    imgMMask = np.zeros((imgLMask.shape))
    imgMMask[_y:_y+imgM.shape[0],_x:_x+imgM.shape[1]]=1

    # _sigma = 0.5
    # imgLMask = gaussian_filter(imgLMask, sigma=_sigma)
    # imgMMask = gaussian_filter(imgMMask, sigma=_sigma)
    # imgRMask = gaussian_filter(imgRMask, sigma=_sigma)

    kernel = np.ones((3,3))
    # imgLMask = cv2.dilate(imgLMask, kernel, iterations=10)
    # imgMMask = cv2.dilate(imgMMask, kernel, iterations=10)
    # imgRMask = cv2.dilate(imgRMask, kernel, iterations=10)


    imgLMMask = (imgLMask*imgMMask)
    imgRMMask = (imgRMask*(imgMMask+imgLMask))
    imgRMMask[np.nonzero(imgRMMask)] = 1

    # imgLMMask = gaussian_filter(imgLMMask, sigma=_sigma)
    # imgRMMask = gaussian_filter(imgRMMask, sigma=_sigma)
    # plt.imshow(imgLMask)
    # plt.show()
    # plt.imshow(imgMMask)
    # plt.show()
    # plt.imshow(imgRMask)
    # plt.show()

    # plt.imshow(imgLMMask)
    # plt.show()
    plt.imshow(imgRMask)
    plt.show()
    # plt.imshow(gaussian_filter(imgRMask,sigma=1))
    # plt.show()
    plt.imshow(canvasR)
    plt.show()
    # plt.imshow((imgLMask+imgMMask))
    # plt.show()

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
    plt.imshow(LMMmaskMultiplier)
    plt.show()
    plt.imshow(imgLMMask)
    plt.show()
    plt.imshow(imgLMMask*LMMmaskMultiplier)
    plt.show()
    plt.imshow(imgLMMask*(1-LMMmaskMultiplier))
    plt.show()

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
    plt.imshow(RMMmaskMultiplier)
    plt.show()
    plt.imshow(imgRMMask)
    plt.show()
    plt.imshow(imgRMMask*RMMmaskMultiplier)
    plt.show()
    plt.imshow(imgRMMask*(1-RMMmaskMultiplier))
    plt.show()

    imgLMMask = imgLMMask.astype('bool')
    imgRMMask = imgRMMask.astype('bool')

    alpha = 0.5
    canvasImgAlpha = canvasL+canvasM

    canvasImgAlpha[imgLMMask] = (imgLMMask*LMMmaskMultiplier*canvasM+imgLMMask*(1-LMMmaskMultiplier)*canvasL)[imgLMMask]

    # canvasImgAlpha[imgLMMask] = alpha*canvasL[imgLMMask]+(1-alpha)*canvasM[imgLMMask]
    # canvasImgAlpha[imgRMMask] = alpha*canvasImgAlpha[imgRMMask]
    # canvasR[imgRMMask] = (1-alpha)*canvasR[imgRMMask]

    canvasR[imgRMMask] = (imgRMMask*(RMMmaskMultiplier)*canvasR)[imgRMMask]

    canvasImgAlpha[imgRMMask] = (imgRMMask*(1-RMMmaskMultiplier)*canvasImgAlpha)[imgRMMask]
    canvasImgAlpha=canvasImgAlpha+canvasR


    canvasImgAlpha[outlineL] = cv2.dilate(canvasImgAlpha.copy(),kernel,iterations=1)[outlineL.astype('bool')]
    canvasImgAlpha[outlineR] = cv2.dilate(canvasImgAlpha.copy(),kernel,iterations=1)[outlineR.astype('bool')]
    # canvasImgAlpha = cv2.dilate(canvasImgAlpha, kernel, iterations=1)
    # canvasImgAlpha = gaussian_filter(canvasImgAlpha,sigma=01)
    canvasImgPoisson = canvasL+canvasM
    # overlap_img = np.zeros((canvasL.shape))
    # overlap_img[imgLMMask] = canvasL[imgLMMask]
    # canvasImgPoisson = seamlessCloningPoisson(overlap_img, canvasImgPoisson, imgLMMask[:,:,0], 0, 0)
    # canvasImgPoisson = canvasImgPoisson+canvasR
    # overlap_img = np.zeros((canvasL.shape))
    # overlap_img[imgRMMask] = canvasR[imgRMMask]
    # canvasImgPoisson = seamlessCloningPoisson(overlap_img, canvasImgPoisson, imgRMMask[:,:,0], 0, 0)

    return canvasImgAlpha,canvasImgPoisson



    #+canvasR
    # multimaskLMM = imgLMMask*alpha
    # multimaskRMM = imgRMMask*


    # canvasL[imgRMMask] = (1-alpha)*canvasR[imgRMMask]+alpha*canvasL[imgRMMask]
    # canvasR[imgRMMask] = 0

    # imgLMask[imgLMMask] = alpha
    # imgMMask[imgLMMask] = 1-alpha
    # imgMMask[imgRMMask] = alpha
    # imgRMask[imgRMMask] = 1-alpha
    # canvasImgAlpha = imgMMask*canvasM+imgLMask*canvasL#+imgRMask*canvasR
    # canvasL[imgLMMask] = alpha*canvasL[imgLMMask]
    # canvasM[imgLMMask] = (1-alpha)*canvasM[imgLMMask]
