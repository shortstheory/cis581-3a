import numpy as np
import cv2
from scipy import signal
import math
from matplotlib import pyplot as plt
import sys
from scipy.ndimage import gaussian_filter
sys.path.append('gradient_blending')
from seamlessCloningPoisson import *

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
    imgLMask = np.ones((imgL.shape))
    imgRMask = np.ones((imgR.shape))

    imgOutline = np.zeros((imgL.shape))
    imgOutline[0,0:imgOutline.shape[1]-1] = 1
    imgOutline[imgOutline.shape[0]-1,0:imgOutline.shape[1]-1] = 1
    imgOutline[0:imgOutline.shape[0]-1,0] = 1
    imgOutline[0:imgOutline.shape[0]-1,imgOutline.shape[1]-1] = 1

    canvasL = cv2.warpPerspective(imgL, T@HLM,(int(imgL.shape[1]*2),int(imgL.shape[0]*1.5)))
    imgLMask = cv2.warpPerspective(imgLMask, T@HLM,(int(imgL.shape[1]*2),int(imgL.shape[0]*1.5)))
    outlineL = cv2.warpPerspective(imgOutline, T@HLM,(int(imgL.shape[1]*2),int(imgL.shape[0]*1.5)))

    plt.imshow(outlineL)
    plt.show()
    outlineL = outlineL.astype('bool')

    # keep this here!
    imgMMask = np.zeros((imgLMask.shape))

    _x = abs(max(0,-xmin))
    _y = abs(max(0,-ymin))

    offsetX = int(_x)
    offsetY = int(_y)
    T = [[1, 0, offsetX], [0, 1, offsetY], [0, 0, 1]]

    canvasR = cv2.warpPerspective(imgR, T@np.linalg.inv(HMR),(int(imgL.shape[1]*2),int(imgL.shape[0]*1.5)))
    imgRMask = cv2.warpPerspective(imgRMask, T@np.linalg.inv(HMR),(int(imgL.shape[1]*2),int(imgL.shape[0]*1.5)))
    outlineR = cv2.warpPerspective(imgOutline, T@np.linalg.inv(HMR),(int(imgL.shape[1]*2),int(imgL.shape[0]*1.5)))

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
    imgLMMask = imgLMMask.astype('bool')
    imgRMMask = imgRMMask.astype('bool')
    alpha = 0.5
    canvasImgAlpha = canvasL+canvasM
    canvasImgAlpha[imgLMMask] = alpha*canvasL[imgLMMask]+(1-alpha)*canvasM[imgLMMask]
    canvasImgAlpha[imgRMMask] = alpha*canvasImgAlpha[imgRMMask]
    canvasR[imgRMMask] = (1-alpha)*canvasR[imgRMMask]
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
