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
    imgLMask = np.ones((imgL.shape))
    imgRMask = np.ones((imgR.shape))

    canvasL = cv2.warpPerspective(imgL, T@HLM,(int(imgL.shape[1]*2),int(imgL.shape[0]*1.5)))
    imgLMask = cv2.warpPerspective(imgLMask, T@HLM,(int(imgL.shape[1]*2),int(imgL.shape[0]*1.5)))

    # keep this here!
    imgMMask = np.zeros((imgLMask.shape))

    _x = abs(max(0,-xmin))
    _y = abs(max(0,-ymin))

    offsetX = int(_x)
    offsetY = int(_y)
    T = [[1, 0, offsetX], [0, 1, offsetY], [0, 0, 1]]

    canvasR = cv2.warpPerspective(imgR, T@np.linalg.inv(HMR),(int(imgL.shape[1]*2),int(imgL.shape[0]*1.5)))
    imgRMask = cv2.warpPerspective(imgRMask, T@np.linalg.inv(HMR),(int(imgL.shape[1]*2),int(imgL.shape[0]*1.5)))

    canvasL = canvasL#+canvasR
    canvasM = np.zeros((canvasL.shape))
    canvasImg = np.zeros((canvasL.shape))
    canvasM[_y:_y+imgM.shape[0],_x:_x+imgM.shape[1]]=imgM   

    imgMMask = np.zeros((imgLMask.shape))
    imgMMask[_y:_y+imgM.shape[0],_x:_x+imgM.shape[1]]=1

    imgLMMask = (imgLMask*imgMMask)#.astype('bool')
    imgRMMask = (imgRMask*(imgMMask+imgLMask))
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
    plt.imshow(canvasR)
    plt.show()
    # plt.imshow((imgLMask+imgMMask))
    # plt.show()
    imgLMMask = imgLMMask.astype('bool')
    imgRMMask = imgRMMask.astype('bool')
    alpha = 0.5
    canvasImg = canvasL+canvasM
    canvasImg[imgLMMask] = alpha*canvasL[imgLMMask]+(1-alpha)*canvasM[imgLMMask]
    canvasImg[imgRMMask] = alpha*canvasImg[imgRMMask]
    canvasR[imgRMMask] = (1-alpha)*canvasR[imgRMMask]
    canvasImg=canvasImg+canvasR
    return canvasImg



    #+canvasR
    # multimaskLMM = imgLMMask*alpha
    # multimaskRMM = imgRMMask*


    # canvasL[imgRMMask] = (1-alpha)*canvasR[imgRMMask]+alpha*canvasL[imgRMMask]
    # canvasR[imgRMMask] = 0

    # imgLMask[imgLMMask] = alpha
    # imgMMask[imgLMMask] = 1-alpha
    # imgMMask[imgRMMask] = alpha
    # imgRMask[imgRMMask] = 1-alpha
    # canvasImg = imgMMask*canvasM+imgLMask*canvasL#+imgRMask*canvasR
    # canvasL[imgLMMask] = alpha*canvasL[imgLMMask]
    # canvasM[imgLMMask] = (1-alpha)*canvasM[imgLMMask]
