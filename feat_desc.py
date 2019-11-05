import numpy as np
import cv2
from scipy import signal
'''
  File name: feat_desc.py
  Author:
  Date created:
'''

'''
  File clarification:
    Extracting Feature Descriptor for each feature point. You should use the subsampled image around each point feature, 
    just extract axis-aligned 8x8 patches. Note that it’s extremely important to sample these patches from the larger 40x40 
    window to have a nice big blurred descriptor. 
    - Input img: H × W matrix representing the gray scale input image.
    - Input x: N × 1 vector representing the column coordinates of corners.
    - Input y: N × 1 vector representing the row coordinates of corners.
    - Outpuy descs: 64 × N matrix, with column i being the 64 dimensional descriptor (8 × 8 grid linearized) computed at location (xi , yi) in img.
'''

# def feat_desc(img, x, y):
#     sift = cv2.xfeatures2d.SIFT_create()
#     kp=[]
#     for (_x,_y) in zip(x,y):
#         kp.append(cv2.KeyPoint(_x,_y,40))
#     kp,descs = sift.compute(img,kp)
#     print()
#     return descs
def findDerivatives(I_gray):
    # using the Gaussian kernel taught in class
    G = 1/159.0*np.array([[2, 4, 5, 4, 2], [4, 9, 12, 9, 4], [5, 12, 15, 12, 5], [4, 9, 12, 9, 4], [2, 4, 5, 4, 2]])

    # convolution of dx,dy with the Gaussian is equivalent to taking dx,dy with the smoothened image
    dx,dy = np.gradient(G, axis = (1,0))
    Magx = signal.convolve2d(I_gray, dx, 'same')
    Magy = signal.convolve2d(I_gray, dy, 'same')
    Mag = np.sqrt(Magx*Magx + Magy*Magy)

    # gives us the direction of the gradient at a pixel
    Ori = np.arctan2(Magy, Magx)
    return (Mag, Magx, Magy, Ori)

def feat_desc(img, x, y):
    Mag, Magx, Magy, Ori = findDerivatives(img)
    img = Mag
    descs=np.zeros((64,len(x)))
    padImage = np.zeros((img.shape[0]+40,img.shape[1]+40))
    padImage[20:img.shape[0]+20,20:img.shape[1]+20] = img
    # #first 20 rows
    # for i in range(20):
    #     for j in range(20,img.shape[1]+20):
    #         padImage[i,j] = padImage[i+20,j]
    # #first 20 columns
    # for i in range(20,img.shape[0]+20):
    #     for j in range(20):
    #         padImage[i,j] = padImage[i,j+20]
    # # last 20 rows
    # for i in range(img.shape[0]+20,img.shape[0]+40):
    #     for j in range(20,img.shape[1]+20):
    #         padImage[i,j] = padImage[i-20,j]
    # #last 20 cols
    # for i in range(20,img.shape[0]+20):
    #     for j in range(img.shape[1]+20,img.shape[1]+40):
    #         padImage[i,j] = padImage[i,j-20]

    # print(padImage)
    k = 0
    for (_x,_y) in zip(x,y):
        _x = int(_x)
        _y = int(_y)
        patch = padImage[_y:_y+40,_x:_x+40]
        blurredPatch = cv2.GaussianBlur(patch,(5,5),1)
        desc = []
        for i in range(0,40,5):
            for j in range(0,40,5):
                smallPatch = patch[i:i+5,j:j+5]
                maxValue=np.max(smallPatch.flatten())
                desc.append(maxValue)
                # print(maxValue)
                # print(smallPatch.shape)
        desc = np.array(desc)
        desc = (desc - desc.mean())/desc.std()
        descs[:,k]=desc
        k=k+1
    return descs