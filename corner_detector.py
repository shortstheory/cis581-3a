'''
  File name: corner_detector.py
  Author:
  Date created:
'''

'''
  File clarification:
    Detects corner features in an image. You can probably find free “harris” corner detector on-line, 
    and you are allowed to use them.
    - Input img: H × W matrix representing the gray scale input image.
    - Output cimg: H × W matrix representing the corner metric matrix.
'''
import cv2
import numpy as np

def corner_detector(img):
    # Your Code Here
    # cimg = cv2.cornerHarris(img,2,3,0.04)
    # cimg[cimg<0.01*cimg.max()]=0
    # return cimg
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints = sift.detect(img,None)
    cimg = np.zeros((img.shape))
    print(len(keypoints))
    for kp in keypoints:
        # print(kp.pt)
        cimg[round(kp.pt[1])][round(kp.pt[0])] = max(cimg[round(kp.pt[1])][round(kp.pt[0])],kp.response)
    return cimg
