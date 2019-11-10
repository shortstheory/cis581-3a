from matplotlib import pyplot as plt
import cv2
import numpy as np
from get_homography import *
from mymosaic import *
import sys

# more sets of images available in the images/ folder!
imgL = cv2.imread('images/shoemaker-left.jpg')
imgM = cv2.imread('images/shoemaker-middle.jpg')
imgR = cv2.imread('images/shoemaker-right.jpg')

# find left -> middle homography and middle->right homography
HLM = get_homography(imgL,imgM,False,"L")
HMR = get_homography(imgM,imgR,False,"R")

# right->middle is the inverse of what we found above
HRM = np.linalg.inv(HMR)

canvas = mymosaic(imgL,imgM,imgR,HLM,HRM)
if cv2.imwrite("output.png",canvas):
    print("Output saved to output.png")

