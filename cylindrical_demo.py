from matplotlib import pyplot as plt
import cv2
import numpy as np
import utilities
from mymosaic import *
import sys
from get_cylindrical import *

# more sets of images available in the images/ folder!
img1=cv2.imread('images/shoemaker-left.jpg')
img2=cv2.imread('images/shoemaker-middle.jpg')
img3=cv2.imread('images/shoemaker-right.jpg')
R = 3442.0/3 #May be changed to any desired cylindrical circumference value (size of the unwrapped image)
f = 500.0 #Focal length (in the units of number of pixel length)
get_cylindrical(img1,img2,img3,f,R)
imgL = cv2.imread('cylin1.jpeg')
imgM = cv2.imread('cylin2.jpeg')
imgR = cv2.imread('cylin3.jpeg')

# find left -> middle homography and right->middle homography
HLM = utilities.get_homography(imgL,imgM,False,"L")
HRM = utilities.get_homography(imgR,imgM,False,"R")

canvas = mymosaic(imgL,imgM,imgR,HLM,HRM)
if cv2.imwrite("output.png",canvas):
    print("Output saved to output.png")

