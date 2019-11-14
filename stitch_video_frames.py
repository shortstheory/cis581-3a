#!/usr/bin/env python3
from matplotlib import pyplot as plt
import cv2
import numpy as np
import utilities
from mymosaic import *
import sys

frames = 300

for i in range(1,frames+1):
    x = format(i,'04')
    imgL = cv2.imread('left/'+x+'.jpg')
    imgM = cv2.imread('middle/'+x+'.jpg')
    imgR = cv2.imread('right/'+x+'.jpg')

    # find left -> middle homography and right->middle homography
    HLM = utilities.get_homography(imgL,imgM,False,"L")
    print(HLM)
    HRM = utilities.get_homography(imgR,imgM,False,"R")
    print(HRM)
    canvas = mymosaic(imgL,imgM,imgR,HLM,HRM,False)
    if cv2.imwrite("panovid/output" + str(i) + ".png",canvas):
        print("Output saved to output" + str(i) + ".png")

