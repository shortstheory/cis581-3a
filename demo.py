import cv2
import numpy as np
import corner_detector

filename = 'middle.jpg'
img=cv2.imread(filename)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
