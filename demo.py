import cv2
import numpy as np
from corner_detector import *

filename = 'middle.jpg'
img=cv2.imread(filename)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
c = corner_detector(gray)
print(c[c>0].shape)