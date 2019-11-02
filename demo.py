import cv2
import numpy as np
from corner_detector import *
from anms import *
filename = 'small-middle.jpg'
img=cv2.imread(filename)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
c = corner_detector(gray)
x,y,rmax=anms(c, 1000)