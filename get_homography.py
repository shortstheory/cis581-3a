from feat_desc import *
import cv2
import numpy as np
from corner_detector import *
from anms import *
from feat_match import *
from ransac_est_homography import *
from scipy import ndimage
from cylindrical_projection import *
from stitch_image import *
from warp_img_perspective import *

# Finds a homography to transform img1 -> img2
def get_homography(img1, img2):
    max_anms=4000

    gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    c = corner_detector(gray)
    X1,Y1,rmax=anms(c, max_anms)
    d1 = feat_desc(gray,X1,Y1)
    kp1=[]
    for (_x,_y) in zip(X1,Y1):
        kp1.append(cv2.KeyPoint(_x,_y,40))
    gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    c = corner_detector(gray)
    X2,Y2,rmax=anms(c, max_anms)
    d2 = feat_desc(gray,X2,Y2)
    kp2=[]
    for (_x,_y) in zip(X2,Y2):
        kp2.append(cv2.KeyPoint(_x,_y,40))
    m,dMatch=feat_match(d1, d2)
    x1=[]
    y1=[]
    x2=[]
    y2=[]
    for k,idx in enumerate(m):
        if (idx != -1):
            x1.append(X1[k])
            y1.append(Y1[k])
            x2.append(X2[idx])
            y2.append(Y2[idx])
    x1=np.array(x1)
    x2=np.array(x2)
    y1=np.array(y1)
    y2=np.array(y2)
    print(x1.shape)
    H, inlier_ind=ransac_est_homography(x1,y1,x2,y2,2)
    return H
