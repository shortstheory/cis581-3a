from feat_desc import *
import cv2
import numpy as np
from corner_detector import *
from anms import *
from feat_match import *
from ransac_est_homography import *
from cylindrical_projection import *
from stitch_image import *
from warp_img_perspective import *

# Finds a homography to transform img1 -> img2
def get_homography(img1, img2, createPlots=True, imgNum=0):
    imgName = "img"+str(imgNum)
    max_anms = 2000

    gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    c = corner_detector(gray)
    
    if createPlots:
        imgCopy = img1.copy()
        cCopy = c.copy()
        cCopy = cv2.dilate(cCopy, None)
        imgCopy[c>0] = [0,0,255]
        plt.imshow(cv2.cvtColor(imgCopy, cv2.COLOR_BGR2RGB))
        fig = plt.gcf()
        fig.savefig(imgName+'corner.png',dpi=200)
        plt.show()

    X1,Y1,rmax=anms(c, max_anms)
    d1 = feat_desc(gray,X1,Y1)
    kp1=[]
    for (_x,_y) in zip(X1,Y1):
        kp1.append(cv2.KeyPoint(_x,_y,40))

    gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    c = corner_detector(gray)
    X2,Y2,rmax = anms(c, max_anms)
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
    if createPlots:
        mask = np.array(inlier_ind, dtype=bool)
        mfilter = []
        for idx,i in enumerate(mask):
            if i == True:
                mfilter.append(dMatch[idx])
        img_matches = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], 3), dtype=np.uint8)
        print(len(mfilter))
        f=cv2.drawMatches(img1, kp1, img2, kp2, mfilter, img_matches, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
        fig = plt.gcf()
        fig.savefig(imgName+'inliners.png',dpi=200)
        plt.show()
        f=cv2.drawMatches(img1, kp1, img2, kp2, dMatch, img_matches, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
        fig = plt.gcf()
        fig.savefig(imgName+'outliers.png',dpi=200)
        plt.show()

        plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
        fig = plt.gcf()
        ax1 = fig.add_subplot(111)
        ax1.scatter(X1,Y1,c='r',s=1,label='ANMS')
        ax1.scatter(x1[mask],y1[mask],c='b',s=1,label='RANSAC')
        fig.savefig(imgName+'ransac.png',dpi=200)
        plt.show()
    return H
