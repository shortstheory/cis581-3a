import numpy as np
import cv2
from scipy import signal
import math

def stitch_image(img1, img2, H):
    left = math.floor(-1*H[0,2])
    top = math.floor(-1*H[1,2])
    H[0,2] = 0
    H[1,2] = 0

    canvas = cv2.warpPerspective(img1, H,(int(img1.shape[1]*2), int(img1.shape[0]*2)))
    upper_right = np.array([img1.shape[1],0,1])
    lower_right = np.array([img1.shape[1],img1.shape[0],1])
    new_ur = H@upper_right.T
    new_ur = new_ur/new_ur[2]
    new_lr = H@lower_right.T
    new_lr = new_lr/new_lr[2]
    y_scale = new_lr[1]-new_ur[1]
    print(new_ur)
    print(new_lr)
    scale_ratio = y_scale/img2.shape[0]
    print(scale_ratio)
    H_img2 = np.array([[scale_ratio,0,0],[0,scale_ratio,0],[0,0,1]])
    print(H_img2)
    scaled_img = cv2.warpPerspective(img2, H_img2,(int(img2.shape[1]*scale_ratio), int(img2.shape[0]*scale_ratio)))
    canvas[top+2:2+top+scaled_img.shape[0],left:left+scaled_img.shape[1]] = scaled_img
    # canvas[top:top+img2.shape[0],left:left+img2.shape[1]] = img2

    return canvas,scaled_img