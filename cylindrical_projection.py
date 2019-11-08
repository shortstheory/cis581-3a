import numpy as np
import cv2
from scipy import signal

def cyldinrical_projection(img, f):
    xc = img.shape[1]/2
    yc = img.shape[0]/2
    x,y = np.meshgrid(np.arange(img.shape[1]),np.arange(img.shape[0]))
    X = np.array([x.flatten()])
    Y = np.array([y.flatten()])
    
    X_cyl = np.rint(f*np.tan((X-xc)/f)+xc)
    Y_cyl = np.rint(np.divide((Y-yc),np.cos((X-xc)/f))+yc)
    pts_set = np.vstack((Y_cyl,X_cyl))
    print(pts_set)
    inds = np.ravel_multi_index(pts_set,(img.shape[0], img.shape[1]))
    flat_img = img.reshape((img.shape[0]*img.shape[1],3))
    res_img = flat_img[inds]
    res_img = res_img.reshape((img.shape[0],img.shape[1],3))
    return res_img





