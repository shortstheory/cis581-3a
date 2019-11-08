import numpy as np
import cv2
from scipy import signal
def warp_img_perspective(img, H):
    x,y = np.meshgrid(np.arange(img.shape[1]),np.arange(img.shape[0]))
    X = np.array([x.flatten()])
    Y = np.array([y.flatten()])
    
    pts_set = np.vstack((X,Y,np.ones(X.shape)))
    transform = np.linalg.inv(H)@pts_set
    print(pts_set.shape)
    pts = np.vstack((transform[0],transform[1]))
    print(pts)
    # inds = np.ravel_multi_index(pts.astype('int'),(img.shape[0], img.shape[1]))
    # flat_img = img.reshape((img.shape[0]*img.shape[1],3))
    # res_img = flat_img[inds]
    # res_img = res_img.reshape((img.shape[0],img.shape[1],3))
    # return res_img