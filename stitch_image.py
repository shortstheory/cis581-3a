import numpy as np
import cv2
from scipy import signal
import math
from matplotlib import pyplot as plt


def stitch_image(img1, img2, H):
    left = math.ceil(-1*H[0,2])
    top = max(math.ceil(-1*H[1,2]),0)
    canvas_mask = np.ones((img1.shape))

    H[0,2] = 150
    H[1,2] = 0
    # img1 = cv2.copyMakeBorder(img1, top, 0,left,0,cv2.BORDER_CONSTANT)
    canvas = cv2.warpPerspective(img1, H,(int(img1.shape[1]*2), int(img1.shape[0]*2)),cv2.WARP_INVERSE_MAP)
    mosaic = canvas
    # scaled_img = mosaic
    # canvas_mask = cv2.warpPerspective(canvas_mask, H,(int(canvas_mask.shape[1]*2), int(canvas_mask.shape[0]*2)))
    # # plt.imshow(canvas_mask)
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
    H_img2 = np.array([[scale_ratio,0,0],[0,scale_ratio,top],[0,0,1]])
    print(H_img2)
    print(canvas.shape)
    scaled_img = cv2.warpPerspective(img2, H_img2,(int(img2.shape[1]*scale_ratio), int(img2.shape[0]*scale_ratio)+top))
    
    # img2_mask = np.ones((img2.shape))
    # img2_mask = cv2.warpPerspective(img2_mask, H_img2,(int(img2_mask.shape[1]*scale_ratio), int(img2_mask.shape[0]*scale_ratio)+top))
    canvas[0:scaled_img.shape[0],left:left+scaled_img.shape[1]] = scaled_img
    # img2_canvas = np.zeros((canvas_mask.shape))
    # img2_canvas_mask = np.zeros((canvas_mask.shape))
    # img2_canvas[0:scaled_img.shape[0],left:left+scaled_img.shape[1]] = scaled_img
    # img2_canvas_mask[0:scaled_img.shape[0],left:left+scaled_img.shape[1]] = img2_mask
    # print('mask')
    # img2_canvas = img2_canvas.astype('int')
    # plt.show()
    # print('done')
    # overlap_region = (img2_canvas*canvas_mask.astype('bool'))
    # overlap_region = overlap_region.astype('bool')
    # mosaic = canvas.copy()
    # mosaic[0:scaled_img.shape[0],left-120:-120+left+scaled_img.shape[1]] = scaled_img
    # alpha = 0
    # # mosaic[overlap_region]=0
    # yam = np.zeros(mosaic.shape)
    # yam[overlap_region] = alpha*canvas[overlap_region].astype('float')+(1-alpha)*img2_canvas[overlap_region].astype('float')
    # yam = yam.astype('int')
    # plt.imshow(yam)
    # # mosaic[overlap_region] = yam[overlap_region]
    # plt.show()
    
    # plt.imshow(np.logical_and(img2_canvas.astype('bool'),canvas_mask.astype('bool')).astype('int'))
    # canvas[0:0+img2.shape[0],left:left+img2.shape[1]] = img2

    return mosaic,scaled_img