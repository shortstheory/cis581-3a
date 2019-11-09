from matplotlib import pyplot as plt
import cv2
import numpy as np

def interp2(v, xq, yq):

    if len(xq.shape) == 2 or len(yq.shape) == 2:
        dim_input = 2
        q_h = xq.shape[0]
        q_w = xq.shape[1]
        xq = xq.flatten()
        yq = yq.flatten()

    h = v.shape[0]
    w = v.shape[1]
    if xq.shape != yq.shape:
        raise 'query coordinates Xq Yq should have same shape'

    x_floor = np.floor(xq).astype(np.int32)
    y_floor = np.floor(yq).astype(np.int32)
    x_ceil = np.ceil(xq).astype(np.int32)
    y_ceil = np.ceil(yq).astype(np.int32)

    x_floor[x_floor < 0] = 0
    y_floor[y_floor < 0] = 0
    x_ceil[x_ceil < 0] = 0
    y_ceil[y_ceil < 0] = 0

    x_floor[x_floor >= w-1] = w-1
    y_floor[y_floor >= h-1] = h-1
    x_ceil[x_ceil >= w-1] = w-1
    y_ceil[y_ceil >= h-1] = h-1

    v1 = v[y_floor, x_floor]
    v2 = v[y_floor, x_ceil]
    v3 = v[y_ceil, x_floor]
    v4 = v[y_ceil, x_ceil]

    lh = yq - y_floor
    lw = xq - x_floor
    hh = 1 - lh
    hw = 1 - lw

    w1 = hh * hw
    w2 = hh * lw
    w3 = lh * hw
    w4 = lh * lw

    interp_val = v1 * w1 + w2 * v2 + w3 * v3 + w4 * v4

    if dim_input == 2:
        return interp_val.reshape(q_h, q_w)
    return interp_val


def getHomographedImg(img1,H,shape0, shape1):
#     corners = np.asarray([[0,0,1],[0,img1.shape[0],1],[img1.shape[1],0,1],[img1.shape[1],img1.shape[0],1]]).T
#     cornersT = np.matmul(H,corners)
#     cornersT = cornersT/cornersT[-1,:]
#     cornersT = cornersT.round()
#     xmax = int(np.max(cornersT[0,:]))
#     xmin = int(np.min(cornersT[0,:]))
#     ymax = int(np.max(cornersT[1,:]))
#     ymin = int(np.min(cornersT[1,:]))
#     sizex = int(xmax-xmin)
#     sizey = int(ymax-ymin)
#     print(xmin,xmax,ymin,ymax)
    x_r = np.array(range(shape0))
#     x_r = x_r+xmin
    y_r = np.array(range(shape1))
#     y_r = y_r+ymin

    x_co1,y_co1 = np.meshgrid(x_r,y_r)
    x_co = x_co1.flatten()
    y_co = y_co1.flatten()
    ones = np.ones(x_co.shape)
    Cord = np.vstack([x_co,y_co,ones])
    CordOg = np.matmul(np.linalg.inv(H),Cord)
    CordOg = CordOg/CordOg[2,:]
    x_og = CordOg[0,:].reshape(x_co1.shape)
    y_og = CordOg[1,:].reshape(y_co1.shape)
    im1BT = interp2(img1[:,:,0],x_og,y_og)
    im1GT = interp2(img1[:,:,1],x_og,y_og)
    im1RT = interp2(img1[:,:,2],x_og,y_og)

    outliersX1 = np.argwhere(x_og>img1.shape[1])
    outliersX2 = np.argwhere(x_og<0)
    outliersY1 = np.argwhere(y_og>img1.shape[0])
    outliersY2 = np.argwhere(y_og<0)
    outliers = np.vstack([outliersX1,outliersX2,outliersY1,outliersY2])
    im1BT[outliers[:,0],outliers[:,1]]=0
    im1GT[outliers[:,0],outliers[:,1]]=0
    im1RT[outliers[:,0],outliers[:,1]]=0
    # print('-------',outliers[:,1].max())
    #print('-----',xmin)
    #     cornersT2 = cornersT.copy()
    #     print('cornerssss',cornersT)
    #     cornersT2[0,:] = cornersT[0,:]-xmin
    #     cornersT2[1,:] = cornersT[1,:]-ymin
    #print(cornersT2)
    #print(xmin)
    img1T = np.zeros([shape1,shape0,3])
    img1T[:,:,0] = im1BT
    img1T[:,:,1] = im1GT
    img1T[:,:,2] = im1RT
    # print('image values', img1T[48-5,473-5,:])
    # print(x_og.shape)
    # print(outliersX1.shape)
    # print(outliersX1[:,1].max())
    plt.imshow(img1T.astype(int))
    plt.show()
    #print(xmax-xmin,xmin,ymax-ymin,ymin)
    return img1T
