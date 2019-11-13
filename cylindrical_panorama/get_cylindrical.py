from utilities import interp2
from skimage.io import imsave
import numpy as np
from matplotlib import pyplot as plt

def get_cylindrical(img1,img2,img3,f):
    imgs = [img1,img2,img3]
    number = 1
    for img1 in imgs:
        xc = img1.shape[1]/2
        yc = img1.shape[0]/2
        edgecenters = np.asarray([[0,img1.shape[0]/2],[img1.shape[1]/2,0],[img1.shape[1],img1.shape[0]/2],[img1.shape[1]/2,img1.shape[0]],[0,0],[0,img1.shape[0]],[img1.shape[1],0],[img1.shape[1],img1.shape[0]]])
        x_ecs = edgecenters[:,0]
        y_ecs = edgecenters[:,1]
        x_ecsT = f*np.arctan((x_ecs-xc)/f) + xc			#Finding the transformed coordinates of the extreme points
        y_ecsT = (y_ecs-yc)*f/(np.sqrt((x_ecs-xc)**2 + f**2))+yc
        xmax = int(np.max(x_ecsT))
        xmin = int(np.min(x_ecsT))
        ymax = int(np.max(y_ecsT))
        ymin = int(np.min(y_ecsT))
        sizex = int(xmax-xmin)
        sizey = int(ymax-ymin)
        xT_r = np.array(range(sizex))
        xT_r = xT_r+xmin
        yT_r = np.array(range(sizey))
        yT_r = yT_r+ymin
        XT,YT = np.meshgrid(xT_r,yT_r)
        X_og = np.tan((XT-xc)/f)*f + xc				#Applying inverse transformation to map to the original coordinate
        Y_og = (YT - yc)*np.sqrt((X_og-xc)**2 + f**2)/f + yc	
        im1BT = interp2(img1[:,:,0],X_og,Y_og)			#Getting the intensities at the respective locations by interpolation
        im1GT = interp2(img1[:,:,1],X_og,Y_og)
        im1RT = interp2(img1[:,:,2],X_og,Y_og)
        outliersX1 = np.argwhere(X_og>img1.shape[1])
        outliersX2 = np.argwhere(X_og<0)
        outliersY1 = np.argwhere(Y_og>img1.shape[0])
        outliersY2 = np.argwhere(Y_og<0)
        outliers = np.vstack([outliersX1,outliersX2,outliersY1,outliersY2])
        im1BT[outliers[:,0],outliers[:,1]]=0
        im1GT[outliers[:,0],outliers[:,1]]=0
        im1RT[outliers[:,0],outliers[:,1]]=0
        img1T = np.zeros([sizey,sizex,3])
        img1T[:,:,2] = im1BT
        img1T[:,:,1] = im1GT
        img1T[:,:,0] = im1RT
        filename = 'cylin'+str (number) + '.jpeg'
        number = number+1
        imsave(filename,img1T)
