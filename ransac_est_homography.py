'''
  File name: ransac_est_homography.py
  Author:
  Date created:
'''

'''
  File clarification:
    Use a robust method (RANSAC) to compute a homography. Use 4-point RANSAC as 
    described in class to compute a robust homography estimate:
    - Input x1, y1, x2, y2: N × 1 vectors representing the correspondences feature coordinates in the first and second image. 
                            It means the point (x1_i , y1_i) in the first image are matched to (x2_i , y2_i) in the second image.
    - Input thresh: the threshold on distance used to determine if transformed points agree.
    - Outpuy H: 3 × 3 matrix representing the homograph matrix computed in final step of RANSAC.
    - Output inlier_ind: N × 1 vector representing if the correspondence is inlier or not. 1 means inlier, 0 means outlier.
'''
import numpy as np
def ransac_est_homography(x, y, X, Y, threshold):
    # Your Code Here
    N = x.size
    A = np.zeros([2 * N, 9])
    ranIter = 10000
    ux = x.reshape(-1,N)
    uy = y.reshape(-1,N)
    uz = np.ones(N).reshape(-1,N)
    uPtsFull = np.vstack([ux,uy,uz])
    vx = X.reshape(-1,N)
    vy = Y.reshape(-1,N)
    vz = np.ones(N).reshape(-1,N)
    vPtsFull = np.vstack([vx,vy,vz])
    i = 0
    maxInlierCount = 0
    distanceSum = 999999999
    # populates A with points
    while i < N:
        a = np.array([x[i], y[i], 1]).reshape(-1, 3)
        c = np.array([[X[i]], [Y[i]]])
        d = - c * a
        A[2 * i, 0 : 3], A[2 * i + 1, 3 : 6]= a, a
        A[2 * i : 2 * i + 2, 6 : ] = d
        i += 1
    # print("A Matrix: ", end=" ")
    # print(A)
    # print("---------")
  # compute the solution of A
    ptindices = []
    inlier_ind = []
    for i in range(ranIter):
        At = np.zeros([8,9])
        indexList = list(range(N))
        backup = []
        for j in range(4):
            rnd = np.random.randint(0,len(indexList))
            ind = indexList.pop(rnd)
            backup.append(ind)
            # print(ind,end=" ")
            At[2*j,:]=A[2*ind,:]
            At[2*j +1,:] = A[2*ind+1,:]
        # print("")
        # print(At)
        [u,d,v] = np.linalg.svd(At,full_matrices=True)
        minIdx = np.min(d)
        # print(minIdx)
        Ht = v[-1,:]/v[-1,-1]
        H = Ht.reshape(3,3)
        vPtsPred = np.matmul(H,uPtsFull)
        vPtsPred = vPtsPred/vPtsPred[-1,:]
        distances = np.sqrt(((vPtsPred-vPtsFull)**2).sum(axis=0))
        inliers = 1*np.less_equal(distances,threshold)
        dSum = np.sum(distances[inliers])
        inlierCount = np.sum(1*inliers)
        if (maxInlierCount == inlierCount and distanceSum>dSum) or maxInlierCount<inlierCount:
            maxInlierCount = inlierCount
            ptindices = np.argwhere(inliers==1)
            inlier_ind = inliers
            distanceSum = dSum
            print(dSum)
    ptindices = ptindices[:,0]
    # print(maxInlierCount)
    Anew = np.zeros([2*ptindices.shape[0],9])
    j=0
    for i in range(ptindices.shape[0]):
        Anew[2*j,:] = A[2*ptindices[i],:]
        Anew[2*j+1,:] = A[2*ptindices[i]+1,:]
        j=j+1
    U, s, V = np.linalg.svd(Anew, full_matrices=True)
    h = V[-1, :]/V[-1,-1]
    # print(h)
    print("MaxInlierCount" + str(maxInlierCount))
    H = h.reshape(3, 3)
    return H, inlier_ind
