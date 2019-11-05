import cv2
import numpy as np
from scipy import signal
'''
  File name: feat_match.py
  Author:
  Date created:
'''

'''
  File clarification:
    Matching feature descriptors between two images. You can use k-d tree to find the k nearest neighbour. 
    Remember to filter the correspondences using the ratio of the best and second-best match SSD. You can set the threshold to 0.6.
    - Input descs1: 64 × N1 matrix representing the corner descriptors of first image.
    - Input descs2: 64 × N2 matrix representing the corner descriptors of second image.
    - Outpuy match: N1 × 1 vector where match i points to the index of the descriptor in descs2 that matches with the
                    feature i in descriptor descs1. If no match is found, you should put match i = −1.
'''

def feat_match(descs1, descs2):
    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
    matches = matcher.knnMatch(np.float32(descs1.T), np.float32(descs2.T), 2)
    match = []
    dMatch = []
    for d1,d2 in matches:
        if d1.distance < 0.7*d2.distance:
            match.append(d1.trainIdx)
            dMatch.append(d1)
        else:
            match.append(-1)
    return match,dMatch
# def feat_match(descs1, descs2):
#     match = []
#     dMatch = []
#     descs1 = descs1.T
#     descs2 = descs2.T
#     for desc in descs1:
#         diff = descs2-desc
#         norms = np.linalg.norm(diff,axis=1)
#         idxs = norms.argsort()
#         if norms[idxs[0]] < 0.7*norms[idxs[1]]:
#             dMatch.append(cv2.DMatch(len(match),idxs[0],norms[idxs[0]]))
#             match.append(idxs[0])
#         else:
#             match.append(-1)
#     return match,dMatch
