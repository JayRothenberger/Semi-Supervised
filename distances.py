import numpy as np
"""
distance functions must take three arguments (img0, img1, top_1) and output a real number 

top_1 is the top_1 confidence for the prediction

Jay.C.Rothenberger@ou.edu
"""


def euclidean(img0, img1, top_1):
    # euclidean distance
    return np.linalg.norm(img0 - img1)


def euclidean_with_confidence(img0, img1, top_1, threshold=.5):
    # euclidean distance
    if 1 - top_1 > threshold:
        return np.linalg.norm(img0 - img1)
    else:
        return np.linalg.norm(256*np.ones_like(img0))*(top_1 + 1)


def confidence(img0, img1, top_1):
    # instead of distance between two images we measure the absolute confidence of a prediction
    return 1 - top_1
