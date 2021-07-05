#!/usr/bin/env python3

"""
A place for helper functions for automatic-recolorization, like getting Histograms and color pixels from images
"""

import os
import numpy as np
import cv2

class Mask(object):
    def __init__(self, size=256):
        self.input_ab = np.zeros((2, size, size))
        self.mask = np.zeros((1, size, size))

    def put_point(self, loc, p, val):
        # input_ab    2x256x256    current user ab input (will be updated)
        # mask        1x256x256    binary mask of current user input (will be updated)
        # loc         2 tuple      (h,w) of where to put the user input
        # p           scalar       half-patch size
        # val         2 tuple      (a,b) value of user input
        self.input_ab[:,loc[0]-p:loc[0]+p+1,loc[1]-p:loc[1]+p+1] = np.array(val)[:,np.newaxis,np.newaxis]
        self.mask[:,loc[0]-p:loc[0]+p+1,loc[1]-p:loc[1]+p+1] = 1

def save(path, name, img):
    """Save image to disk"""
    cv2.imwrite(os.path.join(path, name), img[:, :, ::-1])

# ideepcolor
# Pixels
def get_color_pixels(img, grid_size=100):
    """
    :param img: original color image
    :param grid_size: distance between pixels of grid in pixels
    :return Mask: Mask of pixels
    """
    pass
