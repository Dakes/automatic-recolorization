#!/usr/bin/env python3

"""
A place for helper functions for automatic-recolorization, like getting Histograms and color pixels from images
"""

import os
import numpy as np
import cv2
import csv

class Mask(object):
    def __init__(self, size=256, p=1):
        self._init_mask(size)
        self.size = size
        self.p = p

    def _init_mask(self, size):
        self.input_ab = np.zeros((2, size, size))
        self.mask = np.zeros((1, size, size))


    def put_point(self, loc, val):
        # input_ab    2x256x256 (size)    current user ab input (will be updated)
        # mask        1x256x256 (size)    binary mask of current user input (will be updated)
        # loc         2 tuple      (h,w) of where to put the user input
        # p           scalar       half-patch size
        # val         2 tuple      (a,b) value of user input
        p = self.p
        if p is not None:
            self.input_ab[:,loc[0]-p:loc[0]+p+1,loc[1]-p:loc[1]+p+1] = np.array(val)[:,np.newaxis,np.newaxis]
            self.mask[:,loc[0]-p:loc[0]+p+1,loc[1]-p:loc[1]+p+1] = 1
        else:
            # broken
            self.input_ab[:, loc[0], loc[1]] = np.array(val)[:,np.newaxis,np.newaxis]
            self.mask[:, loc[0], loc[1]] = 1


    def save(self, path, name, round_to_int=True, method="csv"):
        save_path = os.path.join(path, name+".mask")

        if method == "numpy" or method == "np":
            np.savez_compressed(save_path+"np.savez_compressed", self.input_ab, self.mask)

        elif method == "csv":
            # header = ["y", "x", "a", "b"]
            with open(save_path + ".csv", "w") as f:
                writer = csv.writer(f, delimiter=";")
                # writer.writerow(header)
                for y in range(self.size):
                    for x in range(self.size):
                        if self.mask[0][y][x] == 0:
                            continue
                        a = self.input_ab[0][y][x] if not round_to_int else int(self.input_ab[0][y][x])
                        b = self.input_ab[1][y][x] if not round_to_int else int(self.input_ab[1][y][x])
                        row = [y, x, a, b]
                        writer.writerow(row)

        # TODO
        elif method == "bytes":
            mask_file = open(save_path + ".csv")
            mask_file.write("y;x;mask;a;b")
            for y in range(self.size):
                for x in range(self.size):
                    if self.mask[0][x][y] == 0:
                        continue
                    mask_file.write("")

            mask_file.close()


    def load(self, path, name, method="csv"):
        save_path = os.path.join(path, name+".mask")
        self._init_mask(self.size)
        if method == "csv":
            with open(save_path + ".csv") as f:
                reader = csv.reader(f, delimiter=";")
                data = list(reader)
            for row in data:
                y = int(row[0])
                x = int(row[1])

                try:
                    a = int(row[2])
                    b = int(row[3])
                except ValueError:
                    a = float(row[2])
                    b = float(row[3])

                self.put_point((y,x), (a,b))

            

def save(path, name, img):
    """Save image to disk"""
    cv2.imwrite(os.path.join(path, name), img[:, :, ::-1])

# ideepcolor
# Pixels
def get_color_mask(img, grid_size=100, size=256):
    """
    :param img: original color image as lab (lab, y, x)
    :param grid_size: distance between pixels of grid in pixels 0-256 (mask size)
    :return Mask: Mask of pixels
    """
    # np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

    # print(img[2][0][0])

    mask = Mask(size=size)

    h = len(img[0])
    w = len(img[0][0])

    for y in range(size):
        if y % grid_size != 0:
            continue
        for x in range(size):
            if x % grid_size != 0:
                continue
            # print(img[1][y][x], img[2][y][x])
            y_img, x_img = _coord_mask_to_img(h, w, y, x, size)
            mask.put_point((y, x), [ img[1][y_img][x_img], img[2][y_img][x_img] ], p=0 )

    # mask.put_point([135,160], 3, [100,-69])
    # print(mask.input_ab)
    return mask

def _coord_img_to_mask(h, w, y, x, size=256):
    return (_coord_transform(size, h, y), _coord_transform(size, w, x))

def _coord_mask_to_img(h, w, y, x, size=256):
    return (_coord_transform(h, size, y), _coord_transform(w, size, x))

def _coord_transform(src, target, val):
    return int((src/target)*val)

