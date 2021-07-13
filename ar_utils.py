#!/usr/bin/env python3

"""
A place for helper functions for automatic-recolorization, like getting Histograms and color pixels from images
"""

import os
from pickle import STRING
import numpy as np
import cv2
import csv
import struct

methods = ["ideepcolor-px-grid", "ideepcolor-px-selective", "ideepcolor-global", "HistoGAN"]

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
        # TODO: also save size and p values
        # TODO: change to bitwise save
        save_path = os.path.join(path, gen_new_mask_filename(name))

        if method == "numpy" or method == "np":
            np.savez_compressed(save_path+"np.savez_compressed", a=self.input_ab, b=self.mask)

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
        """
        :param path: Path, where the sidecar file is stored
        :param name: Filename of original or grayscale image
        """
        save_path = os.path.join(path, gen_new_mask_filename(name))
        self._init_mask(self.size)
        if method == "numpy":
            loaded = np.load(save_path)
            self.input_ab = loaded["a"]
            self.mask = loaded["b"]

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

            
# TODO: rename to save_img_lab
def save(path, name, img):
    """Save image to disk"""
    cv2.imwrite(os.path.join(path, name), img[:, :, ::-1])
def save_img(path, name, img):
    cv2.imwrite(os.path.join(path, name), img)

# ideepcolor
# Pixels
def get_color_mask(img, grid_size=100, size=256, p=0):
    """
    :param img: original color image as lab (lab, y, x)
    :param grid_size: distance between pixels of grid in pixels 0-256 (mask size)
    :return Mask: Mask of pixels
    """
    # np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

    # print(img[2][0][0])

    mask = Mask(size=size, p=p)

    h = len(img[0])
    w = len(img[0][0])

    for y in range(size):
        if y % grid_size != 0:
            continue
        for x in range(size):
            if x % grid_size != 0:
                continue
            y_img, x_img = _coord_mask_to_img(h, w, y, x, size)
            mask.put_point((y, x), [ img[1][y_img][x_img], img[2][y_img][x_img] ])

    # mask.put_point([135,160], 3, [100,-69])
    # print(mask.input_ab)
    return mask

def gen_new_gray_filename(orig_fn):
    orig_fn_wo_ext, ext, dummy = get_fn_wo_ext(orig_fn)
    return orig_fn_wo_ext + ".gray" + ext

def gen_new_recolored_filename(orig_fn, method):
    orig_fn_wo_ext, ext, dummy = get_fn_wo_ext(orig_fn)
    new_fn = orig_fn_wo_ext + "_recolored_" + method + ext
    return new_fn

def gen_new_mask_filename(input_image_path) -> str:
    """
    Generates a new filename without extension by appending the parameters.
    """
    orig_filename_wo_ext = get_fn_wo_ext(input_image_path)[0]
    # pixel_used = int( (load_size*load_size) / grid_size )
    # new_filename = orig_filename_wo_ext + "_" +  method + "_" + str(load_size) + "_" + str(grid_size) + "_" + str(pixel_used)
    new_filename = orig_filename_wo_ext + ".mask"
    return new_filename

def get_fn_wo_ext(filepath):
    filename = os.path.basename(filepath)
    filename_wo_ext, first_extension = os.path.splitext(filename)
    # ensure .gray.png extension is removed fully
    filename_wo_ext, second_extension = os.path.splitext(filename_wo_ext)
    return filename_wo_ext, first_extension, second_extension


# DEPRECATED
def gen_new_hist_filename(method, input_image_path, load_size) -> str:
    """
    Generates a new filename without extension by appending the parameters.
    :param method:
    """
    orig_filename = os.path.basename(input_image_path)
    orig_filename_wo_ext, extension = os.path.splitext(orig_filename)
    # ensure .gray.png extension is removed fully
    orig_filename_wo_ext, extension = os.path.splitext(orig_filename_wo_ext)
    new_filename = orig_filename_wo_ext + "_" +  method + "_" + str(load_size)
    return new_filename

def save_glob_dist(out_path, name, glob_dist, elements=313) -> str:
    """
    :param path: output folder
    :param name: either original image filename or full path to original image
    :return: str New path + filename it was saved to
    """
    # TODO: check if elements after 256 are empty and only save 256 first with 1 index Byte
    if elements > 512:
        warn = "Warning: elements are bigger than 512 and wont fit in 2 Bytes"
        print(warn)
        raise Exception(warn)
    if elements < 256:
        print("Elements < 256. You are wasting one Byte! Consider saving the index as unsigned char ('B')")

    path = _encode_glob_dist_path(out_path, name)
    # wb: write binary
    with open("path", "wb") as f:
        for idx, val in enumerate(glob_dist):
            # don't save elements without content
            if val == 0.:
                continue
            # h = unsigned short (2 Byte)
            f.write(struct.pack('h', idx))
            # f = float (4 Byte)
            f.write(struct.pack('f', val))
    return path

def load_glob_dist(out_path, name, elements=313) -> np.ndarray:
    """
    :param path: output folder
    :param name: either original image filename or full path to original image
    :param elements: length of glob_dist array.
    """
    glob_dist = np.zeros(elements)
    path = _encode_glob_dist_path(out_path, name)
    with open(path, "rb") as f:
        while True:
            # h: u short = 2 Byte
            i = f.read(2)
            if not i:
                break
            idx = struct.unpack('h', i)[0]
            # f: float = 4 Byte
            v = f.read(4)
            if not v:
                break
            val = struct.unpack('f', v)[0]
            glob_dist[idx] = val
    return glob_dist



def _encode_glob_dist_path(out_path, image) -> str:
    img_name = os.path.basename(image)
    img_name_wo_ext, extension = os.path.splitext(img_name)
    new_filename = img_name_wo_ext + ".glob_dist"
    new_path = os.path.join(out_path, new_filename)
    return new_path

def _coord_img_to_mask(h, w, y, x, size=256):
    return (_coord_transform(size, h, y), _coord_transform(size, w, x))

def _coord_mask_to_img(h, w, y, x, size=256):
    return (_coord_transform(h, size, y), _coord_transform(w, size, x))

def _coord_transform(src, target, val):
    return int((src/target)*val)

