#!/usr/bin/env python3

"""
A place for helper functions for automatic-recolorization, like getting Histograms and color pixels from images
"""

import os
import numpy as np
import cv2
import csv
import struct

methods = [
    "ideepcolor-px-grid",
    "ideepcolor-px-selective",
    "ideepcolor-global",
    "ideepcolor-stock",
]


class Mask(object):
    def __init__(self, size=256, p=1):
        self.size = size
        self.p = p
        self.grid_size = None  # Only needed to generate a filename in Decoder.
        self._init_mask()

    def _init_mask(self):
        self.input_ab = np.zeros((2, self.size, self.size))
        self.mask = np.zeros((1, self.size, self.size))

    def put_point(self, loc, val):
        # input_ab    2x256x256 (size)    current user ab input (will be updated)
        # mask        1x256x256 (size)    binary mask of current user input (will be updated)
        # loc         2 tuple      (h,w) of where to put the user input
        # p           scalar       half-patch size
        # val         2 tuple      (a,b) value of user input
        p = self.p
        self.input_ab[:, loc[0] - p : loc[0] + p + 1, loc[1] - p : loc[1] + p + 1] = np.array(val)[:, np.newaxis, np.newaxis]
        self.mask[:, loc[0] - p : loc[0] + p + 1, loc[1] - p : loc[1] + p + 1] = 1

    def save(self, path, name, grid_size=None, round_to_int=True, method="bytes"):
        """
        :param grid_size: optional, for grids, if set saves grid size, saves on coordinates
        """
        # TODO: round to 2 and change to bitwise save (Saves one bit per a/b)
        save_path = os.path.join(path, gen_new_mask_filename(name))

        if method == "numpy" or method == "np":
            np.savez_compressed(save_path + "np.savez_compressed", a=self.input_ab, b=self.mask)

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

        elif method == "bytes":
            if self.size > 256:
                coord_type = "H"
            else:
                coord_type = "B"
            with open(save_path, "wb") as f:
                # first two Bytes save the mask size. -> coord Byte size and for restoring mask size
                f.write(struct.pack("H", self.size))
                # 3. Byte stores p size and if grid size is saved in next 2 Bytes -> last bit 1 (unsigned char "B")
                third_byte = self.p
                if grid_size:
                    # assuming p size is <128, which would be ridiculous anyway
                    third_byte = third_byte + (1 << 7)  # set last bit to 1
                f.write(struct.pack("B", third_byte))
                # 4. Byte optional grid_size (unsigned char) unlikely to be >255
                if grid_size:
                    f.write(struct.pack("B", grid_size))

                for y in range(self.size):
                    for x in range(self.size):
                        if self.mask[0][y][x] == 0:
                            continue
                        a = int(self.input_ab[0][y][x])
                        b = int(self.input_ab[1][y][x])
                        # Not using grid, save coordinates
                        if grid_size is None:
                            f.write(struct.pack(coord_type, y))
                            f.write(struct.pack(coord_type, x))
                        f.write(struct.pack("b", a))
                        f.write(struct.pack("b", b))

    def load(self, path, name, method="bytes"):
        """
        :param path: Path, where the sidecar file is stored
        :param name: Filename of original or grayscale image
        """
        save_path = os.path.join(path, gen_new_mask_filename(name))
        self._init_mask()
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

                self.put_point((y, x), (a, b))

        elif method == "bytes":
            with open(save_path, "rb") as f:
                # first 2 Byte: mask size
                saved_mask_size = struct.unpack("H", f.read(2))[0]
                # Restore saved mask size
                self.size = saved_mask_size
                self._init_mask()
                # 3. Byte: p size
                third_byte = struct.unpack("B", f.read(1))[0]
                # if lsb is 0 (<128) -> no grid_size saved -> coordinates needed
                self.p = third_byte & 0b1111111
                if (third_byte - self.p) >= (1 << 7):
                    # read optional 4. Byte
                    self.grid_size = struct.unpack("B", f.read(1))[0]

                coord_type, coord_bytes = ( ("H", 2) if saved_mask_size > 256 else ("B", 1) )
                for y in range(self.size):
                    for x in range(self.size):
                        if self.grid_size:
                            if y % self.grid_size != 0 or x % self.grid_size != 0:
                                continue
                        if self.grid_size is None:
                            y = f.read(coord_bytes)
                            x = f.read(coord_bytes)
                            if not all((y, x)):
                                break
                            y = struct.unpack(coord_type, y)[0]
                            x = struct.unpack(coord_type, x)[0]

                        a = f.read(1)
                        b = f.read(1)
                        if not all((a, b)):
                            break
                        a = struct.unpack("b", a)[0]
                        b = struct.unpack("b", b)[0]
                        self.put_point((y, x), (a, b))


# TODO: rename to save_img_lab
def save(path, name, img):
    """Save image to disk"""
    cv2.imwrite(os.path.join(path, name), img[:, :, ::-1])


def save_img(path, name, img):
    cv2.imwrite(os.path.join(path, name), img)


# ideepcolor
# Pixels


def get_fn_wo_ext(filepath):
    filename = os.path.basename(filepath)
    filename_wo_ext, first_extension = os.path.splitext(filename)
    # ensure .gray.png extension is removed fully
    filename_wo_ext, second_extension = os.path.splitext(filename_wo_ext)
    return filename_wo_ext, first_extension, second_extension


def gen_new_gray_filename(orig_fn):
    orig_fn_wo_ext, ext, dummy = get_fn_wo_ext(orig_fn)
    return orig_fn_wo_ext + ".gray" + ext


def gen_new_recolored_filename(orig_fn, method, extras=[]):
    """
    Generates the new filename for recolored files, to save them as.
    :param orig_fn: Original RGB image filename
    :param method: Method, that was used for recolorization, will be appended to name
    :param extras: Optional. All Extras to append to filename (Like mask, grid Size), List
    """
    orig_fn_wo_ext, ext, dummy = get_fn_wo_ext(orig_fn)
    new_fn = orig_fn_wo_ext + "_recolored_" + method
    if extras:
        for e in extras:
            if not e:
                continue
            new_fn = new_fn + "_" + str(e)
    new_fn = new_fn + ext
    return new_fn


def gen_new_mask_filename(input_image_path, extras=None) -> str:
    """
    Generates a new filename without extension by appending the parameters.
    :param extras: if empty: nothing extra, else: String or list with extra info, like plot, size
    """
    new_fn = get_fn_wo_ext(input_image_path)[0]
    # pixel_used = int( (load_size*load_size) / grid_size )
    # new_filename = orig_filename_wo_ext + "_" +  method + "_" + str(load_size) + "_" + str(grid_size) + "_" + str(pixel_used)

    if type(extras) == str:
        new_fn = new_fn + "_" + extras
    elif type(extras) == list:
        for e in extras:
            if not e:
                continue
            new_fn = new_fn + "_" + str(e)

    new_fn = new_fn + ".mask"
    return new_fn


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
    new_filename = orig_filename_wo_ext + "_" + method + "_" + str(load_size)
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
        print(
            "Elements < 256. You are wasting one Byte! Consider saving the index as unsigned char ('B')"
        )

    path = _encode_glob_dist_path(os.path.join(out_path, os.path.basename(name)))
    # wb: write binary
    with open(path, "wb") as f:
        for idx, val in enumerate(glob_dist):
            # don't save elements without content
            if val == 0.0:
                continue
            # h = unsigned short (2 Byte)
            f.write(struct.pack("h", idx))
            # f = float (4 Byte)
            f.write(struct.pack("f", val))
    return path


def load_glob_dist(img_path, elements=313) -> np.ndarray:
    """
    :param img_path: path to image with sidecar .glob_dist file or file itself
    :param elements: length of glob_dist array.
    """
    # TODO: shift elements into 256, if leading & trailing 0 and save offset -> 1 Byte for idx
    glob_dist = np.zeros(elements)
    path = _encode_glob_dist_path(img_path)
    with open(path, "rb") as f:
        while True:
            # h: u short = 2 Byte
            i = f.read(2)
            if not i:
                break
            idx = struct.unpack("h", i)[0]
            # f: float = 4 Byte
            v = f.read(4)
            if not v:
                break
            val = struct.unpack("f", v)[0]
            glob_dist[idx] = val
    return glob_dist


def _encode_glob_dist_path(img_path) -> str:
    img_name = os.path.basename(img_path)
    out_path = os.path.dirname(img_path)
    img_name_wo_ext, extension = os.path.splitext(img_name)
    img_name_wo_ext, dummy = os.path.splitext(img_name_wo_ext)
    new_filename = img_name_wo_ext + ".glob_dist"
    new_path = os.path.join(out_path, new_filename)
    return new_path


def _coord_img_to_mask(h, w, y, x, size=256):
    return (_coord_transform(size, h, y), _coord_transform(size, w, x))


def _coord_mask_to_img(h, w, y, x, size=256):
    return (_coord_transform(h, size, y), _coord_transform(w, size, x))


def _coord_transform(src, target, val):
    return int((src / target) * val)
