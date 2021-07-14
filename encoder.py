#!/usr/bin/env python3
"""
"""


import os, sys
import argparse
import cv2
import numpy as np
from skimage import color
from PIL import Image
import ar_utils
import importlib

class Encoder(object):
    def __init__(self, output_path="intermediate_representation", method=ar_utils.methods[0], size=256, p=0, grid_size=10) -> None:
        self.methods = ar_utils.methods
        self.method = method
        self.watch = False
        self.size = size
        self.p = p
        self.grid_size = grid_size
        # self.input_path = input_path
        self.output_path = output_path

        sys.path.insert(1, os.path.abspath("./interactive-deep-colorization/caffe_files"))

    def main(self):
        parser = argparse.ArgumentParser(prog='Recolor Encoder',
                                            description='Encodes images, to be decoded by Recolor')

        parser.add_argument('-o', '--output_path', action='store', dest='output_path', type=str,
                               default='intermediate_representation',
                               help='The path to the folder or file, where the grayscale version and color information will be written to')
        parser.add_argument('-i', '--input_path', action='store', dest='input_path', type=str, default='input_images',
                               help='Path to individual image, or folder with images')
        parser.add_argument('-m', '--method', action='store', dest='method', type=str, default=self.methods[0],
                            help='The colorization method to use. Possible values: \"' + ', '.join(self.methods) + '\"')
        parser.add_argument('-w','--watch', dest='watch', help='watch input folder for new images', action='store_true')

        # for ideepcolor-px
        parser.add_argument('-s', '--size', action='store', dest='size', type=int, default=256,
                               help='Size of the indermediate mask to store the color pixels. Power of 2. \
                               The bigger, the more accurate the result, but requires more storage, and RAM capacity (decoder) \
                               (For 2048 up to 21GB RAM)')
        parser.add_argument('-g', '--grid_size', action='store', dest='grid_size', type=int, default=10,
                               help='Spacing between color pixels in intermediate mask (--size)')
        parser.add_argument('-p', '--p', action='store', dest='p', type=int, default=0,
                               help='The "radius" the color values will have. \
                               A higher value means one color pixel will later cover multiple gray pixels. Default: 0')

        args = parser.parse_args()
        self.watch = args.watch
        self.size = args.size
        self.grid_size = args.grid_size
        self.p = args.p
        self.output_path = args.output_path

        try:
            os.mkdir(self.output_path)
        except FileExistsError:
            pass

        # TODO: implement watch functionality
        if not os.path.isdir(args.input_path):
            try:
                Image.open(args.input_path) # Just to test if file is image
                self.encode(args.input_path)
            except IOError as err:
                print("Error: File is not a image file: " + args.input_path)
        else:
            for fil in os.scandir(args.input_path):
                if os.path.isdir(fil):
                    continue
                fil.path
                try:
                    # to check if file is valid image
                    Image.open(fil.path)
                    self.encode(fil.path)
                except IOError as err:
                    print("Warning: Found non image file: " + fil.path)
                    pass




    def load_image(self, path):
        # load image to lab
        img_rgb = cv2.cvtColor(cv2.imread(path, 1), cv2.COLOR_BGR2RGB)
        img_lab_fullres = color.rgb2lab(img_rgb).transpose((2, 0, 1))
        return img_lab_fullres

    def load_image_to_gray(self, path):
        return cv2.cvtColor(cv2.imread(path, 1), cv2.COLOR_BGR2GRAY)

    def encode(self, img_path):
        """
        Executes the right encoding method depending on self.method set.
        Converts img to grayscale and saves in self.output_path
        :return:
        """
        img_lab_fullres = self.load_image(img_path)
        img_gray = self.load_image_to_gray(img_path)
        ar_utils.save_img(self.output_path, ar_utils.gen_new_gray_filename(img_path), img_gray)


        mask = ar_utils.Mask()

        if self.method == "ideepcolor-px-grid":
            filename_mask = ar_utils.gen_new_mask_filename(img_path)
            # mask = ar_utils.Mask(size, p)
            mask = ar_utils.get_color_mask(img_lab_fullres, self.grid_size, self.size, self.p)
            # TODO: consider ditching gen. filenames and just use .mask ext
            mask.save(self.output_path, os.path.basename(filename_mask))
            return mask

        elif self.method == "ideepcolor-px-selective":
            # TODO: implement
            pass

        elif self.method == "ideepcolor-global":
            self.encode_ideepcolor_global(img_path, self.size)

        elif self.method == "HistoGAN":
            # TODO: implement
            pass


    def encode_ideepcolor_global(self, img_path, size) -> np.ndarray:
        os.environ['GLOG_minloglevel'] = '2' # suprress Caffe verbose prints
        import caffe
        lab = importlib.import_module("interactive-deep-colorization.data.lab_gamut")

        prev_wd = os.getcwd()
        os.chdir('./interactive-deep-colorization')
        # models need to be downloaded before, using "interactive-deep-colorization/models/fetch_models.sh"
        global_stats_model = os.path.abspath('./models/global_model/global_stats.prototxt')
        weights = os.path.abspath('./models/global_model/dummy.caffemodel')
        gt_glob_net = caffe.Net(global_stats_model, 1, weights=weights)

        # load image
        ref_img_fullres = caffe.io.load_image(os.path.abspath(img_path))
        img_glob_dist = (255*caffe.io.resize_image(ref_img_fullres,(size,size))).astype('uint8')
        gt_glob_net.blobs['img_bgr'].data[...] = img_glob_dist[:,:,::-1].transpose((2,0,1))
        gt_glob_net.forward()
        glob_dist_in = gt_glob_net.blobs['gt_glob_ab_313_drop'].data[0,:-1,0,0].copy()
        os.chdir(prev_wd)

        ar_utils.save_glob_dist(self.output_path, img_path, glob_dist_in)
        return glob_dist_in



if __name__ == "__main__":
    ec = Encoder()
    ec.main()
