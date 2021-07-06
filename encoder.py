#!/usr/bin/env python3

import os
import argparse
import cv2
from skimage import color
from PIL import Image
import ar_utils

class Encoder(object):
    def __init__(self, method="ideepcolor-px") -> None:
        self.methods = ["ideepcolor-px", "ideepcolor-hist", "HistoGAN"]
        self.method = method
        self.watch = False

    def main(self):
        parser = argparse.ArgumentParser(prog='Recolor Encoder',
                                            description='Encodes images, to be decoded by Recolor')

        # Add the arguments
        parser.add_argument('-o', '--output_path', action='store', dest='output_path', type=str,
                               default='output_images',
                               help='The path to the folder or file, where the output will be written to... What did you expect?')
        parser.add_argument('-i', '--input_path', action='store', dest='input_path', type=str,
                               default='input_images',
                               help='Path to individual image, or folder with images (with --watch)')
        parser.add_argument('-m', '--method', action='store', dest='method', type=str, default=self.methods[0],
                            help='The colorization method to use. Possible values: \"' + ', '.join(self.methods) + '\"')
        parser.add_argument('-w','--watch', dest='watch', help='watch input folder for new images', action='store_true')

        args = parser.parse_args()
        self.watch = args.watch

        if not os.path.isdir(args.input_path):
            # TODO: check if image
            self.encode(args.input_path, args.output_path, args.output_path)

        # recursive colorize pictures in folder
        elif os.path.isdir(args.input_path):
            try:
                os.mkdir(args.output_path)
            except FileExistsError:
                pass
            for root, d_names, f_names in os.walk(args.input_path):
                for file_name in f_names:
                    file_path = os.path.join(root, file_name)
                    relative_path = os.path.relpath(file_path, args.input_path)
                    output_file = os.path.join(args.output_path, relative_path)

                    out_folder = os.path.dirname(output_file)
                    try:
                        os.mkdir(out_folder)
                    except FileExistsError:
                        pass

                    try:
                        # to check if valid image
                        Image.open(file_path)
                        self.img_recolor(args, file_path, out_folder) # TODO
                    except IOError as err:
                        pass


        self.encode()

    def load_image(self, path):
        # load image to lab
        img_rgb = cv2.cvtColor(cv2.imread(path, 1), cv2.COLOR_BGR2RGB)
        img_lab_fullres = color.rgb2lab(img_rgb).transpose((2, 0, 1))
        return img_lab_fullres


    def encode(self, img_path, out_folder, size=256, grid_size=10, p=0, method="ideepcolor-px-grid"):
        """
        Loads a color image, extracts the color pixels using the given parameters and saves it as a csv.
        :return: Mask
        """
        filename = ar_utils.gen_new_filename(img_path, size, grid_size, method)

        img_lab_fullres = self.load_image(img_path)
        mask = ar_utils.Mask()

        if method == "ideepcolor-px-grid":
            # mask = ar_utils.Mask(size, p)
            mask = ar_utils.get_color_mask(img_lab_fullres, grid_size, size, p)
        mask.save(out_folder, os.path.splitext(filename)[0])
        return mask


if __name__ == "__main__":
    ec = Encoder()
    ec.main()
