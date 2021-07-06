#!/usr/bin/env python3

import argparse
import importlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os, sys
import ar_utils, encoder

# -'s in import not allowed
# ideepcolor = importlib.import_module("interactive-deep-colorization")
CI = importlib.import_module("interactive-deep-colorization.data.colorize_image")
# CICIT = importlib.import_module("interactive-deep-colorization.data.colorize_image.ColorizeImageTorch")
# ideepcolor_pytorch = importlib.import_module("colorization-pytorch")

sys.path.insert(1, os.path.abspath("interactive-deep-colorization"))

# TODO: add full auto mode: watch folder for new images
# TODO: add config file parser. Encoder, Decoder etc.
# TODO: Encoder, Decoder
class Recolor(object):
    def __init__(self):
        self.methods = ["ideepcolor-px", "ideepcolor-hist", "HistoGAN"]
        self.method = self.methods[0]
        # True for retrained, false for caffe model
        self.maskcent = False
        self.load_size = 2048 # 256

        # Whether to save the mask of colorization pixels
        self.input_mask = True
        # Whether to open an extra window with the output
        self.show_plot = False


    def main(self):
        parser = argparse.ArgumentParser(prog='Recolor',
                                            description='TODO')

        # Add the arguments
        parser.add_argument('-o', '--output_path', action='store', dest='output_path', type=str,
                               default='output_images',
                               help='The path to the folder or file, where the output will be written to... What did you expect?')
        parser.add_argument('-i', '--input_path', action='store', dest='input_path', type=str,
                               default='input_images',
                               help='TODO')
        parser.add_argument('-m', '--method', action='store', dest='method', type=str, default=self.methods[0],
                            help='The colorization method to use. Possible values: \"' + ', '.join(self.methods) + '\"')
        parser.add_argument('--color_model', dest='color_model', help='colorization model (color & dist for Pytorch)', type=str,
                        default='colorization-pytorch/checkpoints/siggraph_caffemodel/latest_net_G.pth')
        # 'colorization-pytorch/checkpoints/siggraph_retrained/latest_net_G.pth'

        # TODO: test gpu on cuda gpu
        parser.add_argument('--gpu', dest='gpu', help='gpu id', type=int, default=None)
        # TODO: remove?
        parser.add_argument('--cpu_mode', dest='cpu_mode', help='do not use gpu', action='store_true')
        parser.add_argument('--pytorch_maskcent', dest='pytorch_maskcent', help='need to center mask (activate for siggraph_pretrained but not for converted caffemodel)', action='store_true')
        parser.add_argument('--show_plot', dest='show_plot', help='show pyplot plot of images', action='store_true')
        parser.add_argument('--decode_only', dest='decode_only', help='only decode, use pre-encoded CSVs', action='store_true')


        args = parser.parse_args()
        self.maskcent = args.pytorch_maskcent
        self.show_plot = args.show_plot


        if args.method not in self.methods:
            print("Method not valid. One of: \"" + ', '.join(self.methods) + '\"')
            sys.exit(1)
        self.method = args.method

        if not os.path.isdir(args.input_path):
            if not os.path.isfile(args.input_path):
                print('The input_path is not a directory or file')
                sys.exit(1)

        if not os.path.isdir(args.input_path):
            # TODO: check if image
            self.img_recolor(args, args.input_path, args.output_path)

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


    def img_recolor(self, args, input_image_path, output_folder):
        img_out_fullres = None
        img_in_fullres = None
        new_filename = None
        if self.method == "ideepcolor-px":
            img_out_fullres, img_in_fullres, new_filename = self.ideepcolor_px_recolor(args, input_image_path, output_folder)

        # TODO: save parameters in sidecar file (col. method, pixel grid size, or specific pixels, density)
        ar_utils.save(output_folder, new_filename, img_out_fullres)

        if self.show_plot:
            # show user input, along with output
            plt.figure(figsize=(10,6))
            if self.input_mask:
                plt.imshow(np.concatenate((img_in_fullres, img_out_fullres), axis=1))
                plt.title('Input grayscale with auto points / Output colorization')
            else:
                plt.imshow(np.concatenate((img_out_fullres, ), axis=1))
                plt.title('Output colorization')
            plt.axis('off')
            plt.show()


    def ideepcolor_px_recolor(self, args, input_image_path, output_folder):
        """
        ideepcolor pixel colorization method
        TODO: extend to use different pixel extraction methods
        :return: tuple of colorized image and input pixel mask (as image) and new filename (second is optional, dependend on self.input_mask)
        """
        # TODO: automate grid_size and size to get certain density
        grid_size = 10

        # generate new filename with parameters used
        new_filename = ar_utils.gen_new_filename(input_image_path, self.load_size, grid_size, self.method)

        colorModel = CI.ColorizeImageTorch(Xd=self.load_size, maskcent=self.maskcent)
        colorModel.prep_net(path=os.path.abspath(args.color_model), gpu_id=args.gpu)

        # distModel = CI.ColorizeImageTorchDist(Xd=self.load_size, maskcent=self.maskcent)
        # distModel.prep_net(path=os.path.abspath(args.color_model), dist=True, gpu_id=args.gpu)

        colorModel.load_image(input_image_path)
        orig_lab_img = colorModel.img_lab # img_lab is sizexsize, img_lab_fullres fullres, obviously
        h = len(orig_lab_img[0])
        w = len(orig_lab_img[0][0])


        # TODO: automate points and get value from original
        # initialize with no user inputs
        ec = encoder.Encoder()
        mask = ec.encode(input_image_path, output_folder, self.load_size, grid_size, p=0, method="ideepcolor-px-grid")

        # call forward
        img_out = colorModel.net_forward(mask.input_ab, mask.mask)

        # get mask, input image, and result in full resolution
        # mask_fullres = colorModel.get_img_mask_fullres() # get input mask in full res
        img_out_fullres = colorModel.get_img_fullres() # get image at full resolution



        if self.input_mask:
            img_mask_fullres = colorModel.get_input_img_fullres() # get input image with pixel mask in full res
            return (img_out_fullres, img_mask_fullres, new_filename)
        return (img_out_fullres, None, new_filename)




if __name__ == "__main__":
    rc = Recolor()
    rc.main()
