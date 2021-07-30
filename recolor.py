#!/usr/bin/env python3

import argparse
import importlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os, sys
import ar_utils, encoder, decoder

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
        self.methods = ar_utils.methods
        self.method = self.methods[0]
        # True for retrained, false for caffe model
        self.maskcent = False
        self.load_size = 256

        # Whether to save the mask of colorization pixels
        self.input_mask = True
        # Whether to open an extra window with the output
        self.show_plot = False


    def main(self):
        parser = argparse.ArgumentParser(prog='Recolor', description='TODO')

        parser.add_argument('-o', '--output_path', action='store', dest='output_path', type=str,
                               default='output_images',
                               help='The path to the folder or file, where the output will be written to. ')
        parser.add_argument('-i', '--input_path', action='store', dest='input_path', type=str,
                               default='input_images',
                               help='The path to the folder with input color images... What did you expect?')
        parser.add_argument('-ir', '--intermediate_representation', action='store', dest='intermediate_representation', type=str,
                               default='intermediate_representation',
                               help='The path, where the grayscale images + color cues will be stored')
        parser.add_argument('-m', '--method', action='store', dest='method', type=str, default=self.methods[0],
                            help='The colorization method to use. Possible values: \"' + ', '.join(self.methods) + '\"')

        # parser.add_argument('--color_model', dest='color_model', help='colorization model (color & dist for Pytorch)', type=str,
        #                 default='colorization-pytorch/checkpoints/siggraph_caffemodel/latest_net_G.pth')
        # 'colorization-pytorch/checkpoints/siggraph_retrained/latest_net_G.pth'

        # for ideepcolor-px
        parser.add_argument('-s', '--size', action='store', dest='size', type=int, default=256,
                               help='Size of the indermediate mask to store the color pixels. Power of 2. \
                               The bigger, the more accurate the result, but requires more storage, and RAM capacity (decoder) \
                               (For 2048 up to 21GB RAM)')
        parser.add_argument('-g', '--grid_size', action='store', dest='grid_size', type=int, default=10,
                               help='Spacing between color pixels in intermediate mask (--size).  -1: fill every spot in mask.  0: dont use any color pixel ')
        parser.add_argument('-p', '--p', action='store', dest='p', type=int, default=0,
                               help='The "radius" the color values will have. \
                               A higher value means one color pixel will later cover multiple gray pixels. Default: 0')
        parser.add_argument('-plt','--plot', dest='plot', help='Generate Plots for visualization', action='store_true')

        # TODO: test gpu on cuda gpu
        parser.add_argument('--gpu_id', dest='gpu_id', help='gpu id', type=int, default=-1)
        # TODO: remove?
        parser.add_argument('--cpu_mode', dest='cpu_mode', help='do not use gpu', action='store_true')
        # parser.add_argument('--pytorch_maskcent', dest='pytorch_maskcent', help='need to center mask (activate for siggraph_pretrained but not for converted caffemodel)', action='store_true')
        parser.add_argument('--save_mask', dest='save_mask', help='save mask of input pixels as image (ideepcolor-px)', action='store_true')


        args = parser.parse_args()
        # self.maskcent = args.pytorch_maskcent
        # self.show_plot = args.show_plot

        if args.cpu_mode:
            self.gpu_id = -1
            args.gpu_id = -1


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

            self.img_recolor(args, args.input_path)
            # try:
            #     # to check if valid image
            #     Image.open(args.input_path)
            #     self.img_recolor(args, args.input_path)
            # except IOError as err:
            #     print(err)
            #     sys.exit()


        # colorize all pictures in folder
        elif os.path.isdir(args.input_path):
            try:
                os.mkdir(args.output_path)
                os.mkdir(args.intermediate_representation)
            except FileExistsError as err:
                pass
            for root, d_names, f_names in os.walk(args.input_path):
                for file_name in f_names:
                    file_path = os.path.join(root, file_name)
                    relative_path = os.path.relpath(file_path, args.input_path)
                    output_file = os.path.join(args.output_path, relative_path)

                    # out_folder = os.path.dirname(output_file)
                    # try:
                    #     os.mkdir(out_folder)
                    # except FileExistsError:
                    #     pass

                    try:
                        # to check if valid image
                        Image.open(file_path)
                        self.img_recolor(args, file_path) # TODO
                    except IOError as err:
                        print(err)
                        pass


    def img_recolor(self, args, input_image_path):
        """
        Performs Encoding and Decoding at once
        """
        ec = encoder.Encoder(output_path=args.intermediate_representation, method=args.method, size=args.size, p=args.p, grid_size=args.grid_size, plot=args.plot)
        dc = decoder.Decoder(output_path=args.output_path, method=args.method, size=args.size, p=args.p, gpu_id=args.gpu_id)

        ec.encode(input_image_path)
        img_gray_name = ar_utils.gen_new_gray_filename(input_image_path)
        img_gray_path = os.path.join(args.intermediate_representation, img_gray_name)
        dc.decode(img_gray_path)

        # img_out_fullres = None
        # img_in_fullres = None
        # new_filename = None
        # if self.method == "ideepcolor-px":
        #     img_out_fullres, img_in_fullres, new_filename = self.ideepcolor_px_recolor(args, input_image_path, output_folder)

        # # TODO: save parameters in sidecar file (col. method, pixel grid size, or specific pixels, density)
        # ar_utils.save(output_folder, new_filename, img_out_fullres)

        # if self.show_plot:
        #     # show user input, along with output
        #     plt.figure(figsize=(10,6))
        #     if self.input_mask:
        #         plt.imshow(np.concatenate((img_in_fullres, img_out_fullres), axis=1))
        #         plt.title('Input grayscale with auto points / Output colorization')
        #     else:
        #         plt.imshow(np.concatenate((img_out_fullres, ), axis=1))
        #         plt.title('Output colorization')
        #     plt.axis('off')
        #     plt.show()

    # DEPRECATED replaced by decoder and encoder
    def ideepcolor_px_recolor(self, args, input_image_path, output_folder):
        """
        ideepcolor pixel colorization method
        TODO: extend to use different pixel extraction methods
        :return: tuple of colorized image and input pixel mask (as image) and new filename (second is optional, dependend on self.input_mask)
        """
        # TODO: automate grid_size and size to get certain density
        grid_size = 10

        # generate new filename with parameters used
        new_filename = ar_utils.gen_new_px_grid_filename(input_image_path, self.load_size, grid_size, self.method)

        colorModel = CI.ColorizeImageTorch(Xd=self.load_size, maskcent=self.maskcent)
        colorModel.prep_net(path=os.path.abspath(args.color_model), gpu_id=args.gpu)


        colorModel.load_image(input_image_path)
        orig_lab_img = colorModel.img_lab # img_lab is sizexsize, img_lab_fullres fullres, obviously
        h = len(orig_lab_img[0])
        w = len(orig_lab_img[0][0])

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

    # DEPRECATED replaced by encoder and decoder
    def ideepcolor_hist_recoler(self, args, input_image_path, output_folder):
        # generate new filename with parameters used
        new_filename = ar_utils.gen_new_hist_filename(input_image_path, self.load_size, self.method)
        distModel = CI.ColorizeImageTorchDist(Xd=self.load_size, maskcent=self.maskcent)
        distModel.prep_net(path=os.path.abspath(args.color_model), dist=True, gpu_id=args.gpu)
        distModel.load_image(input_image_path)
        # orig_lab_img = distModel.img_lab

    




if __name__ == "__main__":
    rc = Recolor()
    rc.main()
