#!/usr/bin/env python3
import os, sys
import argparse
import ar_utils
import importlib
from PIL import Image

CI = importlib.import_module("interactive-deep-colorization.data.colorize_image")

class Decoder(object):
    def __init__(self, output_path="output_images", gpu_id=-1, method=ar_utils.methods[0], size=256, p=0, display_mask=False) -> None:
        self.gpu_id = gpu_id
        self.methods = ar_utils.methods
        self.method = method
        self.watch = False
        self.size = size
        self.p = p
        self.display_mask = display_mask
        # self.input_path = input_path
        self.output_path = output_path

        self.maskcent = False
        self.color_model = 'colorization-pytorch/checkpoints/siggraph_caffemodel/latest_net_G.pth'

        sys.path.insert(1, os.path.abspath("interactive-deep-colorization"))


    def main(self):
        parser = argparse.ArgumentParser(prog='Recolor Decoder',
                                            description='Encodes images, to be decoded by Recolor')

        parser.add_argument('-o', '--output_path', action='store', dest='output_path', type=str,
                               default='output_images',
                               help='The path to the folder or file, where the grayscale version and color information will be written to')
        parser.add_argument('-i', '--input_path', action='store', dest='input_path', type=str, default='intermediate_representation',
                               help='Path to individual grayscale image with color sidecar file, or folder with multiple. ')
        parser.add_argument('-d', '--display_mask', action='store_true', dest='display_mask',
                               help='Whether to save the Intermediate Representation mask as an image for visualization. Only works with ideepcolor-px method. ')
        parser.add_argument('-m', '--method', action='store', dest='method', type=str, default=ar_utils.methods[0],
                            help='The colorization method to use. Possible values: \"' + ', '.join(ar_utils.methods) + '\"')
        parser.add_argument('-w','--watch', dest='watch', help='watch input folder for new images', action='store_true')

        # for ideepcolor-px
        parser.add_argument('-s', '--size', action='store', dest='size', type=int, default=256,
                               help='Size of the indermediate mask to store the color pixels. Power of 2. \
                               The bigger, the more accurate the result, but requires more storage, and RAM capacity (decoder) \
                               (For 2048 up to 21GB RAM)')
        # parser.add_argument('-g', '--grid_size', action='store', dest='grid_size', type=int, default=10,
                               # help='Sacing between color pixels in intermediate mask (--size)')
        parser.add_argument('-p', '--p', action='store', dest='p', type=int, default=0,
                               help='The "radius" the color values will have. \
                               A higher value means one color pixel will later cover multiple gray pixels. Default: 0')
        args = parser.parse_args()
        self.method = args.method
        self.watch = args.watch
        self.size = args.size
        # self.grid_size = args.grid_size
        self.p = args.p
        self.display_mask = args.display_mask
        self.output_path = args.output_path

        try:
            os.mkdir(self.output_path)
        except FileExistsError:
            pass

        # TODO: implement watch functionality
        if not os.path.isdir(args.input_path):
            try:
                Image.open(args.input_path) # Just to test if file is image
                self.decode(args.input_path)
            except IOError as err:
                print("Error: File is not a image file: " + args.input_path)
        else:
            for fil in os.scandir(args.input_path):
                if os.path.isdir(fil.path):
                    continue
                try:
                    # to check if file is valid image
                    Image.open(fil.path) # Just to test if file is image
                    self.decode(fil.path)
                except IOError as err:
                    # print("Warning: Found non image file: " + fil.path)
                    pass


    def decode(self, img_gray_path):
        if self.method == "ideepcolor-px-grid":
            # filename_mask = ar_utils.gen_new_mask_filename(img_gray_path)
            self.decode_ideepcolor_px(img_gray_path)

        elif self.method == "ideepcolor-px-selective":
            # TODO: implement
            pass

        elif self.method == "ideepcolor-global":
            self.decode_ideepcolor_global(img_gray_path)

        elif self.method == "HistoGAN":
            # TODO: implement
            pass

    def decode_ideepcolor_px(self, img_gray_path):
        mask = ar_utils.Mask(self.size, self.p)
        mask.load(os.path.dirname(img_gray_path), os.path.basename(img_gray_path))

        colorModel = CI.ColorizeImageTorch(Xd=mask.size, maskcent=self.maskcent)
        colorModel.prep_net(path=os.path.abspath(self.color_model), gpu_id=self.gpu_id)

        colorModel.load_image(img_gray_path)

        img_out = colorModel.net_forward(mask.input_ab, mask.mask)
        img_out_fullres = colorModel.get_img_fullres()
        self._save_img_out(img_gray_path, img_out_fullres)
        new_rc_mask_filename = None
        if self.display_mask:
            img_mask_fullres = colorModel.get_input_img_fullres()
            self._save_img_out(img_gray_path, img_mask_fullres, method=self.method+"_mask")

        return (img_out_fullres, new_rc_mask_filename)

    def decode_ideepcolor_global(self, img_gray_path):
        glob_dist = ar_utils.load_glob_dist(img_gray_path)
        img_gray_abspath = os.path.abspath(img_gray_path)

        prev_wd = os.getcwd()
        os.chdir('./interactive-deep-colorization')
        cid = CI.ColorizeImageCaffeGlobDist(self.size)
        cid.prep_net(self.gpu_id, prototxt_path='./models/global_model/deploy_nodist.prototxt',
                     caffemodel_path='./models/global_model/global_model.caffemodel')
        cid.load_image(img_gray_abspath)
        # dummy Mask
        dummy_mask = ar_utils.Mask(self.size)
        img_pred = cid.net_forward(dummy_mask.input_ab, dummy_mask.mask, glob_dist)
        img_out_fullres = cid.get_img_fullres()
        os.chdir(prev_wd)

        self._save_img_out(img_gray_path, img_out_fullres)
        return img_out_fullres

    def _save_img_out(self, img_gray_path, img, method=None):
        if method is None:
            method = self.method
        new_rc_filename = ar_utils.gen_new_recolored_filename(img_gray_path, method)
        ar_utils.save(self.output_path, new_rc_filename, img)


if __name__ == "__main__":
    dc = Decoder()
    dc.main()
