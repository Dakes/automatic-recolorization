#!/usr/bin/env python3
import os, sys
import argparse
import ar_utils
import importlib
from PIL import Image

CI = importlib.import_module("interactive-deep-colorization.data.colorize_image")

class Decoder(object):
    def __init__(self, output_path="output_images", gpu_id=-1, method=ar_utils.methods[0], size=256, p=0, plot=False) -> None:
        self.gpu_id = None if gpu_id < 0 else gpu_id
        self.methods = ar_utils.methods
        self.method = method
        self.watch = False
        self.size = size
        # set default size for global mode
        if self.method == self.methods[2]:
            self.size = 256
        self.p = p
        self.plot = plot
        # self.input_path = input_path
        self.output_path = output_path
        # lower CPU priority (to not freeze PC)
        os.nice(19)
        try:
            os.makedirs(self.output_path, exist_ok=True)
        except FileExistsError:
            pass

        self.maskcent = False
        self.color_model = 'colorization-pytorch/checkpoints/siggraph_caffemodel/latest_net_G.pth'
        self.caffe_net = "./models/reference_model/deploy_nodist.prototxt"
        self.caffe_model = "./models/reference_model/model.caffemodel"
        self.global_prototxt = "./models/global_model/deploy_nodist.prototxt"
        self.global_caffemodel = "./models/global_model/global_model.caffemodel"

        sys.path.insert(1, os.path.abspath("interactive-deep-colorization"))
        os.environ['GLOG_minloglevel'] = '2'  # supress Caffe verbose prints


    def main(self):
        parser = argparse.ArgumentParser(
            prog="Recolor Decoder", description="Encodes images, to be decoded by Recolor"
        )

        parser.add_argument(
            "-o", "--output_path",
            action="store",
            dest="output_path",
            type=str,
            default="output_images",
            help="The path to the folder or file, where the grayscale version and color information will be written to",
        )
        parser.add_argument(
            "-i", "--input_path",
            action="store",
            dest="input_path",
            type=str,
            default="intermediate_representation",
            help="Path to individual grayscale image with color sidecar file, or folder with multiple. ",
        )
        parser.add_argument(
            "-m", "--method",
            action="store",
            dest="method",
            type=str,
            default=ar_utils.methods[0],
            help='The colorization method to use. Possible values: "'
            + ", ".join(ar_utils.methods) + '"',
        )
        parser.add_argument(
            "-w", "--watch",
            dest="watch",
            help="watch input folder for new images",
            action="store_true",
        )

        parser.add_argument(
            "-plt", "--plot",
            dest="plot",
            help="Generate Plots for visualization",
            action="store_true",
        )
        
        args = parser.parse_args()
        self.method = args.method
        self.watch = args.watch
        self.size = args.size
        # self.grid_size = args.grid_size
        self.output_path = args.output_path
        self.plot = args.plot

        try:
            os.makedirs(self.output_path, exist_ok=True)
        except FileExistsError:
            pass

        # TODO: implement watch functionality
        if not os.path.isdir(args.input_path):
            try:
                Image.open(args.input_path) # Just to test if file is image
                self.decode(args.input_path)
            except IOError as err:
                print("Error: File is not an image file: " + args.input_path)
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
        if "ideepcolor-px" in self.method:
            # filename_mask = ar_utils.gen_new_mask_filename(img_gray_path)
            self.decode_ideepcolor_px(img_gray_path)

        elif self.method == "ideepcolor-global":
            self.decode_ideepcolor_global(img_gray_path)

        # ideepcolor-stock
        elif self.method == ar_utils.methods[3]:
            # same as global, but without global hints
            self.decode_ideepcolor_global(img_gray_path, stock=True)
        
        else:
            print("Error: method not valid:", self.method)

    def decode_ideepcolor_px(self, img_gray_path, model="pytorch"):
        
        mask = ar_utils.Mask(self.size, self.p)

        # "ideepcolor-px-grid+selective"
        if self.method == ar_utils.methods[5]:
            mask.load(os.path.dirname(img_gray_path), os.path.basename(img_gray_path), name_extra="1", initialize=True)
            mask.grid_size = None
            mask.load(os.path.dirname(img_gray_path), os.path.basename(img_gray_path), name_extra="2", initialize=False)
        else:
            mask.load(os.path.dirname(img_gray_path), os.path.basename(img_gray_path))

        prev_wd = os.getcwd()
        if model == "pytorch":
            colorModel = CI.ColorizeImageTorch(Xd=mask.size, maskcent=self.maskcent)
            gpu_id = None if self.gpu_id < 0 else self.gpu_id
            colorModel.prep_net(path=os.path.abspath(self.color_model), gpu_id=gpu_id)
        elif model == "caffe":
            ideepcolor_folder = "./interactive-deep-colorization"
            # check if already in folder
            if not os.path.basename(ideepcolor_folder) == os.path.basename(os.getcwd()):
                os.chdir(ideepcolor_folder)
            colorModel = CI.ColorizeImageCaffe(Xd=mask.size)
            colorModel.prep_net(self.gpu_id, self.caffe_net, self.caffe_model)
            
        colorModel.load_image(img_gray_path)

        img_out = colorModel.net_forward(mask.input_ab, mask.mask)
        img_out_fullres = colorModel.get_img_fullres()

        os.chdir(prev_wd)

        self._save_img_out(img_gray_path, img_out_fullres, extras=[mask.size, mask.grid_size])
        new_rc_mask_filename = None
        # only save plot for grid method, selective has its own
        if self.plot and (self.method == ar_utils.methods[0] or self.method == ar_utils.methods[4] or self.method == ar_utils.methods[5]):
            img_mask_fullres = colorModel.get_input_img_fullres()
            # img_real_mask_fullres = colorModel.get_img_mask_fullres()
            self._save_img_out(img_gray_path, img_mask_fullres,
                               extras=[mask.size, mask.grid_size, ".mask_rgb"])
            # self._save_img_out(img_gray_path, img_real_mask_fullres,
            #                    extras=[mask.size, mask.grid_size, ".mask_rgb_real"])

        return (img_out_fullres, new_rc_mask_filename)

    def decode_ideepcolor_global(self, img_gray_path, stock=False):
        # TODO: don't recreate cid
        img_gray_abspath = os.path.abspath(img_gray_path)

        prev_wd = os.getcwd()
        ideepcolor_folder = "./interactive-deep-colorization"
        # check if already in folder
        if not os.path.basename(ideepcolor_folder) == os.path.basename(os.getcwd()):
            os.chdir(ideepcolor_folder)
        
        cid = CI.ColorizeImageCaffeGlobDist(self.size)
        cid.prep_net(self.gpu_id,
                     prototxt_path=self.global_prototxt,
                     caffemodel_path=self.global_caffemodel)
        cid.load_image(img_gray_abspath)
        dummy_mask = ar_utils.Mask(self.size)
        if not stock:
            glob_dist = ar_utils.load_glob_dist(img_gray_path)
            img_pred = cid.net_forward(dummy_mask.input_ab, dummy_mask.mask, glob_dist)
        else:
            img_pred = cid.net_forward(dummy_mask.input_ab, dummy_mask.mask)
        img_out_fullres = cid.get_img_fullres()
        os.chdir(prev_wd)

        self._save_img_out(img_gray_path, img_out_fullres)
        return img_out_fullres

    def _save_img_out(self, img_gray_path, img, method=None, extras=None):
        if method is None:
            method = self.method
        
        new_rc_filename = ar_utils.gen_new_recolored_filename(img_gray_path, method, extras)
        ar_utils.save(self.output_path, new_rc_filename, img)


if __name__ == "__main__":
    dc = Decoder()
    dc.main()
