#!/usr/bin/env python3
"""
"""


import os, sys
import argparse
from typing_extensions import ParamSpecArgs
import cv2
import numpy as np
from skimage import color
from PIL import Image
import ar_utils
import importlib

class Encoder(object):
    def __init__(self, output_path="intermediate_representation", method=ar_utils.methods[0], size=256, p=0, grid_size=10, plot=False) -> None:
        self.methods = ar_utils.methods
        self.method = method
        self.watch = False
        self.size = size
        # set default size for global mode
        if self.method == self.methods[2]:
            self.size = 256
        self.p = p
        self.grid_size = grid_size
        # self.input_path = input_path
        self.output_path = output_path
        self.plot = plot
        try:
            os.mkdir(self.output_path)
        except FileExistsError:
            pass

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
                               help='Spacing between color pixels in intermediate mask (--size)  1: fill every spot in mask.  0: dont use any color pixel ')
        parser.add_argument('-p', '--p', action='store', dest='p', type=int, default=0,
                               help='The "radius" the color values will have. \
                               A higher value means one color pixel will later cover multiple gray pixels. Default: 0')
        parser.add_argument('-plt','--plot', dest='plot', help='Generate Plots for visualization', action='store_true')

        args = parser.parse_args()
        self.watch = args.watch
        self.size = args.size
        self.grid_size = args.grid_size
        self.p = args.p
        self.output_path = args.output_path
        self.method = args.method
        self.plot = args.plot

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




    def load_image(self, path, colorspace="lab"):
        if colorspace == "lab":
            img_rgb = cv2.cvtColor(cv2.imread(path, 1), cv2.COLOR_BGR2RGB)
            img_lab_fullres = color.rgb2lab(img_rgb).transpose((2, 0, 1))
            return img_lab_fullres
        elif colorspace == "rgb":
            img = cv2.cvtColor(cv2.imread(path, 1), cv2.COLOR_BGR2RGB)
            return img

    def load_image_to_gray(self, path):
        return cv2.cvtColor(cv2.imread(path, 1), cv2.COLOR_BGR2GRAY)

    def encode(self, img_path):
        """
        Executes the right encoding method depending on self.method set.
        Converts img to grayscale and saves in self.output_path
        :return:
        """
        self.image_path = img_path
        img_lab_fullres = self.load_image(img_path)
        img_gray = self.load_image_to_gray(img_path)
        ar_utils.save_img(self.output_path, ar_utils.gen_new_gray_filename(img_path), img_gray)


        # mask = ar_utils.Mask()

        if "ideepcolor-px" in self.method:
            filename_mask = ar_utils.gen_new_mask_filename(img_path)
            mask = None
            if self.method == "ideepcolor-px-grid":
                mask = self.get_color_mask_grid(img_lab_fullres, self.grid_size, self.size, self.p)
                mask.save(self.output_path, os.path.basename(filename_mask), grid_size=self.grid_size)
            elif self.method == "ideepcolor-px-selective":
                mask = self.get_color_mask_selective(img_lab_fullres, img_path)
                mask.save(self.output_path, os.path.basename(filename_mask))

        elif self.method == "ideepcolor-global":
            self.encode_ideepcolor_global(img_path, self.size)

        # ideepcolor-stock: no encoding necessary
        elif self.method == ar_utils.methods[3]:
            pass

        elif self.method == "HistoGAN":
            # TODO: implement
            pass
        else:
            print("Error: method not valid:", self.method)



    def encode_ideepcolor_global(self, img_path, size) -> np.ndarray:
        os.environ['GLOG_minloglevel'] = '2' # suprress Caffe verbose prints
        import caffe
        lab = importlib.import_module("interactive-deep-colorization.data.lab_gamut")

        img_path = os.path.abspath(img_path)
        prev_wd = os.getcwd()

        ideepcolor_folder = "./interactive-deep-colorization"
        # check if already in folder
        if not os.path.basename(ideepcolor_folder) == os.path.basename(os.getcwd()):
            os.chdir(ideepcolor_folder)
            
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

    
    def get_color_mask_grid(self, img, grid_size=100, size=256, p=0):
        """
        :param img: original color image as lab (lab, y, x)
        :param grid_size: distance between pixels of grid in pixels 0 - mask size (-1: every space filled, 0: None filled (stock coloring))
        :return Mask: Mask of pixels
        """
        # TODO: save plot of grid
        mask = ar_utils.Mask(size=size, p=p)
        if grid_size == 0:
            return mask

        h = len(img[0])
        w = len(img[0][0])


        for y in range(size):
            if y % grid_size != 0:
                continue
            for x in range(size):
                if x % grid_size != 0:
                    continue
                y_img, x_img = ar_utils._coord_mask_to_img(h, w, y, x, size)
                mask.put_point((y, x), [ img[1][y_img][x_img], img[2][y_img][x_img] ])

        # mask.put_point([135,160], 3, [100,-69])
        # print(mask.input_ab)
        return mask


    # Everything for selective color mask

    def get_color_mask_selective(self, img, img_path, round_to=10, scaling_factor=None):
        from skimage import filters, color, restoration, util, transform
        # PARAM: hardcoded, round_to (for cityscapes rather smaller (8). Default: 10)
        # PARAM: hardcoded, scaling_factor: 8 for highres, or higher. 4, 2 for cityscapes and low res
        if not scaling_factor:
            scaling_factor = int( min(img.shape[-2:])/250 ) # cityscapes(vga; w:480) -> 2, dragon_pool(w:2370) -> 9
        print("Scaling factor: ", scaling_factor)

        mask = ar_utils.Mask(size=self.size, p=self.p)
        # reload as rgb 0-255
        rgb = self.load_image(img_path, colorspace="rgb")

        # Median Filter; remove extreme individual noise pixels
        # PARAM: k for median blur
        k = 5
        rgb = cv2.medianBlur(rgb, k)

        # for picking out color values without noise later
        lab_median = self.rgb_to_lab(rgb)
        a_median = lab_median[1].astype(int)
        b_median = lab_median[2].astype(int)
        
        # scale down image
        img_resized = transform.resize(rgb,
                                       (img.shape[1] // scaling_factor, img.shape[2] // scaling_factor),
                                       anti_aliasing=True)

        # Bilateral Filter; Edge preserving blur
        # PARAM: sigma_spatial, (/500)
        sigma_spatial = min(img.shape[-1:]) / 250
        print("Sigma Spatial (Bilateral)", sigma_spatial)
        # PARAM: sigma_color: sig-default*100
        sigma_color = restoration.estimate_sigma(img_resized)*100
        # img_resized = cv2.bilateralFilter(np.uint8(img_resized), -1, 2, 10)
        print("Sigma Color (Bilateral)", sigma_color)
        img_resized = restoration.denoise_bilateral(img_resized, multichannel=True,
                                                    sigma_spatial=sigma_spatial,
                                                    sigma_color=sigma_color)


        img_resized = self.rgb_to_lab(img_resized)

        a_orig = img[1].astype(int)
        b_orig = img[2].astype(int)
        L = img_resized[0].astype(int)
        a = img_resized[1].astype(int)
        b = img_resized[2].astype(int)


        # shift ab into positive
        a = a.astype(int)+100
        b = b.astype(int)+100
        
        # Gaussian blur; smooth out colors a bit more, reduces points overall
        # PARAM: calculated sigma
        sigma = min(a.shape)/250 # Gaussian (/250)
        print("Sigma Gaussian:", sigma)
        a = filters.gaussian(a, sigma, preserve_range=True)
        b = filters.gaussian(b, sigma, preserve_range=True)



        # shift back to ab space -100-100
        a = util.img_as_ubyte(a.astype(int)).astype(int)-100
        b = util.img_as_ubyte(b.astype(int)).astype(int)-100

        ab = self.get_ab(a, b, round_to)
        ab_ids = self.set_color_area_ids(ab)
        centres = self.get_centres(ab_ids)

        # delete points near edges, since sometimes they tend to bunch on the edges, and those are not super important for colorization anyway
        h, w = a.shape
        dist = 2 # distance from edges
        keep = np.ones(len(centres), dtype=bool)
        for idx, c in enumerate(centres):
            if c[0] < dist or c[0] > h-dist or c[1] < dist or c[1] > w-dist:
                keep[idx] = False
        centres = centres[keep]
        
        # Save image with red dots for selected pixels
        if self.plot:
            import matplotlib.pyplot as plt
            rgb = self.lab_to_rgb(img)
            plt.imshow(rgb)
            y = np.array( [row[0] for row in centres] )*scaling_factor
            x = np.array( [row[1] for row in centres] )*scaling_factor
            plt.scatter(x=x, y=y, c='r', s=1)
            # TODO: save plot
            plt_fn = ar_utils.gen_new_mask_filename(self.image_path, "selective_plot")
            plt_path = os.path.join(self.output_path, plt_fn)
            plt.savefig(plt_path+".png", bbox_inches='tight', dpi=1500)
            plt.clf()
            plt.close()

        # Use found interesting pixels as coordinates to fill mask
        h, w = a_orig.shape
        for px in centres:
            # scale up to resolution of input image
            loc = (px[0]*scaling_factor, px[1]*scaling_factor)
            # use colors from median filtered image
            val = (a_median[loc], b_median[loc])
            loc = ar_utils._coord_img_to_mask(h, w, loc[0], loc[1], size=self.size)
            mask.put_point(loc, val)

        return mask

    def round_arr_to(self, arr, r_to=20):
        """
        Rounds numpy array to nearest r_to
        """
        return np.around(arr/r_to, decimals=0)*r_to

    def get_ab(self, a, b, round_to=10):
        """
        Returns an array combined of a and b channels, where each color value has a unique value
        """
        an = self.round_arr_to(a, round_to)
        bn = self.round_arr_to(b, round_to)
        # shift into positive
        an = an + 100
        bn = bn + 100
        ab = an.astype(int)*1000 + bn.astype(int)
        ab = ab.astype(int)
        # ab = np.array([ab, make_arr(ab, l=None)])
        ab = np.array(ab, dtype="uint32")
        return ab

    def flood_fill(self, a, yx, newval):
        from skimage.measure import label
        a = np.array(a)
        y, x = yx
        l = label(a==a[y, x])
        a[l==l[y, x]] = newval
        return a

    def set_color_area_ids(self, ab):
        """
        Replaces every seperate blob of a color with a unique id
        """
        # replace pixel values via ff with unique id
        id_ = 0
        # to make cv2.floodFill work
        # ab = np.ascontiguousarray(ab, dtype=np.uint8)
        unique_colors = np.unique(ab)
        for col in unique_colors:
            # while as long as there is this color in the array
            while np.where(ab==col)[0].size:
                found_pos = np.where(ab==col)
                y = found_pos[0][0]
                x = found_pos[1][0]
                # run ff from this pixel and give all connected same colors the same id
                ab = self.flood_fill(ab, (y, x), id_)
                # cv2.floodFill(ab_uint8, None, (x, y), id)
                id_ = id_+1
        return ab

    def get_centres(self, ab_ids):
        """
        Returns the most centre pixel position of every blob with a unique id
        """
        import scipy.spatial.distance
        ids =  np.unique(ab_ids)
        centres = []

        for id in ids:
            area_coords = np.where(ab_ids == id)
            area_coords_y = area_coords[0]
            area_coords_x = area_coords[1]
            # get centre
            centre = ( int((sum(area_coords_y)/len(area_coords_y)))
                    , int((sum(area_coords_x)/len(area_coords_x))) )

            # since centre could be outside shape, search nearest point to centre
            closest = None
            dist = float('inf')
            # TODO: get more points if area is above certain size
            # NOTE: this is really slow, if centre not in shape. Maybe just use a random point. 
            for idx, i in enumerate(area_coords_y):
                # break if calculated centre is inside area
                if centre[0] in area_coords_y and centre[1] in area_coords_x:
                    closest = centre
                    break
                
                n_dist = distance.euclidean((area_coords_y[idx], area_coords_x[idx]), centre)
                if n_dist < dist:
                    dist = n_dist
                    closest = (area_coords_y[idx], area_coords_x[idx])
            centres.append(closest)
        return np.array(centres)

    def rgb_to_lab(self, rgb):
        return color.rgb2lab(rgb).transpose((2, 0, 1))

    def lab_to_rgb(self, *args):
        """
        Either takes lab array of shape (3, h, w) or as 3 separate 2D Arrays, l, a, b
        """
        lab = None
        if len(args) == 1:
            lab = args[0]
        elif len(args) == 3:
            lab = [args[0], args[1], args[2]]
        else:
            print("Wrong number of arguments in lab_to_rgb. ")
            return None
        return np.fliplr(np.rot90(color.lab2rgb(np.transpose(lab)), 3))



if __name__ == "__main__":
    ec = Encoder()
    ec.main()
