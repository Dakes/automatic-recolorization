#!/usr/bin/env python3

"""
Calculates the Image Quality using PSNR and MS-SSIM (by  default) recursively and writes them to a plain text .org file in the same directory.
"""

import os, sys
import numpy as np
import argparse
# from sewar import full_ref
import cv2
from PIL import Image
import concurrent.futures
from multiprocessing import Pool
from pathlib import Path
import warnings
from fast_qa.fast_qa import ssim, ms_ssim, vif_spatial
import lpips
import torch

from skimage import color
import skimage
from packaging import version
if version.parse(skimage.__version__) < version.parse("0.17.0"):
    from skimage.measure import compare_ssim as ssim_sk
    from skimage.measure import compare_psnr as psnr_sk
else:
    from skimage.metrics import structural_similarity as ssim_sk
    from skimage.metrics import peak_signal_noise_ratio as psnr_sk

class ImageQuality(object):
    def __init__(self, in_path="output_images", reference_path="../pictures/", out_file="image_quality.org",
                 recursive=True, skip=False, truncate=4, ab=False, format_org=False,
                 no_header_name=False, ssim=False, vif=False):
        self.in_path = in_path
        self.ref_path = reference_path
        self.out_file = out_file
        self.recursive = recursive
        self.cpus = os.cpu_count()
        self.skip = skip
        self.truncate = truncate
        self.ab = ab
        self.format_org = format_org
        self.ssim = ssim
        self.vif = vif
        self.no_header_name = no_header_name

        # Disable Complex to float casting warning
        warnings.filterwarnings('ignore')
        # lower CPU priority (to not freeze PC)
        os.nice(19)

    def main(self):
        parser = argparse.ArgumentParser(
            prog="Image Quality Calculater",
            description="Computes the Image Quality between generated and original image, using MS-SSIM and PSNR",
        )

        parser.add_argument(
            "-i", "--input_path",
            action="store",
            dest="input_path",
            type=str,
            default=self.in_path,
            help="Path to folder with recolored images. ",
        )
        parser.add_argument(
            "-r", "--reference_path",
            action="store",
            dest="reference_path",
            type=str,
            default=self.ref_path,
            help="Path to folder with original rgb images",
        )
        parser.add_argument(
            "-o", "--output_file",
            action="store",
            dest="output_file",
            type=str,
            default=self.out_file,
            help="The name of the file, where the quality results will be written to. Default: image_quality.org",
        )
        parser.add_argument(
            "-n", "--non-recursive",
            dest="non_recursive",
            help="Just look in input_path, non recursively",
            action="store_true",
        )
        parser.add_argument(
            "-s", "--skip",
            dest="skip",
            help="skip folder, if output .org file already exists",
            action="store_true",
        )
        parser.add_argument(
            "-t", "--truncate",
            dest="truncate",
            type=int,
            help="Truncate output float values to n digits. Max precision: 16 digits. Default: 4",
            default=4,
        )
        parser.add_argument(
            "-ab", "--ab",
            dest="ab",
            help="Calculate values on a&b Lab color channels only. ",
            action="store_true",
        )
        parser.add_argument(
            "-org", "--format_org",
            dest="format_org",
            help="Format produces tables in org files, using Emacs. Requires Emacs to be installed. ",
            action="store_true",
        )
        parser.add_argument(
            "--no_header_name", 
            dest="no_header_name",
            help="Don't use Headers for every source file name (** Name) and put everything into one table. \n\
            Good for averaging values, when only one type of modified image, but many are in one dir. For that add a formula in Emacs org-mode",
            action="store_true",
        )
        parser.add_argument(
            "-ms-ssim", "--ms-ssim",
            dest="msssim",
            help="Calculate SSIM. ",
            action="store_true",
        )
        parser.add_argument(
            "-ssim", "--ssim",
            dest="ssim",
            help="Calculate SSIM. ",
            action="store_true",
        )
        parser.add_argument(
            "-psnr", "--psnr",
            dest="psnr",
            help="Calculate SSIM. ",
            action="store_true",
        )
        parser.add_argument(
            "-vif", "--vif",
            dest="vif",
            help="Calculate VIF spatial. ",
            action="store_true",
        )
        parser.add_argument(
            "-lpips", "--lpips",
            dest="lpips",
            help="Calculate LPIPS (Learned Perceptual Image Patch Similarity). Warning: only works on RGB. if --ab is set, calculates on RGB. ",
            action="store_true",
        )

        args = parser.parse_args()
        self.in_path = args.input_path
        self.ref_path = args.reference_path
        self.out_file = args.output_file
        self.recursive = not args.non_recursive
        self.skip = args.skip
        self.truncate = args.truncate
        self.ab = args.ab
        self.format_org = args.format_org
        self.no_header_name = args.no_header_name
        
        self.msssim = args.msssim
        self.ssim = args.ssim
        self.psnr = args.psnr
        self.vif = args.vif
        self.lpips = args.lpips

        # set default methods, if non are given
        if not self.msssim and not self.ssim and not self.psnr and not self.vif and not self.lpips:
            self.msssim = True
            self.psnr = True
            self.lpips = True

        if self.lpips:
            # TODO: add spatial as parameter
            self.loss_fn = lpips.LPIPS(net='alex', verbose=True)  # Can also set net = 'squeeze' or 'vgg' squeeze: more lightweight
            if self.ab:
                print("Warning LPIPS only runs on RGB. LPIPS will run on RGB, the rest on the ab channels. ")

        self.get_and_write_quality()

    def get_and_write_quality(self):
        """
        To be executed, when imported. 
        """
        # TODO: make it efficiently use multithreading, generate huge list of src & target image / folder -> MT that
        ref_paths, ref_names = self.get_ref_paths_names()
        
        # iterate through all subfolders of in_path
        for root, dirs, files in os.walk(self.in_path):
            # root: dir in which to place .org file later
            # files: list of all files in root
            # iterate through all input files and search for recolored versions
            if self.skip and Path(os.path.join(root, self.out_file)).exists():
                print("File '" + os.path.join(root, self.out_file) + "' already exists, skipping. ")
                if not self.recursive:
                    break
                else:
                    continue

            print("Now in: ", root)
            qualities = []
            for idx, ref_name in enumerate(ref_names):
                files_matching_ref = self.find_files(ref_name, root)
                if not files_matching_ref:
                    continue
                # qualities: Array of dictionaries
                qualities = qualities + self.calc_quality(ref_paths[idx], files_matching_ref)
            self.write_quality(qualities, ref_names, os.path.join(root, self.out_file))
            if not self.recursive:
                break

    def calc_quality(self, ref_path, recolored_paths):
        """
        :param ref_path: reference image full path
        :param recolored_paths: Array of image paths to get quality to
        :return: Array of dictionaries 2D-Array [[path, PSNR, MS-SSIM], ...]
        """

        if type(recolored_paths) is str:
            recolored_paths = [recolored_paths]

        mp_args = []
        for i in recolored_paths:
            # skip mask plot visulizations
            # TODO: check properly if file is image
            if ".mask" in i or ".glob_dist" in i:
                continue
            mp_args.append((ref_path, i))
        
        qualities = self.run_multiprocessing(self.calc_quality_image, mp_args)
        
        return qualities

    def calc_quality_image(self, ref_path, rec_path):
        """
        Calculate quality measures for single image, parallelized
        :param ref_img: already loaded reference img
        :param rec_path: path to recolored image
        :return: Dictionary. {"File": recolor.png, "Metric": value, ...}
        """
        ref_img = cv2.cvtColor(cv2.imread(ref_path, 1), cv2.COLOR_BGR2RGB)
        img = cv2.cvtColor(cv2.imread(rec_path, 1), cv2.COLOR_BGR2RGB)
        result = {}

        if self.lpips:
            ref_tensor = lpips.im2tensor(lpips.load_image(ref_path))
            rec_tensor = lpips.im2tensor(lpips.load_image(rec_path))

            lpips_val = self.loss_fn.forward(ref_tensor, rec_tensor)
            result["LPIPS"] = float(lpips_val)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            

            # RGB
            if not self.ab:
                if self.psnr:
                    psnr = executor.submit(psnr_sk, ref_img, img)
                    psnr_r = executor.submit(psnr_sk, ref_img[0], img[0])
                    psnr_g = executor.submit(psnr_sk, ref_img[1], img[1])
                    psnr_b = executor.submit(psnr_sk, ref_img[2], img[2])

                    psnr_r = psnr_r.result()
                    psnr_g = psnr_g.result()
                    psnr_b = psnr_b.result()

                    psnr = np.mean([psnr_r, psnr_g, psnr_b])
                    result["PSNR"] = psnr

                if self.msssim:
                    msssim = executor.submit(full_ref.msssim,
                                                ref_img, img,
                                                # default values
                                                weights=[0.0448, 0.2856, 0.3001, 0.2363, 0.1333],
                                                ws=11, K1=0.01, K2=0.03, MAX=None)
                    
                    msssim_r = executor.submit(ms_ssim, ref_img[0], img[0], max_val=255)
                    msssim_g = executor.submit(ms_ssim, ref_img[1], img[1], max_val=255)
                    msssim_b = executor.submit(ms_ssim, ref_img[2], img[2], max_val=255)
                    msssim_r = msssim_r.result()
                    msssim_g = msssim_g.result()
                    msssim_b = msssim_b.result()

                    msssim = np.mean([msssim_r, msssim_g, msssim_b])
                    result["MS-SSIM"] = msssim
                    
                if self.ssim:
                    ssim_r = executor.submit(ssim, ref_img[0], img[0], max_val=255)
                    ssim_g = executor.submit(ssim, ref_img[1], img[1], max_val=255)
                    ssim_b = executor.submit(ssim, ref_img[2], img[2], max_val=255)
                    ssim_r = ssim_r.result()
                    ssim_g = ssim_g.result()
                    ssim_b = ssim_b.result()
                    ssim_fast_qa = np.mean([ssim_r, ssim_g, ssim_b])
                    result["SSIM"] = ssim_fast_qa

                if self.vif:
                    vif_spatial_r = executor.submit(vif_spatial, ref_img[0], img[0], max_val=255)
                    vif_spatial_g = executor.submit(vif_spatial, ref_img[1], img[1], max_val=255)
                    vif_spatial_b = executor.submit(vif_spatial, ref_img[2], img[2], max_val=255)
                    vif_spatial_r = vif_spatial_r.result()
                    vif_spatial_g = vif_spatial_g.result()
                    vif_spatial_b = vif_spatial_b.result()
                    vif = np.mean([vif_spatial_r, vif_spatial_g, vif_spatial_b])
                    result["VIF-SPATIAL"] = vif

            # ab only
            else:
                img_lab = color.rgb2lab(img).transpose((2, 0, 1)) + 100
                img_lab = img_lab.astype(int)
                ref_img_lab = color.rgb2lab(ref_img).transpose((2, 0, 1)) + 100
                ref_img_lab = ref_img_lab.astype(int)

                
                if self.psnr:
                    psnr_a = executor.submit(psnr_sk, ref_img_lab[1], img_lab[1], data_range=200)
                    psnr_b = executor.submit(psnr_sk, ref_img_lab[2], img_lab[2], data_range=200)
                    psnr_a = psnr_a.result()
                    psnr_b = psnr_b.result()
                    psnr = np.mean([psnr_a, psnr_b])
                    result["PSNR"] = psnr
                    
                if self.msssim:
                    msssim_qa_a = executor.submit(ms_ssim, ref_img_lab[1], img_lab[1], max_val=200)
                    msssim_qa_b = executor.submit(ms_ssim, ref_img_lab[2], img_lab[2], max_val=200)
                    msssim_qa_a = msssim_qa_a.result()
                    msssim_qa_b = msssim_qa_b.result()
                    msssim_fast_qa = np.mean([msssim_qa_a, msssim_qa_b])
                    result["MS-SSIM"] = msssim_fast_qa

                if self.ssim:
                    ssim_a = executor.submit(ssim, ref_img_lab[1], img_lab[1], max_val=200)
                    ssim_b = executor.submit(ssim, ref_img_lab[2], img_lab[2], max_val=200)
                    ssim_a = ssim_a.result()
                    ssim_b = ssim_b.result()
                    ssim_fast_qa = np.mean([ssim_a, ssim_b])
                    result["SSIM"] = ssim_fast_qa

                if self.vif:
                    vif_spatial_a = executor.submit(vif_spatial, ref_img_lab[1], img_lab[1], max_val=200)
                    vif_spatial_b = executor.submit(vif_spatial, ref_img_lab[2], img_lab[2], max_val=200)
                    vif_spatial_a = vif_spatial_a.result()
                    vif_spatial_b = vif_spatial_b.result()
                    vif = np.mean([vif_spatial_a, vif_spatial_b])
                    result["VIF-SPATIAL"] = vif


                


            result["File"] = rec_path
            

        
        return result

    def run_multiprocessing(self, func, args_tuple, n_processors=None):
        if not n_processors:
            n_processors = int( self.cpus // 3 )
        with Pool(processes=n_processors) as pool:
            return pool.starmap(func, args_tuple)

    def write_quality(self, qualities, ref_names, out_file):
        if not qualities:
            print("No images in current directory. ")
            return

        format_string = "%." + str(self.truncate) + "f"
        with open(out_file, "w") as f:
            f.write("* Image Quality of ")
            if self.ab:
                f.write("Color Channels\n")
            else:
                f.write("Color + Luminance\n")


        # To iterate over number of quality metrics later
        qual_count = None
        
        written_tbl_head = False
        for ref_name in ref_names:
            if not self.no_header_name:
                written_tbl_head = False
            for qual in qualities:
                # qual: dict
                qual_names = list(qual.keys())
                qual_names.remove("File")
                if qual_count is None:
                    qual_count = qual_names
                if not ref_name in qual["File"]:
                    continue
                if not written_tbl_head:
                    with open(out_file, "a") as f:
                        written_tbl_head = True
                        f.write("\n")
                        if not self.no_header_name:
                            f.write("** " + ref_name + "\n")
                        # write table head
                        f.write("| Image Name | ")
                        for qn in qual_names:
                            f.write(qn + " | ")
                        f.write("\n")
                        # write table delimiter
                        f.write("|------------")
                        for qn in qual_names:
                            f.write(" +----------- ")
                        f.write("|\n")
                        
                with open(out_file, "a") as f:
                    f.write("| " + os.path.basename(qual["File"]) + " | " )
                    # TODO: maybe handle float nan
                    for qn in qual_names:
                        f.write(format_string%(qual[qn]) + "| ")
                    f.write("\n")

        # write last row for mean and formula, if everything is in one table
        if self.no_header_name and qual_count:
            mean_formula = "@>${col}=vmean(@{col}..@>>)"
            with open(out_file, "a") as f:
                # write table delimiter
                f.write("|------------")
                for qn in qual_count:
                    f.write(" +----------- ")
                f.write("|\n")
                # Write row for Mean
                f.write("| Mean | ")
                for qn in qual_count:
                    f.write(" | ")
                f.write("\n")
                # write formula
                f.write("#+TBLFM: ")
                first_column = 2
                for i, qn in enumerate(qual_count):
                    f.write(mean_formula.format(col=first_column+i))
                    f.write("::")

        if self.format_org:
            os.system("emacs --batch " + out_file + 
            """ --eval="(require 'org)" \
            --eval="(org-table-recalculate-buffer-tables)" \
            --eval="(save-buffer)"
            """)
            if os.path.exists(out_file + "~"):
                os.remove(out_file + "~")
        print("Wrote: ", out_file)
    
    def find_files(self, search_string, path, recursive=False):
        """
        Returns an array with all full file paths to files containing 'search_string' in 'path'.
        """
        result = []
        for root, dirs, files in os.walk(path):
            for fil in files:
                if search_string in fil and fil not in result:
                    result.append(os.path.abspath(os.path.join(root, fil)))
            # First dir is path itself, so break if no subdirs should be searched
            if not recursive:
                break

        return result

    def get_ref_paths_names(self):
        """Returns an array with all reference images and a second array with all of their filenames (wo extension)"""
        # iterate through reference folder to get all ref image paths
        ref_paths = []
        for subdir, dirs, files in os.walk(self.ref_path):
            for fil in files:
                path = os.path.abspath(os.path.join(subdir, fil))
                try:
                    Image.open(path) # Just to test if file is image
                    ref_paths.append(path)
                except IOError as err:
                    # print("Warning: Found non image file, skipping: " + path)
                    pass

        # get all file names without extension, to search for their recolored versions
        ref_names = []
        for idx, p in enumerate(ref_paths):
            filename_wo_ext, extension = os.path.splitext(os.path.basename(p))
            ref_names.append(filename_wo_ext)
        return (ref_paths, ref_names)


if __name__ == "__main__":
    iq = ImageQuality()
    iq.main()
