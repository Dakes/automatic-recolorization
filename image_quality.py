#!/usr/bin/env python3

"""
Calculates the Image Quality using PSNR and MS-SSIM recursively and writes them to a plain text .org file in the same directory. 
"""

import os, sys
import argparse
from sewar import full_ref
import cv2
from PIL import Image
import concurrent.futures
from multiprocessing import Pool
from pathlib import Path
import warnings


class ImageQuality(object):
    def __init__(self, in_path="output_images", reference_path="../pictures/", out_file="image_quality.org",
                 recursive=True, skip=False, truncate=4):
        self.in_path = in_path
        self.ref_path = reference_path
        self.out_file = out_file
        self.recursive = recursive
        self.cpus = os.cpu_count()
        self.skip = skip
        self.truncate = truncate

        # Disable Complex to float casting warning
        warnings.filterwarnings('ignore')

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
            help="Path to folder with recolored images",
        )
        parser.add_argument(
            "-o", "--output_file",
            action="store",
            dest="output_file",
            type=str,
            default=self.out_file,
            help="The path to the file, where the quality results will be written to. Default: in input folder",
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
            help="Truncate output float values to n digits. Max precision: 16 digits. ",
            default=4,
        )

        args = parser.parse_args()
        self.in_path = args.input_path
        self.ref_path = args.reference_path
        self.out_file = args.output_file
        self.recursive = not args.non_recursive
        self.skip = args.skip
        self.truncate = args.truncate

        self.get_and_write_quality()

    def get_and_write_quality(self):
        """
        To be executed, when imported. 
        """
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
                qualities = qualities + self.calc_quality(ref_paths[idx], files_matching_ref)
                
            self.write_quality(qualities, ref_names, os.path.join(root, self.out_file))
            if not self.recursive:
                break

    def calc_quality(self, ref_path, recolored_paths):
        """
        :param ref_path: reference image full path
        :param recolored_paths: Array of image paths to get quality to
        :return: 2D-Array [[path, PSNR, MS-SSIM], ...]
        """
        
        if type(recolored_paths) is str:
            recolored_paths = [recolored_paths]

        ref_img = cv2.cvtColor(cv2.imread(ref_path, 1), cv2.COLOR_BGR2RGB)

        mp_args = []
        for i in recolored_paths:
            # skip mask plot visulizations
            if ".mask" in i:
                continue
            mp_args.append((ref_img, i))
        
        qualities = self.run_multiprocessing(self.calc_quality_image, mp_args)
        
        return qualities

    def calc_quality_image(self, ref_img, rec_path):
        """
        Calculate quality measures for single image, parallelized
        :param ref_img: already loaded reference img
        :param rec_path: path to recolored image
        """
        img = cv2.cvtColor(cv2.imread(rec_path, 1), cv2.COLOR_BGR2RGB)
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # psnr = full_ref.psnr(
            #     ref_img, img, MAX=None
            # )
            psnr = executor.submit(full_ref.psnr, ref_img, img, MAX=None)

            # msssim = full_ref.msssim(
            #     ref_img, img,
            #     # default values
            #     weights=[0.0448, 0.2856, 0.3001, 0.2363, 0.1333],
            #     ws=11, K1=0.01, K2=0.03, MAX=None
            # )
            msssim = executor.submit(full_ref.msssim,
                                        ref_img, img,
                                        # default values
                                        weights=[0.0448, 0.2856, 0.3001, 0.2363, 0.1333],
                                        ws=11, K1=0.01, K2=0.03, MAX=None)

            psnr = psnr.result()
            msssim = msssim.result()
        return [rec_path, psnr, float(msssim)]

    def run_multiprocessing(self, func, args_tuple, n_processors=None):
        if not n_processors:
            n_processors = int( self.cpus // 2 )
        with Pool(processes=n_processors) as pool:
            return pool.starmap(func, args_tuple)

    def write_quality(self, qualities, ref_names, out_file):
        format_string = "%." + str(self.truncate) + "f"
        with open(out_file, "w") as f:
            f.write("* Image Quality in PSNR & MS-SSIM\n")

        for ref_name in ref_names:
            written_tbl_head = False
            for qual in qualities:
                if not ref_name in qual[0]:
                    continue
                if not written_tbl_head:
                    with open(out_file, "a") as f:
                        written_tbl_head = True
                        f.write("\n")
                        f.write("** " + ref_name + "\n")
                        f.write("| Image Name | PSNR | MS-SSIM |\n")
                        f.write("|------------+------+---------|\n")
                with open(out_file, "a") as f:
                    f.write("|"+ os.path.basename(qual[0]) +"|"+ format_string%(qual[1]) +"|" + format_string%(qual[2]) + "|\n")
        print("Wrote: ", out_file)
                # TODO: optionally format org file, using emacs and shell
    
    def find_files(self, search_string, path, recursive=False):
        """
        Returns an array with all full file paths to files containing 'search_string' in 'path'.
        """
        result = []
        for root, dirs, files in os.walk(path):
            for fil in files:
                if search_string in fil:
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
