#!/usr/bin/env python3

import os
import argparse
import ar_utils

class Encoder(object):
    def __init__(self) -> None:
        self.methods = ["ideepcolor-px", "ideepcolor-hist", "HistoGAN"]
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
                               help='TODO')
        parser.add_argument('-m', '--method', action='store', dest='method', type=str, default=self.methods[0],
                            help='The colorization method to use. Possible values: \"' + ', '.join(self.methods) + '\"')
        parser.add_argument('-w','--watch', dest='watch', help='watch input folder for new images', action='store_true')

        args = parser.parse_args()
        self.watch = args.watch

        mask = ar_utils.Mask()
        mask.put_point([135,160], [100,-69])

    def load_image(self, path):
        pass

    def encode(self, img):
        pass


if __name__ == "__main__":
    ec = Encoder()
    ec.main()
