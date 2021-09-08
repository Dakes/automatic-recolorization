# Automatic Recolorization
This is a prototype for recoloring images, as closely as possible to their original color, while only keeping a few color cues. This was built using a colorization system called [Interactive Deep Colorization](https://github.com/junyanz/interactive-deep-colorization) 

## Image Quality Metrics
image_quality.py

## Recolor
recolor.py runs both encoding and decoding at the same time.

Parameters:  
`-i`, `--input_path`: Path to folder or file of original color image(s). Goes recursively through all subfolders.  
`-o`, `--output_path`: Output folder, where the recolored image(s) should be written to.  
`-ir`, `--intermediate_representation`: Folder, where the Intermediate Representation (gray image + color cues) should be stored.  
`-m`, `--method`: Recolorization method to use. One of: "ideepcolor-px-grid", "ideepcolor-px-selective", "ideepcolor-global", "ideepcolor-stock", "ideepcolor-px-grid-exclude", "ideepcolor-px-grid+selective". For explanation of methods look below.  

`--delete_gray`: delete gray images from IR folder.  
`-s`, `--size`: Mask size to use. Default 256.  
`-g`, `--grid_size`: Distance between points of grid.  
`-p`, `--p`: p size. Radius to fill in around points with the same color.  

`-plt`, `--plot`: Save pyplot plots of selected points in output path.  
`-q`, `--quantize`: Apply Vector Quantization to colors. (Only implemented partially, does not save space. Only visual) 

`--gpu_id`: ID of the CUDA device to use. Default off (CPU only). TODO: Test on CUDA capable GPU  

## Encoder
Encodes a color image into an Intermediate Representation (IR). A grayscale image + color cues.  

Parameters:  
`-i`, `--input_path`: Path to folder or file of original color image(s). Goes recursively through all subfolders.  
`-o`, `--output_path`: Output folder, where the Intermediate Representation (IR) will be written to.  
`-m`, `--method`: Recolorization method to use. One of: "ideepcolor-px-grid", "ideepcolor-px-selective", "ideepcolor-global", "ideepcolor-stock", "ideepcolor-px-grid-exclude", "ideepcolor-px-grid+selective". For explanation of methods look below.  

`-s`, `--size`: Mask size to use. Default 256.  
`-g`, `--grid_size`: Distance between points of grid.  
`-p`, `--p`: p size. Radius to fill in around points with the same color.  

`-plt`, `--plot`: Save pyplot plots of selected points in output path. For px methods.  
`-q`, `--quantize`: Apply Vector Quantization to colors. (Only implemented partially, does not save space. Only visual) 

## Decoder
Converts the IR back to a color image. 

Parameters:  
`-i`, `--input_path`: Path to folder or file of IRs grayscale image. It will automatically match the the name of the sidecar file, with the input grayscale image. Goes recursively through all subfolders.  
`-o`, `--output_path`: Output folder, where the recolored image(s) should be written to.  
`-m`, `--method`: Recolorization method to use. One of: "ideepcolor-px-grid", "ideepcolor-px-selective", "ideepcolor-global", "ideepcolor-stock", "ideepcolor-px-grid-exclude", "ideepcolor-px-grid+selective". For explanation of methods look below.  

`-plt`, `--plot`: Save plot of color points as input into the colorization net.  

`--gpu_id`: ID of the CUDA device to use. Default off (CPU only). TODO: Test on CUDA capable GPU  



## Methods
#### ideepcolor-stock
Interactive Deep Colorization without any color hints. Usually only colorizes vegetation, but that quite good. 

#### ideepcolor-global
Uses the global hints (Histogram, Saturation) method by Interactive Deep Colorization. Small file sizes (~50-300 Byte).  
Usually hits the overall right colortone of the image. Usually only reapplies color, if there is one object in the center of the image.  

#### ideepcolor-px-grid
Uses an evenly spaced grid of color values. 
size (s): Size of the array (mask), that stores the color values. Must be power of 2. Will be stretched over the whole resolution of the input image. Larger value -> Higher precision of recolorization (relevant for small details). Large values require much RAM on Decoder side. (for s:2048 up to 30gb, for large input images).  
grid (g): Spacing of the points in the grid  
s:256, g:1 -> 128KiB  
s:256, g:4 -> 8KiB  
s:256, g:8 -> 2KiB  

Larger spaced grids tend to leave out small details, but smaller spaced ones are quite large in size. 

#### ideepcolor-px-grid-exclude
Leaves out points, if all points in a radius of 1 are similar. -> Points bunch on edges.  
Details restored well. Large areas of similar color might suffer. But slightly smaller filesizes than grid.  
s:256, g:1 -> ~50KiB

#### ideepcolor-px-selective
Searches for important color pixels, omits large equally colored areas, but still uses a few points there.  
Many adjustable parameter in selective function (blurring factor, scaling factor, number of extra points, etc.)  
Here as well, large areas of similar color often suffer from loss of color, or artifacts.  
s:256 -> ~~1KiB

#### ideepcolor-px-grid+selective
Combines grid and selective method, to iron out the issues of both. Leads to the best colorization results. Best used with a large spaced grid >16. If selective parameter are changed (Not yet available as command line parameters), can reach lower file sizes, with still good results. 
s:256 -> ~~500B - 1.5KiB

## Setup
clone with submodules
`git clone --recurse-submodules "https://github.com/Dakes/automatic-recolorization.git"`

Download pretrained models. Either manually, by executing the scripts, from the submodules or by executing `setup.sh`. interactive-deep-colorization models are only needed for global hints method. The rest uses the pytorch implementations. 

In case the original server hosting the models goes down (which happened before) I reuploaded them here: <https://drive.google.com/file/d/1uYc04V5ubTjH1RjRv3hd1bH1q9vdQm31/view?usp=sharing>

Install fitting dependencies from requirements files, or the conda environment. The global method requires caffe, which is only available via conda.

### Conda
1. Install anaconda  
2. `conda env update --file recolorization-conda-env.yml`  
3. `conda activate automatic-recolorization`  
4. run  

### pip
- `pip install --user -r requirements-recolorization.txt`  
- `pip install --user -r requirements-image_quality.txt`  

## TODO

- Test `--gpu_id` on CUDA gpu
- Implement watch functionality, that will automatically convert any new image added to a folder
- Implement space saving benefits of Vector Quantization
- save global hints more compact, using offsets and uchar for coordinate. 
- Apply compression to IR and potentially package image and sidecar file into one archive
- Automatically download && cache pretrained nets. But only the needed ones. 
