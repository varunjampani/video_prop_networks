
A Benchmark Dataset and Evaluation Methodology for Video Object Segmentation (DAVIS)
=====================================================================================

Package containing helper functions for loading and evaluating [DAVIS](https://graphics.ethz.ch/~perazzif/davis/index.html).

A [Matlab](https://github.com/jponttuset/davis-matlab) version of the same package is also available.

Introduction
--------------
DAVIS (Densely Annotated VIdeo Segmentation), consists of fifty high quality,
Full HD video sequences, spanning multiple occurrences of common video object
segmentation challenges such as occlusions, motion-blur and appearance
changes. Each video is accompanied by densely annotated, pixel-accurate and
per-frame ground truth segmentation.

Citation
--------------

Please cite `DAVIS` in your publications if it helps your research:

    `@inproceedings{Perazzi_CVPR_2016,
      author    = {Federico Perazzi and
                   Jordi Pont-Tuset and
                   Brian McWilliams and
                   Luc Van Gool and
                   Markus Gross and
                   Alexander Sorkine-Hornung},
      title     = {A Benchmark Dataset and Evaluation Methodology for Video Object Segmentation},
      booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
      year      = {2016}
    }`

Terms of Use
--------------
DAVIS is released under the BSD License [see LICENSE for details]

Dependencies
------------
C++

 * Boost.Python

Python

 * Cython==0.24
 * PyYAML==3.11
 * argparse==1.2.1
 * easydict==1.6
 * future==0.15.2
 * h5py==2.6.0
 * matplotlib==1.5.1
 * numpy==1.11.0
 * prettytable==0.7.2
 * scikit-image==0.12.3
 * scipy==0.17.0

Installation
--------------
C++

1. ./configure.sh && make -C build/release

Python:

1. pip install virtualenv virtualenvwrapper
2. source /usr/local/bin/virtualenvwrapper.sh
3. mkvirtualenv davis
4. pip install -r python/requirements.txt
5. export PYTHONPATH=$(pwd)/python/lib
6. See ROOT/python/lib/davis/config.py for a list of available options

Documentation
----------------
See source code for documentation.

The directory is structured as follows:

 * `ROOT/cpp`: Implementation and python wrapper of the temporal stability measure.

 * `ROOT/python/tools`: contains scripts for evaluating segmentation.
     - `eval.py` : evaluate a technique and store results in HDF5 file
     - `eval_view.py`: read and display evaluation from HDF5.

 * `ROOT/python/experiments`: contains several demonstrative examples.
 * `ROOT/python/lib/davis`  : library package contains helper functions for parsing and evaluating DAVIS

 * `ROOT/data` :
     - `get_davis.sh`: download input images and annotations.
     - `get_davis_cvpr2016_results.sh`: download the CVPR 2016 submission results.

Contacts
------------------
- [Federico Perazzi](https://graphics.ethz.ch/~perazzif)
- [Jordi Pont-Tuset](http://jponttuset.github.io)
