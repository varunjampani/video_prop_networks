# ----------------------------------------------------------------------------
# A Benchmark Dataset and Evaluation Methodology for Video Object Segmentation
#-----------------------------------------------------------------------------
# Copyright (c) 2016 Federico Perazzi
# Licensed under the BSD License [see LICENSE for details]
# Written by Federico Perazzi
# ----------------------------------------------------------------------------
from logger import logging as log
from timer import Timer
from config import cfg,_set_path_to_cpp_libs
_set_path_to_cpp_libs()

from dataset.loader import DAVISAnnotationLoader,DAVISSegmentationLoader
