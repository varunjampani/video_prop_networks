# ----------------------------------------------------------------------------
# A Benchmark Dataset and Evaluation Methodology for Video Object Segmentation
#-----------------------------------------------------------------------------
# Copyright (c) 2016 Federico Perazzi
# Licensed under the BSD License [see LICENSE for details]
# Written by Federico Perazzi
# Adapted from FAST-RCNN (Ross Girshick)
# ----------------------------------------------------------------------------

""" Configuration file."""

import os
import os.path as osp

import sys
import yaml
from easydict import EasyDict as edict


__C = edict()

# Public access to configuration settings
cfg = __C

# Paths to dataset folders
__C.PATH = edict()

# Dataset Resolution  Available: 1080p,480p
__C.RESOLUTION="480p"

# Root folder of project
__C.PATH.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..','..'))

# Data folder
__C.PATH.DATA_DIR       = osp.abspath(osp.join(__C.PATH.ROOT_DIR, '../../data/DAVIS/'))

__C.PATH.RESULTS_DIR  = osp.abspath(osp.join(__C.PATH.DATA_DIR,  'Results'))

# Resulting segmentation mask folder
__C.PATH.SEGMENTATION_DIR  = osp.abspath(osp.join(__C.PATH.RESULTS_DIR,  'Segmentations', __C.RESOLUTION))

# Evaluation Folder"
__C.PATH.EVAL_DIR     = osp.abspath(osp.join(__C.PATH.RESULTS_DIR,  'Evaluation', __C.RESOLUTION))

# Path to input images
__C.PATH.SEQUENCES_DIR   = osp.join(__C.PATH.DATA_DIR,"JPEGImages",__C.RESOLUTION)

# Path to annotations
__C.PATH.ANNOTATION_DIR  = osp.join(__C.PATH.DATA_DIR,"Annotations",__C.RESOLUTION)

# Paths to files
__C.FILES = edict()

# Path to property file, holding information on evaluation sequences.
__C.FILES.DB_INFO = osp.abspath(osp.join(__C.PATH.DATA_DIR,"Annotations/db_info.yml"))

# Define the set of techniques to be loaded
__C.EVAL_SET="all" # Accepted options [paper,all]

assert __C.EVAL_SET == 'paper' or __C.EVAL_SET == 'all'

# Path to technique file, holding information about benchmark data
# __C.FILES.DB_BENCHMARK          = osp.abspath(
# 		osp.join(__C.PATH.RESULTS_DIR,"Evaluation/db_benchmark.yml"))

__C.FILES.DB_BENCHMARK          = osp.abspath(
		osp.join(__C.PATH.ROOT_DIR,"data/db_benchmark.yml"))

__C.N_JOBS = 32

# append path for cpp libraries
def _set_path_to_cpp_libs():
	sys.path.append(osp.abspath(
		osp.join(cfg.PATH.ROOT_DIR,'build/release')))
