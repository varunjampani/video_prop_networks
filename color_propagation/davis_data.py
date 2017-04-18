#!/usr/bin/env python

'''
    File name: davis_data.py
    Author: Varun Jampani
'''

# ---------------------------------------------------------------------------
# Video Propagation Networks
#----------------------------------------------------------------------------
# Copyright 2017 Max Planck Society
# Distributed under the BSD-3 Software license [see LICENSE.txt for details]
# ---------------------------------------------------------------------------

MAX_INPUT_POINTS = 300000
NUM_PREV_FRAMES = 3
MAX_FRAMES = 25
RAND_SEED = 2345

SEQ_LIST_FILE = '../data/fold_list/all_seqs.txt'
IMAGE_FOLDER = '../data/DAVIS/JPEGImages/480p/'
GT_FOLDER = '../data/DAVIS/Annotations/480p/'
IMAGESET_FOLDER = '../data/fold_list/'
FEATURE_FOLDER = '../data/color_feature_folder/'

MAIN_TRAIN_SEQ = '../data/fold_list/main_train.txt'
MAIN_VAL_SEQ = '../data/fold_list/main_val.txt'

RESULT_FOLDER = '../data/color_results/'
MODEL_FOLDER = '../data/color_models/'
