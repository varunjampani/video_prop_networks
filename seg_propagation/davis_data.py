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

NUM_FOLDS = 5
NUM_STAGES = 1
NUM_PREV_FRAMES = 9
RAND_SEED = 2345
MAX_SPIXELS = 12000

SEQ_LIST_FILE = '../data/fold_list/all_seqs.txt'
IMAGE_FOLDER = '../data/DAVIS/JPEGImages/480p/'
GT_FOLDER = '../data/DAVIS/Annotations/480p/'
IMAGESET_FOLDER = '../data/fold_list/'

SUPERPIXEL_FOLDER = '../data/gslic_spixels/'
SPIXELFEATURE_FOLDER = '../data/'
SPIXEL_GT_FOLDER = '../data/'

TRAIN_DATA_DIR = '../data/training_data/'
SELECTED_MODEL_DIR = '../data/seg_models/'
INTER_MODEL_DIR = '../data/training_data/inter_caffemodels/'
STAGE_UNARY_DIR = '../data/training_data/stage_unaries/'
STAGE_RESULT_DIR = '../data/seg_results/'
SPIXEL_UNARY_FOLDER = '../data/training_data/bnn_unaries/'
TRAIN_LOG_DIR = '../data/training_data/training_logs/'

TRAIN_LIST = {}
TRAIN_LIST['FOLD0_STAGE1'] = '../data/fold_list/train_fold0_stage1.txt'
TRAIN_LIST['FOLD0_STAGE2'] = '../data/fold_list/train_fold0_stage2.txt'
TRAIN_LIST['FOLD1_STAGE1'] = '../data/fold_list/train_fold1_stage1.txt'
TRAIN_LIST['FOLD1_STAGE2'] = '../data/fold_list/train_fold1_stage2.txt'
TRAIN_LIST['FOLD2_STAGE1'] = '../data/fold_list/train_fold2_stage1.txt'
TRAIN_LIST['FOLD2_STAGE2'] = '../data/fold_list/train_fold2_stage2.txt'
TRAIN_LIST['FOLD3_STAGE1'] = '../data/fold_list/train_fold3_stage1.txt'
TRAIN_LIST['FOLD3_STAGE2'] = '../data/fold_list/train_fold3_stage2.txt'
TRAIN_LIST['FOLD4_STAGE1'] = '../data/fold_list/train_fold4_stage1.txt'
TRAIN_LIST['FOLD4_STAGE2'] = '../data/fold_list/train_fold4_stage2.txt'

VAL_LIST = {}
VAL_LIST['FOLD0_STAGE1'] = '../data/fold_list/val_fold0_stage1.txt'
VAL_LIST['FOLD0_STAGE2'] = '../data/fold_list/val_fold0_stage2.txt'
VAL_LIST['FOLD1_STAGE1'] = '../data/fold_list/val_fold1_stage1.txt'
VAL_LIST['FOLD1_STAGE2'] = '../data/fold_list/val_fold1_stage2.txt'
VAL_LIST['FOLD2_STAGE1'] = '../data/fold_list/val_fold2_stage1.txt'
VAL_LIST['FOLD2_STAGE2'] = '../data/fold_list/val_fold2_stage2.txt'
VAL_LIST['FOLD3_STAGE1'] = '../data/fold_list/val_fold3_stage1.txt'
VAL_LIST['FOLD3_STAGE2'] = '../data/fold_list/val_fold3_stage2.txt'
VAL_LIST['FOLD4_STAGE1'] = '../data/fold_list/val_fold4_stage1.txt'
VAL_LIST['FOLD4_STAGE2'] = '../data/fold_list/val_fold4_stage2.txt'

TEST_LIST = {}
TEST_LIST['FOLD0'] = '../data/fold_list/test_fold0.txt'
TEST_LIST['FOLD1'] = '../data/fold_list/test_fold1.txt'
TEST_LIST['FOLD2'] = '../data/fold_list/test_fold2.txt'
TEST_LIST['FOLD3'] = '../data/fold_list/test_fold3.txt'
TEST_LIST['FOLD4'] = '../data/fold_list/test_fold4.txt'
