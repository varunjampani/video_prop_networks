#!/usr/bin/env python

'''
    File name: train_online_seg.py
    Author: Varun Jampani
'''

# ---------------------------------------------------------------------------
# Video Propagation Networks
#----------------------------------------------------------------------------
# Copyright 2017 Max Planck Society
# Distributed under the BSD-3 Software license [see LICENSE.txt for details]
# ---------------------------------------------------------------------------

import numpy as np
import sys

from init_caffe import *
from create_solver import *
from create_online_net import *
from davis_data import *

num_frames = NUM_PREV_FRAMES

def Train_segmentation(fold_id, stage_id):

    lr = float(0.001)
    prefix = INTER_MODEL_DIR + 'FOLD' + fold_id + '_' + 'STAGE' + stage_id
    test_iter = 50
    iter_size = 4
    test_interval = 200
    num_iter = 15000
    snapshot_iter = 200
    debug_info = False
    train_net_file = get_bnn_cnn_train_net_fold_stage(num_frames, fold_id, stage_id, phase = 'TRAIN')
    test_net_file = get_bnn_cnn_train_net_fold_stage(num_frames, fold_id, stage_id, phase = 'TEST')

    solver_proto = create_solver_proto(train_net_file,
                                       test_net_file,
                                       lr,
                                       prefix,
                                       test_iter = test_iter,
                                       test_interval = test_interval,
                                       max_iter=num_iter,
                                       iter_size=iter_size,
                                       snapshot=snapshot_iter,
                                       debug_info=debug_info)
    solver = create_solver(solver_proto)

    if int(stage_id) > 1:
        init_model = SELECTED_MODEL_DIR + 'FOLD' + fold_id + '_' + 'STAGE' +\
            str(int(stage_id) - 1) + '.caffemodel'
    else:
        init_model = SELECTED_MODEL_DIR + 'deeplab_vpn_init_model.caffemodel'

    solver.net.copy_from(init_model)

    solver.solve()


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: ' + sys.argv[0] + ' <fold_id> <stage_id>')
    else:
        Train_segmentation(str(sys.argv[1]), str(sys.argv[2]))
