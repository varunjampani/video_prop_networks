#!/usr/bin/env python

'''
    File name: select_model.py
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
from shutil import copyfile

from davis_data import *
from get_min_val_loss import get_min_val_loss


def select_and_copy(fold_id, stage_id):

    log_file = TRAIN_LOG_DIR + 'FOLD' + fold_id + '_' + 'STAGE' + stage_id + '.log'
    [min_loss, iterations] = get_min_val_loss(log_file)

    if iterations == 0:
        select_model = SELECTED_MODEL_DIR + 'FOLD' + fold_id + '_' + 'STAGE' + str(int(stage_id) - 1) +\
        '.caffemodel'
    else:
        select_model = INTER_MODEL_DIR + 'FOLD' + fold_id + '_' + 'STAGE' + stage_id +\
            '_iter_' + str(iterations) + '.caffemodel'

    dest_model = SELECTED_MODEL_DIR + 'SEG_FOLD' + fold_id + '_' + 'STAGE' + stage_id +\
        '.caffemodel'

    print(select_model)

    copyfile(select_model, dest_model)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: ' + sys.argv[0] + ' <fold_id> <stage_id>')
    else:
        select_and_copy(str(sys.argv[1]), str(sys.argv[2]))
