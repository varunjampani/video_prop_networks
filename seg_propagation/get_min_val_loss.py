#!/usr/bin/env python

'''
    File name: get_min_val_loss.py
    Author: Varun Jampani
'''

# ---------------------------------------------------------------------------
# Video Propagation Networks
#----------------------------------------------------------------------------
# Copyright 2017 Max Planck Society
# Distributed under the BSD-3 Software license [see LICENSE.txt for details]
# ---------------------------------------------------------------------------

import sys
import numpy as np

sys.path.append('../lib/caffe/tools/extra/')

from parse_log import parse_log


def get_min_val_loss(log_file):

    [train_dict, val_dict] = parse_log(log_file)

    # print([train_dict, val_dict])
    min_loss = 100000
    iterations = 0
    for t in range(len(val_dict)):
        loss_value = val_dict[t]['loss']
        if min_loss > loss_value:
            min_loss = loss_value
            iterations = int(val_dict[t]['NumIters'])

    return [min_loss, iterations]



if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: ' + sys.argv[0] + ' <log_file> ')
    else:
        [min_loss, iterations] = get_min_val_loss(sys.argv[1])
        print([min_loss, iterations])
