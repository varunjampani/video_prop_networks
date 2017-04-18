#!/usr/bin/env python

'''
    File name: init_caffe.py
    Author: Varun Jampani
'''

# ---------------------------------------------------------------------------
# Video Propagation Networks
#----------------------------------------------------------------------------
# Copyright 2017 Max Planck Society
# Distributed under the BSD-3 Software license [see LICENSE.txt for details]
# ---------------------------------------------------------------------------

import sys
caffe_root = '../lib/caffe/'
sys.path.insert(0, caffe_root + 'python')
print(caffe_root + 'python')
import caffe
caffe.set_mode_gpu()
