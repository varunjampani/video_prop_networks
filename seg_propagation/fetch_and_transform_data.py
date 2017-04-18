#!/usr/bin/env python

'''
    File name: fetch_and_transform_data.py
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
import scipy.io as sio
from PIL import Image
import cv2
from davis_data import *
from init_caffe import *

# Load all the features, gt and spixels
import time
print('Loading Data...')
t1 = time.time()
all_seqs_features = np.load(SPIXELFEATURE_FOLDER  + 'all_seqs_features.npy').item()
all_seqs_gt = np.load(SPIXEL_GT_FOLDER + 'all_seqs_spixel_gt.npy').item()
all_seqs_spixels = np.load(SUPERPIXEL_FOLDER + 'all_seqs_spixels.npy').item()
print(time.time() - t1)
print('Finished Loading Data.')

max_spixels = MAX_SPIXELS

def transform_and_get_image(im_file, out_size):

    im = caffe.io.load_image(im_file)

    out_height = out_size[0]
    out_width = out_size[1]

    height = im.shape[0]
    width = im.shape[1]

    mean_color_values = np.load('../lib/caffe/python/caffe/imagenet/\
ilsvrc_2012_mean.npy').mean(1).mean(1)

    transformer = caffe.io.Transformer({'img': (1, 3, out_size[0], out_size[1])})
    transformer.set_mean('img', mean_color_values)
    transformer.set_transpose('img', (2, 0, 1))
    transformer.set_channel_swap('img', (2, 1, 0))
    transformer.set_raw_scale('img', 255.0)

    pad_height = out_height - height
    pad_width = out_width - width
    im = np.lib.pad(im, ((0, pad_height), (0, pad_width), (0, 0)), 'constant',
                    constant_values=-5)
    for i in range(0, 3):
        temp = im[:, :, i]
        temp[temp == -5] = mean_color_values[2-i] / 255.0
        im[:, :, i] = temp
    im = np.asarray(transformer.preprocess('img', im))
    im = np.expand_dims(im, axis=0)

    return [im, im[:,:,:480,:854]]

def transform_and_get_unary(seqname):
    # First frame ground-truth
    gt = all_seqs_gt[seqname][0, :]
    unary = np.zeros((1, 2, 1, gt.shape[0]))
    unary[0,0,0,:][gt==0] = 1
    unary[0,1,0,:][gt==1] = 1
    return unary

# For YUVXYT features
def transform_and_get_features(seqname, frame_idx, feature_scales, num_frames=None):

    in_features = np.zeros((6, 1, max_spixels))
    in_features[:, :, :] = all_seqs_features[seqname][:, [frame_idx], :]

    if num_frames is None:
        num_out_frames = all_seqs_features[seqname].shape[1]
        out_features = np.zeros((6, num_out_frames, max_spixels))
        out_features[:, :, :] = all_seqs_features[seqname]
    else:
        num_out_frames = num_frames
        out_features = np.zeros((6, num_frames, max_spixels))
        out_features[:, :, :] = all_seqs_features[seqname][:, frame_idx: frame_idx + num_frames, :]

    for t in range(6):
        in_features[t, :, :] = in_features[t, :, :] * feature_scales[t]
        out_features[t, :, :] = out_features[t, :, :] *  feature_scales[t]

    in_features = in_features[None,:,:,:]
    out_features = out_features[None,:,:,:]
    return [in_features, out_features, num_out_frames]


def transform_and_get_spixels(seqname, frame_idx, num_frames):

    num_out_frames = num_frames
    if num_frames is None:
        num_out_frames = all_seqs_spixels[seqname].shape[0]
        spixels = np.zeros((num_out_frames, 480, 854))
    	spixels = all_seqs_spixels[seqname][:, :, :]
    else:
        spixels = np.zeros((num_frames, 480, 854))
    	spixels = all_seqs_spixels[seqname][frame_idx: frame_idx + num_frames, :, :]
    spixels = spixels[None, :, :, :]

    return spixels


def fetch_and_transform_data(seqname,
                             frame_idx,
                             out_types,
                             feature_scales,
                             fix_num_frames=None):

    inputs = {}
    num_out_frames = 0
    for in_name in out_types:
        if in_name == 'unary':
            inputs['unary'] = transform_and_get_unary(seqname)
        if in_name == 'spixel_indices':
            inputs['spixel_indices'] = transform_and_get_spixels(seqname, frame_idx,
                                                                 fix_num_frames)
        if in_name == 'in_features':
            [inputs['in_features'], inputs['out_features'], num_out_frames] = \
                transform_and_get_features(seqname, frame_idx,
                                            feature_scales, fix_num_frames)

    return [inputs, num_out_frames]
