#!/usr/bin/env python

'''
    File name: prepare_train_data.py
    Author: Varun Jampani
'''

# ---------------------------------------------------------------------------
# Video Propagation Networks
#----------------------------------------------------------------------------
# Copyright 2017 Max Planck Society
# Distributed under the BSD-3 Software license [see LICENSE.txt for details]
# ---------------------------------------------------------------------------

import numpy as np
from PIL import Image
import os
import cv2
import png
import itertools
from scipy.stats import mode
from davis_data import *
from init_caffe import *
from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2
import tempfile
import gc
gc.enable

seq_list_file = SEQ_LIST_FILE
image_folder = IMAGE_FOLDER
feature_folder = SPIXELFEATURE_FOLDER
spixel_folder = SUPERPIXEL_FOLDER
gt_folder = GT_FOLDER
spixel_gt_folder = SPIXEL_GT_FOLDER

all_seqs_features = {}
all_seqs_spixels = {}
all_seqs_gt = {}
all_seqs_spixel_gt = {}

max_spixels = MAX_SPIXELS
ignore_feat_value = -1000
ignore_gt_value = 1000

def load_spixel_feature_model():

    n = caffe.NetSpec()

    n.img_features = L.Input(shape=[dict(dim=[1, 6, 480, 854])])
    n.spixel_indices = L.Input(shape=[dict(dim=[1, 1, 480, 854])])
    n.spixel_features = L.SpixelFeature(n.img_features, n.spixel_indices,
                                        spixel_feature_param = dict(type = P.SpixelFeature.AVGRGB,
                                                                    max_spixels = max_spixels,
                                                                    rgb_scale = 1.0,
                                                                    ignore_feature_value = ignore_feat_value))

    # Save to temporary file and load
    f = tempfile.NamedTemporaryFile(mode='w+', delete=False)
    f.write(str(n.to_proto()))
    f.close()
    return caffe.Net(f.name, caffe.TEST)


def extract_image_features(I,t):

    xvalues, yvalues = np.meshgrid(np.arange(I.shape[1]), np.arange(I.shape[0]))
    tvalues = np.ones((I.shape[0], I.shape[1])) * t

    feat = np.append(I, xvalues[:, :, None], axis=2)
    feat = np.append(feat, yvalues[:, :, None], axis=2)
    feat = np.append(feat, tvalues[:, :, None], axis=2)

    return feat


def convert_to_spixel_features(spixel_feat_net, img_features, spixel_indices):

    img_features2 = np.transpose(img_features,[2,0,1])[None,:,:,:]
    spixel_indices2 = spixel_indices[None, None, :, :]

    net_inputs = {}
    net_inputs['img_features'] = img_features2
    net_inputs['spixel_indices'] = spixel_indices2
    spixel_feat_net.forward_all(**net_inputs)

    spixel_features = np.zeros((6, 1,  max_spixels)) + ignore_feat_value
    spixel_features[:, :, :] = spixel_feat_net.blobs['spixel_features'].data[0,:,:,:]

    return spixel_features


def convert_to_spixel_gt(img_gt, spixel_indices):

    gt = np.zeros((1, max_spixels)) + ignore_gt_value
    s_values = np.unique(spixel_indices)
    for i in s_values:
        indx = np.where(spixel_indices==i)
        if indx[0].shape[0]>0:
            gt[0, i], _ = mode(img_gt[indx[0],indx[1]],axis=None)

    return gt

spixel_feat_net = load_spixel_feature_model()

# Iterate over all sequences
with open(seq_list_file,'r') as f:
    for seq in f:
        seq = seq[:-1]
        seq_dir = image_folder + seq + '/';
        seq_gt_dir = gt_folder + seq + '/'

        # Iterate over all frames in each sequence
        frame_no = 0
        all_frame_features = None
        all_frame_spixels = None
        all_frame_gt = None
        spixel_gt = None
        print(seq)

        while(True):
            print(frame_no)

            # Prepare superpixels, their features and GT files
            img_file = seq_dir + str(frame_no).zfill(5) + '.jpg'
            if os.path.isfile(img_file):
                img = Image.open(img_file)

		        # Convert image into YUV space
                ycbcr = img.convert('YCbCr')
                I = np.ndarray((img.size[1], img.size[0], 3),
                               'u1', ycbcr.tobytes())

		        # Extract image features
                img_features = extract_image_features(I, frame_no)

		        # Read superpixel indices
                superpixel_file = spixel_folder + seq + '/' + str(frame_no).zfill(5) + '.pgm'
                spixel_indices = np.array(cv2.imread(superpixel_file, cv2.IMREAD_UNCHANGED))

                spixel_indx = spixel_indices[None, :, :]
                if all_frame_spixels is None:
                    all_frame_spixels = spixel_indx
                else:
                    all_frame_spixels = np.append(all_frame_spixels, spixel_indx, axis=0)

	            # Convert to superpixel features
                features = convert_to_spixel_features(spixel_feat_net, img_features, spixel_indices)

                if all_frame_features is None:
                    all_frame_features = features
                else:
                    all_frame_features = np.append(all_frame_features, features, axis=1)


                # Prepare GT file
                gt_file = seq_gt_dir + str(frame_no).zfill(5) + '.png'
                r = png.Reader(gt_file)
                width, height, data, meta = r.read()
                gt = np.vstack(itertools.imap(np.uint8, data))
                gt[gt==255] = 1
                if all_frame_gt is None:
                    all_frame_gt = gt
                else:
                    all_frame_gt = np.append(all_frame_gt, gt, axis=1)

                if frame_no == 0:
                    spixel_gt = convert_to_spixel_gt(gt, spixel_indices)

                frame_no += 1
            else:
                break

        all_seqs_gt[seq] = all_frame_gt
        all_seqs_spixel_gt[seq] = spixel_gt
        all_seqs_spixels[seq] = all_frame_spixels
        all_seqs_features[seq] = all_frame_features

np.save(spixel_folder + '/all_seqs_spixels.npy', all_seqs_spixels)
np.save(feature_folder + '/all_seqs_features.npy', all_seqs_features)
np.save(gt_folder + '/all_seqs_gt.npy', all_seqs_gt)
np.save(spixel_gt_folder + '/all_seqs_spixel_gt.npy', all_seqs_spixel_gt)
