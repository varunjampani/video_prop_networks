#!/usr/bin/env python

'''
    File name: prepare_feature_data.py
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
from skimage import color
import os
from davis_data import *
from utils import *
import gc
gc.enable

seq_list_file = SEQ_LIST_FILE
image_folder = IMAGE_FOLDER
feature_folder = FEATURE_FOLDER

def extract_features(I,t):

    xvalues, yvalues = np.meshgrid(np.arange(I.shape[1]), np.arange(I.shape[0]))
    tvalues = np.ones((I.shape[0], I.shape[1])) * t

    feat = np.append(I, xvalues[:, :, None], axis=2)
    feat = np.append(feat, yvalues[:, :, None], axis=2)
    feat = np.append(feat, tvalues[:, :, None], axis=2)

    return feat

all_seqs_features = {}

# Iterate over all sequences
with open(seq_list_file,'r') as f:
    for seq in f:
        seq = seq[:-1]
        seq_dir = image_folder + seq + '/';

        out_dir = feature_folder + seq + '/'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # Iterate over all frames in each sequence
        frame_no = 0
        all_frame_features = None
        print(seq)
        while(True):
            print(frame_no)
            img_file = seq_dir + str(frame_no).zfill(5) + '.jpg'
            if os.path.isfile(img_file):
                print img_file
                img = Image.open(img_file)

                # # Convert image into YCbCr space
                ycbcr = img.convert('YCbCr')
                I = np.ndarray((img.size[1], img.size[0], 3),
                               'u1', ycbcr.tobytes())

                # Extract features and save
                features = extract_features(I, frame_no)
                out_file = out_dir + str(frame_no).zfill(5) + '.npy'
                if all_frame_features is None:
                    all_frame_features = features
                else:
                    all_frame_features = np.append(all_frame_features, features, axis=1)
                np.save(out_file, features)
                frame_no += 1
            else:
                break

            if frame_no > 24:
                break

        out_file = feature_folder + seq + '/all_frame_features.npy'
        np.save(out_file, all_frame_features)

        all_seqs_features[seq] = all_frame_features
        gc.collect()

np.save(feature_folder + '/all_seqs_features.npy', all_seqs_features)
