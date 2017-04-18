#!/usr/bin/env python

'''
    File name: run_segmentation.py
    Author: Varun Jampani
'''

# ---------------------------------------------------------------------------
# Video Propagation Networks
#----------------------------------------------------------------------------
# Copyright 2017 Max Planck Society
# Distributed under the BSD-3 Software license [see LICENSE.txt for details]
# ---------------------------------------------------------------------------

import numpy as np
import itertools
import scipy.io as sio
import os
from scipy import misc
from shutil import copyfile
import copy
import gc

from init_caffe import *
from davis_data import *
from fetch_and_transform_data import fetch_and_transform_data, transform_and_get_image
from create_online_net import load_bnn_cnn_deploy_net_fold_stage

import time

def set_bnn_identity_params(params):

    params['out_seg1'][0].data[...] = \
    np.random.normal(params['out_seg1'][0].data, 0.01)
    params['out_seg2'][0].data[...] = \
    np.random.normal(params['out_seg2'][0].data, 0.01)
    params['out_seg1'][0].data[0,0,0,0] = 1.0
    params['out_seg1'][0].data[0,1,0,0] = 0.0
    params['out_seg1'][0].data[1,0,0,0] = 0.0
    params['out_seg1'][0].data[1,1,0,0] = 1.0

    params['out_seg2'][0].data[0,0,0,0] = 1.0
    params['out_seg2'][0].data[0,1,0,0] = 0.0
    params['out_seg2'][0].data[1,0,0,0] = 0.0
    params['out_seg2'][0].data[1,1,0,0] = 1.0

    params['spixel_out_seg1'][0].data[...] = 0.0
    params['spixel_out_seg1'][0].data[0,0,0,0] = 0.5
    params['spixel_out_seg1'][0].data[1,1,0,0] = 0.5
    params['spixel_out_seg1'][0].data[0,32,0,0] = 0.5
    params['spixel_out_seg1'][0].data[1,33,0,0] = 0.5

    return


def run_segmentation(stage_id, fold_id = -1):

    gc.enable()

    if stage_id > 0:
        caffe_model = SELECTED_MODEL_DIR + 'SEG_FOLD' + str(fold_id) + '_' + 'STAGE' + str(stage_id) +\
            '.caffemodel'

        if stage_id == 1:
            prev_unary_folder = STAGE_UNARY_DIR + 'STAGE0_UNARY/'
        elif stage_id > 1:
            prev_unary_folder = STAGE_UNARY_DIR + 'FOLD' + str(fold_id) + '_STAGE' +\
                str(stage_id - 1) + '_UNARY/'

        all_seqs_prev_unary = np.load(prev_unary_folder + 'all_seqs_unary.npy').item()
        main_unary_folder = STAGE_UNARY_DIR + 'FOLD' + str(fold_id) + '_STAGE' +\
            str(stage_id) + '_UNARY/'
    else:
        main_unary_folder = STAGE_UNARY_DIR + 'STAGE0_UNARY/'

    out_folder = STAGE_RESULT_DIR + 'STAGE' + str(stage_id) + '_RESULT/'

    if fold_id == -1:
        seq_list_f = SEQ_LIST_FILE
    else:
        fold_str = 'FOLD' + str(fold_id)
        seq_list_f = TEST_LIST[fold_str]

    total_frames = NUM_PREV_FRAMES + 1

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    if not os.path.exists(main_unary_folder):
        os.makedirs(main_unary_folder)

    feature_scales = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    scales_1 = [0.07, 0.4, 0.4, 0.02, 0.02, 0.01]
    scales_2 = [0.09, 0.5, 0.5, 0.03, 0.03, 0.2]

    u_scales = np.zeros((1, 1, total_frames - 1, 1))
    k = 1.0
    for s in range(u_scales.shape[2]):
        u_scales[0, 0, u_scales.shape[2] - s - 1, 0] = k
        k = k * 0.5

    all_seqs_unary = {}

    # Iterate over all sequences
    with open(seq_list_f,'r') as f:
        for seq in f:
            print(seq)
            seq = seq[:-1]
            [inputs, num_frames] = fetch_and_transform_data(seq, 0,
                                                            ['unary','in_features','out_features','spixel_indices'],
                                                            feature_scales)

            result_folder = out_folder + '/' + seq + '/'
            if not os.path.exists(result_folder):
                os.makedirs(result_folder)

            first_frame_gt_file = GT_FOLDER + seq + '/' + str(0).zfill(5) + '.png'
            copyfile(first_frame_gt_file, result_folder + '/' + str(0).zfill(5) + '.png')

            first_frame_spixel_unary = inputs['unary']
            all_frame_unary = first_frame_spixel_unary

            prev_frame_unary = None
            net_inputs = {}
            net_inputs['unary'] = inputs['unary']
            net_inputs['scales1'] = np.ones((1, 6, 1, 1))
            net_inputs['scales2'] = np.ones((1, 6, 1, 1))

            for k in range(0, 6):
                net_inputs['scales1'][0, k, 0, 0] = scales_1[k]
                net_inputs['scales2'][0, k, 0, 0] = scales_2[k]

            f_value = total_frames - 1
            ignore_feat_value = -1000

            standard_net = load_bnn_cnn_deploy_net_fold_stage(f_value, stage_id)
            if stage_id > 0:
                standard_net.copy_from(caffe_model)
            else:
                set_bnn_identity_params(standard_net.params)
            print('Model Loaded')

            for t in range(1, num_frames):
                print(t)
                im_file = IMAGE_FOLDER + seq + '/' + str(t).zfill(5) + '.jpg'
                [pad_im, im] = transform_and_get_image(im_file, [481, 857])
                if stage_id > 0:
                    net_inputs['padimg'] = pad_im
                    net_inputs['img'] = im
                if t < f_value:
                    net_inputs['unary'] = copy.copy(inputs['unary'])
                    net_inputs['in_features'] = copy.copy(inputs['out_features'][:, :, 0:t, :])
                    net_inputs['out_features'] = copy.copy(inputs['out_features'][:, :, [t], :])
                    net_inputs['spixel_indices'] = copy.copy(inputs['spixel_indices'][:, [t], :, :])
                    net_inputs['unary_scales'] = copy.copy(u_scales[:, :, -t:, :])
                    net = load_bnn_cnn_deploy_net_fold_stage(t, stage_id)
                    if stage_id > 0:
                        net.copy_from(caffe_model)
                    else:
                        set_bnn_identity_params(net.params)
                    net.forward_all(**net_inputs)
                    prev_frame_unary = net.blobs['out_seg'].data
                    prev_frame_unary_spixels = net.blobs['spixel_out_seg_final'].data
                else:
                    net_inputs['unary'] = copy.copy(inputs['unary'][:, :, t-f_value: t, :])
                    net_inputs['in_features'] = copy.copy(inputs['out_features'][:, :, t-f_value:t, :])
                    net_inputs['out_features'] = copy.copy(inputs['out_features'][:, :, [t], :])
                    net_inputs['spixel_indices'] = copy.copy(inputs['spixel_indices'][:, [t], :, :])
                    net_inputs['unary_scales'] = copy.copy(u_scales[:, :, :, :])
                    t1 = time.time()
                    standard_net.forward_all(**net_inputs)
                    print('Time: ' + str(time.time() - t1))
                    prev_frame_unary = standard_net.blobs['out_seg'].data
                    prev_frame_unary_spixels = standard_net.blobs['spixel_out_seg_final'].data

                # Save result
                result = np.squeeze(prev_frame_unary)
                seg_result = result.argmax(axis=0)
                misc.imsave(result_folder + '/' + str(t).zfill(5) + '.png',
                            seg_result)

                # For saving unaries
                all_frame_unary = np.append(all_frame_unary, prev_frame_unary_spixels, axis=2)

                if stage_id > 0:
                    prev_frame_unary_spixels = all_seqs_prev_unary[seq][:, :, [t], :]
                inputs['unary'] = np.append(inputs['unary'],
                                            prev_frame_unary_spixels,
                                            axis=2)
                gc.collect()

            all_seqs_unary[seq] = all_frame_unary

    np.save(main_unary_folder + '/all_seqs_unary.npy', all_seqs_unary)

    return


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: ' + sys.argv[0] + ' <stage_id> <fold_id=-1>')
    elif len(sys.argv) < 3:
        run_segmentation(int(sys.argv[1]))
    else:
        run_segmentation(int(sys.argv[1]), int(sys.argv[2]))
