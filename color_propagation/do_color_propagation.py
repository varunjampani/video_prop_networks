#!/usr/bin/env python

'''
    File name: do_color_propagation.py
    Author: Varun Jampani
'''

# ---------------------------------------------------------------------------
# Video Propagation Networks
#----------------------------------------------------------------------------
# Copyright 2017 Max Planck Society
# Distributed under the BSD-3 Software license [see LICENSE.txt for details]
# ---------------------------------------------------------------------------

import numpy as np
import scipy.io as sio
import os
from scipy import misc
import random
import copy
import gc
gc.enable()

from utils import *
from init_caffe import *
from davis_data import *
from fetch_and_transform_data import fetch_and_transform_data
from create_online_net import *

import matplotlib.pyplot as plt

max_input_points = MAX_INPUT_POINTS
total_frames = NUM_PREV_FRAMES + 1

def color_propagation(stage_id):

    stage_id = int(stage_id)

    out_folder = RESULT_FOLDER + '/STAGE' + str(stage_id) + '_RESULT/'

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    feature_scales = [0.2, 0.04, 0.04, 0.04]

    # Iterate over all sequences
    with open(MAIN_VAL_SEQ,'r') as f:
        for seq in f:
            print(seq)
            seq = seq[:-1]
            [inputs, num_frames] = fetch_and_transform_data(seq)

            if stage_id > 0:
                prev_color_file = RESULT_FOLDER + '/STAGE' + str(stage_id-1) + '_RESULT/' + seq + '/all_frame_color_result.npy'
                prev_color_result = np.load(prev_color_file)

            result_folder = out_folder + '/' + seq + '/'
            if not os.path.exists(result_folder):
                os.makedirs(result_folder)

            color_result = (np.transpose(np.squeeze(inputs['input_color']), (1, 2, 0)) + 0.5) * 255.0
            gray_result = np.squeeze(inputs['out_features'][:, 0, :, 0:854])[:,:,None]
            full_result = np.append(gray_result, color_result, axis = 2)
            rgb_result = convert_to_rgb(full_result)
            misc.imsave(result_folder + '/' + str(0).zfill(5) + '.png',
                        rgb_result)

            all_frames_color_result = inputs['input_color']

            prev_frame_result = None
            net_inputs = {}
            net_inputs['input_color'] = inputs['input_color']
            net_inputs['scales'] = np.ones((1, 4, 1, 1))

            for k in range(0, 4):
                net_inputs['scales'][0, k, 0, 0] = feature_scales[k]

            f_value = total_frames - 1
            ignore_feat_value = -1000

            if stage_id == 0:
                standard_net = load_bnn_deploy_net(max_input_points)
            else:
                caffe_model = MODEL_FOLDER + 'COLOR_STAGE1.caffemodel'
                standard_net = load_bnn_cnn_deploy_net(max_input_points)
                standard_net.copy_from(caffe_model)

            for t in range(1, MAX_FRAMES):
                print(t)
                if t < f_value:
                    net_inputs['input_color'] = copy.copy(inputs['input_color'])
                    net_inputs['in_features'] = copy.copy(inputs['out_features'][:, :, :, 0: 854*t])
                    net_inputs['out_features'] = copy.copy(inputs['out_features'][:, :, :, 854 * t : 854 * (t+1)])
                else:
                    net_inputs['input_color'] = copy.copy(inputs['input_color'][:, :, :, 854*(t-f_value): 854*t])
                    net_inputs['in_features'] = copy.copy(inputs['out_features'][:, :, :, 854*(t-f_value): 854*t])
                    net_inputs['out_features'] = copy.copy(inputs['out_features'][:, :, :, 854 * t : 854 * (t+1)])

                height = net_inputs['in_features'].shape[2]
                width = net_inputs['in_features'].shape[3]
                num_input_points = height * width

                # Random sampling input points
                if num_input_points > max_input_points:
                    sampled_indices = random.sample(xrange(num_input_points), max_input_points)
                else:
                    sampled_indices = random.sample(xrange(num_input_points), num_input_points)

                h_indices = (np.array(sampled_indices) / width).tolist()
                w_indices = (np.array(sampled_indices) % width).tolist()
                net_inputs['input_color'] = net_inputs['input_color'][:, :, h_indices, w_indices]
                net_inputs['input_color'] = net_inputs['input_color'][:, :, np.newaxis, :]
                net_inputs['in_features'] = net_inputs['in_features'][:, :, h_indices, w_indices]
                net_inputs['in_features'] = net_inputs['in_features'][:, :, np.newaxis, :]
                if num_input_points > max_input_points:
                    prev_frame_result = standard_net.forward_all(**net_inputs)['out_color_result']
                if num_input_points < max_input_points:
                    if stage_id == 0:
                        net = load_bnn_deploy_net(num_input_points)
                    else:
                        caffe_model = MODEL_FOLDER + 'COLOR_STAGE1.caffemodel'
                        net = load_bnn_cnn_deploy_net(num_input_points)
                        net.copy_from(caffe_model)
                    prev_frame_result = net.forward_all(**net_inputs)['out_color_result']

                # import pdb; pdb.set_trace()
                result = np.squeeze(prev_frame_result)
                color_result = (np.transpose(result, (1, 2, 0)) + 0.5) * 255.0
                gray_result = np.squeeze(inputs['out_features'][:, 0, :, 854 * t : 854 * (t+1)])[:,:,None]
                full_result = np.append(gray_result, color_result, axis = 2)
                rgb_result = convert_to_rgb(full_result)
                misc.imsave(result_folder + '/' + str(t).zfill(5) + '.png',
                            rgb_result)
                all_frames_color_result = np.append(all_frames_color_result, prev_frame_result,
                                                    axis=3)

                if stage_id > 0:
                    prev_frame_result = prev_color_result[:, :, :, 854 * t : 854 * (t+1)]

                inputs['input_color'] = np.append(inputs['input_color'],
                                                  prev_frame_result,
                                                  axis=3)
                gc.collect()

            # Save the all frames color result
            out_file = result_folder + '/all_frame_color_result.npy'
            np.save(out_file, all_frames_color_result)

    return result


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: ' + sys.argv[0] + ' <stage_id>')
    else:
        color_propagation(int(sys.argv[1]))
