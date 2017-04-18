#!/usr/bin/env python

'''
    File name: input_data_layer.py
    Author: Varun Jampani
    Adapted from
    https://github.com/LisaAnne/lisa-caffe-public/blob/lstm_video_deploy/examples/LRCN_activity_recognition/sequence_input_layer.py
'''

# ---------------------------------------------------------------------------
# Video Propagation Networks
#----------------------------------------------------------------------------
# Copyright 2017 Max Planck Society
# Distributed under the BSD-3 Software license [see LICENSE.txt for details]
# ---------------------------------------------------------------------------

import io
import numpy as np
import random
from multiprocessing import Pool
from threading import Thread
import sys

from init_caffe import *
from davis_data import *
from fetch_and_transform_data import transform_and_get_image

from random import Random
myrandom = Random(10)

all_seqs_features = np.load(SPIXELFEATURE_FOLDER  + 'all_seqs_features.npy').item()
all_seqs_gt = np.load(GT_FOLDER + 'all_seqs_gt.npy').item()
all_seqs_spixels = np.load(SUPERPIXEL_FOLDER + 'all_seqs_spixels.npy').item()

all_seqs_prev_unary = None
train_seq = []
train_num_images = []
val_seq = []
val_num_images = []

max_spixels = MAX_SPIXELS
num_frames = NUM_PREV_FRAMES

scales_v_1 = [0.07, 0.4, 0.4, 0.02, 0.02, 0.01]
scales_v_2 = [0.09, 0.5, 0.5, 0.03, 0.03, 0.2]

scales1 = np.ones((1, 6, 1, 1))
scales2 = np.ones((1, 6, 1, 1))

for k in range(0, 6):
    scales1[0, k, 0, 0] = scales_v_1[k]
    scales2[0, k, 0, 0] = scales_v_2[k]

u_scales = np.zeros((1, 1, num_frames, 1))
k = 1.0
for s in range(u_scales.shape[2]):
    u_scales[0, 0, u_scales.shape[2] - s - 1, 0] = k
    k = k * 0.5

class DataProcessor(object):
    def __init__(self, data_type):
        self.data_type = data_type
    def __call__(self, im_info):

        seq_ids = im_info[0][0]
        frame_idx = im_info[0][1]
        if self.data_type == 'TRAIN':
            seqname = train_seq[seq_ids]
        elif self.data_type == 'VAL':
            seqname = val_seq[seq_ids]

        start_frame = np.max((0, frame_idx - num_frames))

        inputs = {}
        unary = np.zeros((1, 2, num_frames, max_spixels))
        unary[:, :, start_frame - frame_idx:, :] = all_seqs_prev_unary[seqname][:, :, start_frame : frame_idx, :]
        inputs['unary'] = unary

        in_features = np.zeros((6, num_frames, max_spixels)) - 1000.
        in_features[:, start_frame - frame_idx:, :] = all_seqs_features[seqname][:, start_frame : frame_idx, :]
        in_features = in_features[None,:,:,:]
        inputs['in_features'] = in_features

        out_features = np.zeros((6, 1, max_spixels)) - 1000.
        out_features[:, :, :] = all_seqs_features[seqname][:, [frame_idx], :]
        out_features = out_features[None,:,:,:]
        inputs['out_features'] = out_features

        inputs['scales1'] = scales1
        inputs['scales2'] = scales2
        inputs['unary_scales'] = u_scales

        spixels = np.zeros((1, 480, 854))
    	spixels[:,:,:] = all_seqs_spixels[seqname][[frame_idx], :, :]
        spixels = spixels[None, :, :, :]
        inputs['spixel_indices'] = spixels

        gt = all_seqs_gt[seqname][:, 854 * frame_idx: 854 * (frame_idx + 1)]
        inputs['label'] = gt[None, None, :, :]

        im_file = IMAGE_FOLDER + seqname + '/' + str(frame_idx).zfill(5) + '.jpg'
        [pad_im, im] = transform_and_get_image(im_file, [481, 857])
        inputs['padimg'] = pad_im
        inputs['img'] = im

        return inputs

class sequenceGenerator(object):
    def __init__(self, batch_size, data_type, reset_count):
        self.batch_size = batch_size
        self.data_type = data_type

        self.idx = 0
        self.rounds = 0
        self.reset_count = reset_count

        self.rand_generator = Random(RAND_SEED)

    def __call__(self):

        im_list = []

        if self.data_type == 'TRAIN':
            seq_ids = self.rand_generator.sample(range(0, len(train_seq)),
                                                 self.batch_size)
            frame_ids = []
            ct = 0
            for t in seq_ids:
                frame_ids.append(self.rand_generator.sample(range(1, train_num_images[t]),1))
                im_list.append([t, frame_ids[ct][0]])
                ct = ct + 1

        elif self.data_type == 'VAL':
            seq_ids = []
            for ct in range(self.batch_size):
                seq_ids.append(self.rand_generator.choice(range(0, len(val_seq))))

            frame_ids = []
            ct = 0
            for t in seq_ids:
                frame_ids.append(self.rand_generator.sample(range(1, val_num_images[t]),1))
                im_list.append([t, frame_ids[ct][0]])
                ct = ct + 1

        im_info = zip(im_list)
        self.rounds += 1
        if self.rounds >= self.reset_count:
            self.rounds = 0
            self.idx = 0
            self.rand_generator = Random(RAND_SEED)

        return im_info


def advance_batch(result, sequence_generator, data_processor, pool):
    im_info = sequence_generator()
    tmp = data_processor(im_info[0])
    result['data'] = pool.map(data_processor, im_info)

class BatchAdvancer():
    def __init__(self, result, sequence_generator, image_processor, pool):
        self.result = result
        self.image_processor = image_processor
        self.sequence_generator = sequence_generator
        self.pool = pool

    def __call__(self):
        return advance_batch(self.result,
                             self.sequence_generator,
                             self.image_processor,
                             self.pool)

class InputRead(caffe.Layer):

    def initialize(self):

        self.batch_size = 3
        self.num_tops = 10
        self.top_names = ['img', 'padimg', 'unary', 'in_features', 'out_features', 'spixel_indices',
                          'scales1', 'scales2', 'unary_scales', 'label']
        self.pool_size = 5

    def setup(self, bottom, top):

        random.seed(RAND_SEED)

        params = self.param_str.split('_')

        if len(params) < 1:
            params = ['TRAIN', '1000000', '0', '1']
            print("Using standard initialization of params:", params)

        data_type = str(params[0])
        reset_count = int(params[1])
        fold_id = params[2]
        stage_id = params[3]

        if int(stage_id) == 1:
            prev_unary_folder = STAGE_UNARY_DIR + 'STAGE0_UNARY/'
        else:
            prev_unary_folder = STAGE_UNARY_DIR + 'FOLD' + fold_id + '_STAGE' +\
                str(int(stage_id) - 1) + '_UNARY/'

        global all_seqs_prev_unary
        all_seqs_prev_unary = np.load(prev_unary_folder + 'all_seqs_unary.npy').item()

        global train_seq
        global train_num_images
        if len(train_seq) == 0:
            with open(TRAIN_LIST['FOLD' + fold_id + '_' + 'STAGE' + stage_id]) as f:
                for seq in f:
                    train_seq.append(seq[:-1])
                    train_num_images.append(all_seqs_features[seq[:-1]].shape[1])

        if len(train_seq) != 35 and len(train_seq) != 30:
            sys.exit("Number of training sequences is not 35.")

        global val_seq
        global val_num_images
        if len(val_seq) == 0:
            with open(VAL_LIST['FOLD' + fold_id + '_' + 'STAGE' + stage_id]) as f:
                for seq in f:
                    val_seq.append(seq[:-1])
                    val_num_images.append(all_seqs_features[seq[:-1]].shape[1])

        if len(val_seq) != 5 and len(val_seq) != 20:
            sys.exit("Number of validation sequences is not 5.")

        self.initialize()

        self.thread_result = {}
        self.thread = None
        pool_size = self.pool_size

        self.data_processor = DataProcessor(data_type)
        self.sequence_generator = sequenceGenerator(self.batch_size,
                                                    data_type,
                                                    reset_count)

        self.pool = Pool(processes=pool_size)
        self.batch_advancer = BatchAdvancer(self.thread_result,
                                            self.sequence_generator,
                                            self.data_processor,
                                            self.pool)

        self.dispatch_worker()
        if len(top) != len(self.top_names):
            raise Exception('Incorrect number of outputs (expected %d, got %d)' %
                            (len(self.top_names), len(top)))
        self.join_worker()

    def reshape(self, bottom, top):
        for top_index, name in enumerate(self.top_names):
            if name == 'img':
                shape = (self.batch_size, 3, 480, 854)
            elif name == 'padimg':
                shape = (self.batch_size, 3, 481, 857)
            elif name == 'unary':
                shape = (self.batch_size, 2, num_frames, max_spixels)
            elif name == 'in_features':
                shape = (self.batch_size, 6, num_frames, max_spixels)
            elif name == 'out_features':
                shape = (self.batch_size, 6, 1, max_spixels)
            elif name == 'spixel_indices':
                shape = (self.batch_size, 1, 480, 854)
            elif name == 'scales1' or name == 'scales2':
                shape = (1, 6, 1, 1)
            elif name == 'unary_scales':
                shape = (1, 1, num_frames, 1)
            elif name == 'label':
                shape = (self.batch_size, 1, 480, 854)
            top[top_index].reshape(*shape)
        pass


    def forward(self, bottom, top):

        if self.thread is not None:
            self.join_worker()

        new_result = {}

        for t, name in enumerate(self.top_names):
            new_result[self.top_names[t]] =\
                [None]*len(self.thread_result['data'][0][self.top_names[t]])

        for i in range(self.batch_size):
            for t, name in enumerate(self.top_names):
                if name == 'scales1' or name == 'scales2' or name == 'unary_scales':
                    top[t].data[...] = self.thread_result['data'][i][self.top_names[t]]
                else:
                    top[t].data[i, ...] =\
                        self.thread_result['data'][i][self.top_names[t]]

        self.dispatch_worker()


    def dispatch_worker(self):
        assert self.thread is None
        self.thread = Thread(target=self.batch_advancer)
        self.thread.start()


    def join_worker(self):
        assert self.thread is not None
        self.thread.join()
        self.thread = None


    def backward(self, top, propagate_down, bottom):
        pass
