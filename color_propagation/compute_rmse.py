#!/usr/bin/env python

'''
    File name: compute_rmse.py
    Author: Varun Jampani
'''

# ---------------------------------------------------------------------------
# Video Propagation Networks
#----------------------------------------------------------------------------
# Copyright 2017 Max Planck Society
# Distributed under the BSD-3 Software license [see LICENSE.txt for details]
# ---------------------------------------------------------------------------

import os
import time
import argparse
from utils import *
from davis_data import *
from PIL import Image

import numpy   as np
import os.path as osp

train_seq = []
with open(MAIN_TRAIN_SEQ) as f:
    for seq in f:
        train_seq.append(seq[:-1])


val_seq = []
with open(MAIN_VAL_SEQ) as f:
    for seq in f:
        val_seq.append(seq[:-1])

all_seq = []
with open(SEQ_LIST_FILE) as f:
    for seq in f:
        all_seq.append(seq[:-1])

def parse_args():
	"""Parse input arguments."""

	parser = argparse.ArgumentParser(
			description='Compute RMSE value for colorization result.')

	parser.add_argument(dest='input',default=None,type=str,
			help='Path to the result folder.')

	parser.add_argument(
			'--datatype',default='ALL',type=str,choices=['ALL','TRAIN','VAL'])

	# Parse command-line arguments
	args       = parser.parse_args()
	args.input = osp.abspath(args.input)

	return args

def get_average_rmse(result_folder, datatype='ALL'):

    seqs = None
    if datatype == 'ALL':
        seqs = all_seq
    elif datatype == 'TRAIN':
	seqs = train_seq
    elif datatype == 'VAL':
	seqs = val_seq

    seq_rmse_values = np.zeros(len(seqs))
    seq_psnr_values = np.zeros(len(seqs))
    seq_count = 0
    for seq in seqs:
        print(seq)
        seq_img_dir = IMAGE_FOLDER + '/' + seq + '/'
        seq_result_dir = result_folder + '/' +  seq + '/'
        img_count = 0
        rmse_values = []
        psnr_values = []

        for img_count in range(0, MAX_FRAMES):
            img_file = seq_img_dir + str(img_count).zfill(5) + '.jpg'
            if os.path.isfile(img_file):
                result_file = seq_result_dir + str(img_count).zfill(5) + '.png'
                img = np.array(Image.open(img_file))
                result = np.array(Image.open(result_file))

                [rmse, psnr] = computeRMSE(result, img)
                rmse_values.append(rmse)
                psnr_values.append(psnr)
            else:
                sys.exit('Frame does not exit')

        seq_psnr_values[seq_count] = np.mean(psnr_values)
        seq_rmse_values[seq_count] = np.mean(rmse_values)

        seq_count = seq_count + 1

    return [np.mean(seq_rmse_values), np.mean(seq_psnr_values)]


if __name__ == '__main__':

    args = parse_args()

    [rmse, psnr] = get_average_rmse(args.input, args.datatype)

    print(psnr)
    print(rmse)

    out_file = args.input + '/result.txt'
    f = open(out_file, 'w')
    f.write(str(psnr) + '\n')
    f.write(str(rmse) + '\n')
    f.close()
