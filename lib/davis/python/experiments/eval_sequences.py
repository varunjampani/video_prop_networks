#!/usr/bin/env python

# ----------------------------------------------------------------------------
# A Benchmark Dataset and Evaluation Methodology for Video Object Segmentation
#-----------------------------------------------------------------------------
# Copyright (c) 2016 Federico Perazzi
# Licensed under the BSD License [see LICENSE for details]
# Written by Federico Perazzi
# ----------------------------------------------------------------------------

"""
Peform per-sequence evaluation of a technique and display results.

EXAMPLE:
	python tools/eval_sequence.py

"""

import sys
import h5py
import glob
import argparse
import numpy   as np
import os.path as osp

from davis import cfg,log
from davis.dataset import *

from prettytable import PrettyTable as ptable

def parse_args():
	parser = argparse.ArgumentParser(
			description='Perform full evaluation.')

	parser.add_argument('--measure',
			dest='measure',default='J',
			help='Evaluate results instead of loading from file.')

	parser.add_argument('--statistic',
			dest='statistic',default='M',
			help='Evaluate results instead of loading from file.')

	# Parse command-line arguments
	return parser.parse_args()


if __name__ == '__main__':

	args = parse_args()

	db_sequences  = db_read_sequences()
	db_techniques = db_read_techniques()

	# Read results from file
	log.info("Reading evaluation from: %s"%cfg.FILES.DB_BENCHMARK)
	db_eval_dict = db_read_eval(
			measure=args.measure,raw_eval=False)

	# Generate table
	statistics_to_id = {'M':0,'O':1,'D':2}

	R = []
	for t in db_techniques:
		R.append(np.vstack(db_eval_dict[t.name][
			args.measure].values())[:,statistics_to_id[args.statistic]])

	R = np.array(R).T

	table = ptable(["Sequence"] +
			[t.name for t in db_techniques])

	for n,row in enumerate(R):
		table.add_row([db_sequences[n].name] + \
				['{: .3f}'.format(r) for r in row])

	table.add_row(["Average"] + ['{: .3f}'.format(r)
		for r in np.average(R,axis=0)])

	print "\n" + str(table) + "\n"
