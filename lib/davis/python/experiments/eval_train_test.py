#!/usr/bin/env python

# ----------------------------------------------------------------------------
# S Benchmark Dataset and Evaluation Methodology for Video Object Segmentation
#-----------------------------------------------------------------------------
# Copyright (c) 2016 Federico Perazzi
# Licensed under the BSD License [see LICENSE for details]
# Written by Federico Perazzi
# ----------------------------------------------------------------------------

"""
Peform evaluation on separately on training and test set.

EXAMPLE:
	python tools/eval_all.py

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
import matplotlib.pylab as plt

def parse_args():
	parser = argparse.ArgumentParser(
			description='Split evaluation between training and test set.')

	parser.add_argument('--measure', dest='measure',default='J',
			help='Measure selected for evaluation.')

	parser.add_argument('--statistic', dest='statistic',default='M',
			help='Measure statistics: [M]ean,[R]ecall,[D]ecay.')

	args = parser.parse_args()

	return args

if __name__ == '__main__':

	# Parse command-line arguments
	args = parse_args()

	db_info = db_read_info()
	db_techniques = db_read_techniques()

	attributes = db_info.attributes
	distr      = []
	S          = []


	for t_set in db_info.sets:
		log.info("Filtering techniques in: %s"%(t_set))
		# Filter sequences tagged with set=`t_set`
		X = []
		db_sequences = filter(
				lambda s: t_set == s.set ,db_info.sequences)
		for s in db_sequences:
			X.append([1 if attr in s.attributes else 0for attr in attributes ])

		distr.append(np.round(np.sum(X,axis=0).astype(np.float32)/np.sum(X),3))

		db_eval_dict = db_read_eval(sequence=[s.name for s in db_sequences],
				measure=args.measure,raw_eval=False)

		statistics_to_id = {'M':0,'O':1,'D':2}

		R = []
		for t in db_techniques:
			R.append(np.vstack(db_eval_dict[t.name][
				args.measure].values())[:,statistics_to_id[args.statistic]])

		S.append(np.average(np.array(R).T,axis=0))

	print "\nAttributes Distribution"

	table = ptable(["Set"] + attributes)
	for attr,row in zip(db_info.sets,distr):
		table.add_row([attr] + \
				['{: .2f}'.format(np.round(r,2)) for r in row])
	print table

	table = ptable(["Set"] +
			[t.name for t in db_techniques])

	print "\nEvaluation (%s)"%args.measure
	for attr,row in zip(db_info.sets,S):
		table.add_row([attr] + \
				['{: .2f}'.format(np.round(r,2)) for r in row])

	print table
