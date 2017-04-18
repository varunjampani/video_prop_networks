#!/usr/bin/env python

# ----------------------------------------------------------------------------
# A Benchmark Dataset and Evaluation Methodology for Video Object Segmentation
#-----------------------------------------------------------------------------
# Copyright (c) 2016 Federico Perazzi
# Licensed under the BSD License [see LICENSE for details]
# Written by Federico Perazzi
# ----------------------------------------------------------------------------

"""
Peform per-attribute evaluation as reported in the accompanying paper.

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

def parse_args():

	parser = argparse.ArgumentParser(
			description='Peform per-attribute evaluation.')

	parser.add_argument('--measure', dest='measure',default='J',
			help='Measure selected for evaluation.')

	parser.add_argument('--statistic', dest='statistic',default='M',
			help='Measure statistics: [M]ean,[R]ecall,[D]ecay.')

	parser.add_argument('--attributes', dest='attributes',default=['AC','DB','FM','MB','OCC'],
			nargs='+', help='Select (set of) attributes to be displayed.')

	args = parser.parse_args()

	return args

if __name__ == '__main__':

	# Parse command-line arguments
	args = parse_args()

	db_sequences  = db_read_sequences()
	db_techniques = db_read_techniques()

	A = []


	for attribute in args.attributes:
		# Filter sequences tagged with `attribute`
		log.info("Filtering sequences with attribute: %s"%attribute)
		sequences = filter(
				lambda s: attribute in s.attributes,db_sequences)

		db_eval_dict = db_read_eval(sequence=[s.name for s in sequences],
				measure=args.measure,raw_eval=False)

		statistics_to_id = {'M':0,'O':1,'D':2}

		R = []
		for t in db_techniques:
			R.append(np.vstack(db_eval_dict[t.name][
				args.measure].values())[:,statistics_to_id[args.statistic]])

		A.append(np.average(np.array(R).T,axis=0))

	table = ptable(["Attribute"] +
			[t.name for t in db_techniques])

	for attr,row in zip(args.attributes,A):
		table.add_row([attr] + \
				['{: .2f}'.format(np.round(r,2)) for r in row])

	print "\n" + str(table) + "\n"
