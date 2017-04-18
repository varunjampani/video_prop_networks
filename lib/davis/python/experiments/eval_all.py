#!/usr/bin/env python

# ----------------------------------------------------------------------------
# A Benchmark Dataset and Evaluation Methodology for Video Object Segmentation
#-----------------------------------------------------------------------------
# Copyright (c) 2016 Federico Perazzi
# Licensed under the BSD License [see LICENSE for details]
# Written by Federico Perazzi
# ----------------------------------------------------------------------------

"""
Peform full evaluation as reported in the paper and store results.

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
			description='Perform full evaluation as reported in the paper.')

	parser.add_argument('--compute',
			dest='compute',action='store_true',
			help='Compute results instead of loading from file.')

	# Parse command-line arguments
	return parser.parse_args()

if __name__ == '__main__':

	args = parse_args()

	if args.compute:
		log.info('Running full evaluation on DAVIS')
		log.info('Searching available techniques in: "%s"'%cfg.PATH.SEGMENTATION_DIR)

		# Search available techniques within the default output folder
		techniques = sorted([osp.splitext(osp.basename(t))[0]
				for t in glob.glob(cfg.PATH.SEGMENTATION_DIR+ "/*")])

		log.info('Number of techniques being evaluated: %d'%len(techniques))

		# Read sequences from file
		log.info('Reading sequences from: %s '%osp.basename(cfg.FILES.DB_INFO))
		sequences  = [s.name for s in db_read_sequences()]

		# Compute full evaluation and save results
		for technique in techniques:
			db_save_eval(db_eval(technique,sequences))

		# Read results from file
		db_eval_dict = db_read_eval(raw_eval=False)

		# Save techniques attributes and results
		#db_save_techniques(db_eval_dict)

	log.info('Reading available techniques and results from: %s'%
			osp.basename(cfg.FILES.DB_BENCHMARK))

	db_techniques = db_read_techniques()

	# Display results
	table = ptable(['Measure']+[t.name for t in db_techniques])

	X = np.array([np.hstack([t.J,t.F,t.T])
		for t in db_techniques]).T

	for row,measure in zip(X,['J(M)','J(O)','J(D)','F(M)','F(O)','F(D)','T(M)']):
		table.add_row([measure]+["{: .3f}".format(r) for r in row])

	print "\n" + str(table) + "\n"
