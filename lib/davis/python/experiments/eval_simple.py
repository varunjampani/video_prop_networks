#!/usr/bin/env python

# ----------------------------------------------------------------------------
# A Benchmark Dataset and Evaluation Methodology for Video Object Segmentation
#-----------------------------------------------------------------------------
# Copyright (c) 2016 Federico Perazzi
# Licensed under the BSD License [see LICENSE for details]
# Written by Federico Perazzi
# ----------------------------------------------------------------------------

"""
Evaluate a sequence of DAVIS.

EXAMPLE:
	python tools/eval_all.py

"""

import os.path as osp

from davis         import cfg,log,Timer
from prettytable   import PrettyTable as ptable
from davis.dataset import DAVISAnnotationLoader,DAVISSegmentationLoader

if __name__ == '__main__':

	sequence_name  = 'flamingo'
	technique_name = 'fcp'

	sourcedir = osp.join(cfg.PATH.SEGMENTATION_DIR,'fcp',sequence_name)

	db_annotation   = DAVISAnnotationLoader(cfg,osp.basename(sourcedir))
	db_segmentation = DAVISSegmentationLoader(
			cfg,osp.basename(sourcedir),osp.dirname(sourcedir))

	log.info('Starting evaluation of technique: "%s" on sequence "%s"'%(
		technique_name,sequence_name))


	# Initialize timer
	timer = Timer().tic()

	# Processs sequence
	J,Jm,Jo,Jt = db_annotation.eval(db_segmentation,'J')

	# Report results
	log.info("Processing time: %.3f seconds"%timer.toc())

	table = ptable(['Sequence']+['Jm','Jo','Jt'])
	table.add_row([sequence_name]+["{: .3f}".format(f) for f in [Jm,Jo,Jt]])

	print "\n" + str(table)
