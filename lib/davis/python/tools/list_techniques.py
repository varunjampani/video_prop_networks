#!/usr/bin/env python

# ----------------------------------------------------------------------------
# A Benchmark Dataset and Evaluation Methodology for Video Object Segmentation
#-----------------------------------------------------------------------------
# Copyright (c) 2016 Federico Perazzi
# Licensed under the BSD License [see LICENSE for details]
# Written by Federico Perazzi
# ----------------------------------------------------------------------------

"""
List evaluated technique with corresponsing paper title and authors.

EXAMPLE:
	python tools/list_techniques.py

"""

import yaml
import json
import argparse
import os.path as osp

from davis    import cfg,log
from easydict import EasyDict as edict

from davis.dataset import db_read_techniques

def parse_args():
	"""Parse input arguments."""

	parser = argparse.ArgumentParser(
			description="""List benchmarked techniques.
			""")

	parser.add_argument("--output",
			dest='output',default=None,type=str,
			help='Output folder')

	return parser.parse_args()

if __name__ == '__main__':

	args =parse_args()

	db_techniques = db_read_techniques()

	from prettytable import PrettyTable as ptable

	table = ptable(["Abbr","Title","Authors","Conf","Year"])

	table.align     = 'l'
	technique_table = {}

	for t in db_techniques:
		technique_table[t.name]            = edict()
		technique_table[t.name].title      = t.title
		technique_table[t.name].authors    = t.authors
		technique_table[t.name].conference = t.conference
		technique_table[t.name].year       = t.year
		table.add_row([t.name,t.title,t.authors[0], t.conference,t.year])

	print "\n%s\n"%str(table)

	if args.output is not None:
		log.info("Saving list of techniques in: %s"%args.output)
		with open(args.output,'w') as f:
			f.write(json.dumps(technique_table,indent=2))
