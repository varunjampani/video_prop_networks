 #!/bin/bash

 # ---------------------------------------------------------------------------
 # Video Propagation Networks
 #----------------------------------------------------------------------------
 # Copyright 2017 Max Planck Society
 # Distributed under the BSD-3 Software license [see LICENSE.txt for details]
 # ---------------------------------------------------------------------------

set -e

FOLDID=$1
TRAIN_LOG_DIR="../data/training_data/training_logs/"

for STAGEID in 1
do
  LOG_FILE=$TRAIN_LOG_DIR'FOLD'$FOLDID'_STAGE'$STAGEID'.log'
  echo $FOLDID
  echo $STAGEID
  echo $LOG_FILE
  python train_online_seg.py $FOLDID $STAGEID 2>&1 | tee ${LOG_FILE} || exit 1&&
  python select_model.py $FOLDID $STAGEID || exit 1 &&
  python run_segmentation.py $STAGEID $FOLDID || exit 1
done
