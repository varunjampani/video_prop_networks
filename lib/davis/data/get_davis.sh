#!/bin/sh

# This scripts downloads the DAVIS data and unzips it.
# Adaptation of a script written in Faster R-CNN (Ross Girshick)

FILE=DAVIS-data.zip
URL=https://graphics.ethz.ch/Downloads/Data/Davis
CHECKSUM=0cb3cf9c5617209fa3cc4794e52a2ffa
DIR=$(pwd)/$(dirname "$0")

if [ ! -f $FILE ]; then
	echo "Downloading DAVIS input (1.9GB)..."
	wget $URL/$FILE

else
	echo "File already exists. Checking md5..."
fi

# CHECKING MDS
os=`uname -s`
if [ "$os" = "Linux" ]; then
	checksum=`md5sum $FILE | awk '{ print $1 }'`
elif [ "$os" = "Darwin" ]; then
	checksum=`cat $FILE | md5`
fi
echo $checksum
if [ "$checksum" = "$CHECKSUM" ]; then
	echo "Checksum is correct."
	echo "Unzipping..."
	unzip $FILE
else
	echo "Checksum is incorrect. Need to download again."
fi

rm -rf $FILE
