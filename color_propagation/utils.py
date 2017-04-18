#!/usr/bin/env python

'''
    File name: utils.py
    Author: Varun Jampani
'''

# ---------------------------------------------------------------------------
# Video Propagation Networks
#----------------------------------------------------------------------------
# Copyright 2017 Max Planck Society
# Distributed under the BSD-3 Software license [see LICENSE.txt for details]
# ---------------------------------------------------------------------------

import numpy as np
import scipy.misc
import math

def convert_to_rgb(im):
    full_result = scipy.misc.toimage(im, mode='YCbCr')
    rgb_result = np.array(full_result.convert('RGB'))
    return rgb_result

def computeRMSE(result_image, gt_image):
    RMSE = math.sqrt(np.mean((result_image.astype('float') - gt_image.astype('float'))**2))
    PSNR = 20 * math.log( 255/(RMSE+1e-27), 10)
    return [RMSE,PSNR]
