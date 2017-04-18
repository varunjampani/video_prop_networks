#!/usr/bin/env python

'''
    File name: create_online_net.py
    Author: Varun Jampani
'''

# ---------------------------------------------------------------------------
# Video Propagation Networks
#----------------------------------------------------------------------------
# Copyright 2017 Max Planck Society
# Distributed under the BSD-3 Software license [see LICENSE.txt for details]
# ---------------------------------------------------------------------------

from init_caffe import *
from davis_data import *
from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2
import tempfile

max_spixels = MAX_SPIXELS

def batch_norm(bottom, phase):

    if phase == 'TRAIN':
        bnorm = L.BatchNorm(bottom, batch_norm_param = dict(use_global_stats=False),
                            param=[{'lr_mult':0}, {'lr_mult':0},{'lr_mult':0}])
    else:
        bnorm = L.BatchNorm(bottom, batch_norm_param = dict(use_global_stats=True),
                            param=[{'lr_mult':0}, {'lr_mult':0},{'lr_mult':0}])

    bnorm_scaled = L.Scale(bnorm, scale_param = dict(bias_term = True))

    return bnorm_scaled


def normalize(bottom, dim):

    bottom_relu = L.ReLU(bottom)
    sum = L.Convolution(bottom_relu,
                        convolution_param = dict(num_output = 1, kernel_size = 1, stride = 1,
                                                 weight_filler = dict(type = 'constant', value = 1),
                                                 bias_filler = dict(type = 'constant', value = 0)),
                        param=[{'lr_mult':0, 'decay_mult':0}, {'lr_mult':0, 'decay_mult':0}])

    denom = L.Power(sum, power=(-1.0), shift=1e-12)
    denom = L.Tile(denom, axis=1, tiles=dim)
    return L.Eltwise(bottom_relu, denom, operation=P.Eltwise.PROD)

#####
# 2-Kernel main (BNN + CNN) video propagation network
#####

def create_bnn_cnn_net_fold_stage(num_input_frames, fold_id = '0',
                                  stage_id = '1', phase = None):

    n = caffe.NetSpec()

    if phase == 'TRAIN':
        n.img, n.padimg, n.unary, n.in_features, n.out_features, n.spixel_indices, n.scales1, n.scales2, n.unary_scales, n.label = \
            L.Python(python_param = dict(module = "input_data_layer", layer = "InputRead",
                                        param_str = "TRAIN_1000000_" + fold_id + '_' + stage_id),
                     include = dict(phase = 0),
                     ntop = 10)
    elif phase == 'TEST':
        n.img, n.padimg, n.unary, n.in_features, n.out_features, n.spixel_indices, n.scales1, n.scales2, n.unary_scales, n.label = \
            L.Python(python_param = dict(module = "input_data_layer", layer = "InputRead",
                                         param_str = "VAL_50_" + fold_id + '_' + stage_id),
                     include = dict(phase = 1),
                     ntop = 10)
    else:
        n.img = L.Input(shape=[dict(dim=[1, 3, 480, 854])])
        n.padimg = L.Input(shape=[dict(dim=[1, 3, 481, 857])])

        n.unary = L.Input(shape=[dict(dim=[1, 2, num_input_frames, max_spixels])])
        n.in_features = L.Input(shape=[dict(dim=[1, 6, num_input_frames, max_spixels])])
        n.out_features = L.Input(shape=[dict(dim=[1, 6, 1, max_spixels])])
        n.spixel_indices = L.Input(shape=[dict(dim=[1, 1, 480, 854])])
        n.scales1 = L.Input(shape=[dict(dim=[1, 6, 1, 1])])
        n.scales2 = L.Input(shape=[dict(dim=[1, 6, 1, 1])])
        n.unary_scales = L.Input(shape=[dict(dim=[1, 1, num_input_frames, 1])])

    n.flatten_scales1 = L.Flatten(n.scales1, flatten_param= dict(axis = 0))
    n.flatten_scales2 = L.Flatten(n.scales2, flatten_param= dict(axis = 0))
    n.flatten_unary_scales = L.Flatten(n.unary_scales, flatten_param=dict(axis=0))

    n.in_scaled_features1 = L.Scale(n.in_features, n.flatten_scales1, scale_param= dict(axis = 1))
    n.out_scaled_features1 = L.Scale(n.out_features, n.flatten_scales1, scale_param= dict(axis = 1))

    n.in_scaled_features2 = L.Scale(n.in_features, n.flatten_scales2, scale_param= dict(axis = 1))
    n.out_scaled_features2 = L.Scale(n.out_features, n.flatten_scales2, scale_param= dict(axis = 1))
    n.scaled_unary = L.Scale(n.unary, n.flatten_unary_scales, scale_param=dict(axis=2))


    ### Start of BNN

    # BNN - stage - 1
    n.out_seg1 = L.Permutohedral(n.scaled_unary, n.in_scaled_features1, n.out_scaled_features1,
                                 permutohedral_param = dict(
                                 num_output = 32, group = 1, neighborhood_size = 0, bias_term = True,
                                 norm_type = P.Permutohedral.AFTER,
                                 offset_type = P.Permutohedral.NONE),
                                 filter_filler = dict(type = 'gaussian', std = 0.01),
                                 bias_filler = dict(type = 'constant', value = 0),
                                 param=[{'lr_mult':1, 'decay_mult':1}, {'lr_mult':2, 'decay_mult':0}])

    n.out_seg2 = L.Permutohedral(n.scaled_unary, n.in_scaled_features2, n.out_scaled_features2,
                                 permutohedral_param = dict(
                                 num_output = 32, group = 1, neighborhood_size = 0, bias_term = True,
                                 norm_type = P.Permutohedral.AFTER,
                                 offset_type = P.Permutohedral.NONE),
                                 filter_filler = dict(type = 'gaussian', std = 0.01),
                                 bias_filler = dict(type = 'constant', value = 0),
                                 param=[{'lr_mult':1, 'decay_mult':1}, {'lr_mult':2, 'decay_mult':0}])

    n.concat_out_seg_1 = L.Concat(n.out_seg1, n.out_seg2, concat_param=dict(axis=1))
    n.concat_out_relu_1 = L.ReLU(n.concat_out_seg_1, in_place = True)

    # BNN - stage - 2
    n.out_seg3 = L.Permutohedral(n.concat_out_relu_1, n.out_scaled_features1, n.out_scaled_features1,
                                 permutohedral_param = dict(
                                 num_output = 32, group = 1, neighborhood_size = 0, bias_term = True,
                                 norm_type = P.Permutohedral.AFTER,
                                 offset_type = P.Permutohedral.NONE),
                                 filter_filler = dict(type = 'gaussian', std = 0.01),
                                 bias_filler = dict(type = 'constant', value = 0),
                                 param=[{'lr_mult':1, 'decay_mult':1}, {'lr_mult':2, 'decay_mult':0}])

    n.out_seg4 = L.Permutohedral(n.concat_out_relu_1, n.out_scaled_features2, n.out_scaled_features2,
                                 permutohedral_param = dict(
                                 num_output = 32, group = 1, neighborhood_size = 0, bias_term = True,
                                 norm_type = P.Permutohedral.AFTER,
                                 offset_type = P.Permutohedral.NONE),
                                 filter_filler = dict(type = 'gaussian', std = 0.01),
                                 bias_filler = dict(type = 'constant', value = 0),
                                 param=[{'lr_mult':1, 'decay_mult':1}, {'lr_mult':2, 'decay_mult':0}])
    n.concat_out_seg_2 = L.Concat(n.out_seg3, n.out_seg4, concat_param=dict(axis=1))
    n.concat_out_relu_2 = L.ReLU(n.concat_out_seg_2, in_place=True)

    # BNN - combination
    n.connection_out = L.Concat(n.concat_out_relu_1, n.concat_out_relu_2)
    n.spixel_out_seg = L.Convolution(n.connection_out,
                                     convolution_param = dict(num_output = 2, kernel_size = 1, stride = 1,
                                                              weight_filler = dict(type = 'gaussian', std = 0.01),
                                                              bias_filler = dict(type = 'constant', value = 0)),
                                     param=[{'lr_mult':1, 'decay_mult':1}, {'lr_mult':2, 'decay_mult':0}])
    n.spixel_out_seg_relu = L.ReLU(n.spixel_out_seg, in_place=True)

    # Going from superpixels to pixels
    n.out_seg_bilateral = L.Smear(n.spixel_out_seg_relu, n.spixel_indices)

    ### BNN - DeepLab Combination
    n.deeplab_seg_presoftmax = deeplab(n.padimg, n.img, n.spixel_indices)
    n.deeplab_seg = L.Softmax(n.deeplab_seg_presoftmax)
    n.bnn_deeplab_connection = L.Concat(n.out_seg_bilateral, n.deeplab_seg)
    n.bnn_deeplab_seg = L.Convolution(n.bnn_deeplab_connection,
                                      convolution_param = dict(num_output = 2, kernel_size = 1, stride = 1,
                                                               weight_filler = dict(type = 'gaussian', std = 0.01),
                                                               bias_filler = dict(type = 'constant', value = 0)),
                                      param=[{'lr_mult':1, 'decay_mult':1}, {'lr_mult':2, 'decay_mult':0}])
    n.bnn_deeplab_seg_relu = L.ReLU(n.bnn_deeplab_seg, in_place=True)

    ### Start of CNN

    # CNN - Stage 1
    n.out_seg_spatial1 = L.Convolution(n.bnn_deeplab_seg_relu,
                                       convolution_param = dict(num_output = 32, kernel_size = 3, stride = 1,
                                                                pad_h = 1, pad_w = 1,
                                                                weight_filler = dict(type = 'gaussian', std = 0.01),
                                                                bias_filler = dict(type = 'constant', value = 0)),
                                       param=[{'lr_mult':1, 'decay_mult':1}, {'lr_mult':2, 'decay_mult':0}])
    n.out_seg_spatial_relu1 = L.ReLU(n.out_seg_spatial1, in_place=True)

    # CNN - Stage 2
    n.out_seg_spatial2 = L.Convolution(n.out_seg_spatial_relu1,
                                       convolution_param = dict(num_output = 32, kernel_size = 3, stride = 1,
                                                                pad_h = 1, pad_w = 1,
                                                                weight_filler = dict(type = 'gaussian', std = 0.01),
                                                                bias_filler = dict(type = 'constant', value = 0)),
                                       param=[{'lr_mult':1, 'decay_mult':1}, {'lr_mult':2, 'decay_mult':0}])
    n.out_seg_spatial_relu2 = L.ReLU(n.out_seg_spatial2, in_place=True)

    # CNN - Stage 3
    n.out_seg_spatial = L.Convolution(n.out_seg_spatial_relu2,
                                      convolution_param = dict(num_output = 2, kernel_size = 3, stride = 1,
                                                               pad_h = 1, pad_w = 1,
                                                               weight_filler = dict(type = 'gaussian', std =  0.01),
                                                               bias_filler = dict(type = 'constant', value = 0.5)),
                                      param=[{'lr_mult':1, 'decay_mult':1}, {'lr_mult':2, 'decay_mult':0}])

    # Normalization
    n.out_seg = normalize(n.out_seg_spatial, 2)

    if phase == 'TRAIN' or phase == 'TEST':
        n.loss = L.LossWithoutSoftmax(n.out_seg, n.label, loss_param = dict(ignore_label = 1000), loss_weight=1)
        n.accuracy = L.Accuracy(n.out_seg, n.label, accuracy_param = dict(ignore_label = 1000))
        n.loss2 = L.SoftmaxWithLoss(n.deeplab_seg_presoftmax, n.label,
                                    loss_param = dict(ignore_label = 1000), loss_weight=1)
        n.accuracy2 = L.Accuracy(n.deeplab_seg_presoftmax, n.label, accuracy_param = dict(ignore_label = 1000))
    else:
        n.spixel_out_seg_2 = L.SpixelFeature(n.out_seg, n.spixel_indices,
                                             spixel_feature_param = dict(type = P.SpixelFeature.AVGRGB,
                                                                         max_spixels = 12000, rgb_scale = 1.0))
        n.spixel_out_seg_final = normalize(n.spixel_out_seg_2, 2)

    return n.to_proto()




#####
# 2-Kernel gauss_permutohedral network (for all points)
#####

def create_bnn_deploy_net(num_input_frames):

    n = caffe.NetSpec()
    n.unary = L.Input(shape=[dict(dim=[1, 2, num_input_frames, max_spixels])])
    n.in_features = L.Input(shape=[dict(dim=[1, 6, num_input_frames, max_spixels])])
    n.out_features = L.Input(shape=[dict(dim=[1, 6, 1, max_spixels])])
    n.spixel_indices = L.Input(shape=[dict(dim=[1, 1, 480, 854])])
    n.scales1 = L.Input(shape=[dict(dim=[1, 6, 1, 1])])
    n.scales2 = L.Input(shape=[dict(dim=[1, 6, 1, 1])])
    n.unary_scales = L.Input(shape=[dict(dim=[1, 1, num_input_frames, 1])])

    n.flatten_scales1 = L.Flatten(n.scales1, flatten_param= dict(axis = 0))
    n.flatten_scales2 = L.Flatten(n.scales2, flatten_param= dict(axis = 0))
    n.flatten_unary_scales = L.Flatten(n.unary_scales, flatten_param=dict(axis=0))

    n.in_scaled_features1 = L.Scale(n.in_features, n.flatten_scales1, scale_param= dict(axis = 1))
    n.out_scaled_features1 = L.Scale(n.out_features, n.flatten_scales1, scale_param= dict(axis = 1))

    n.in_scaled_features2 = L.Scale(n.in_features, n.flatten_scales2, scale_param= dict(axis = 1))
    n.out_scaled_features2 = L.Scale(n.out_features, n.flatten_scales2, scale_param= dict(axis = 1))
    n.scaled_unary = L.Scale(n.unary, n.flatten_unary_scales, scale_param=dict(axis=2))


    n.out_seg1 = L.Permutohedral(n.scaled_unary, n.in_scaled_features1, n.out_scaled_features1,
                                 permutohedral_param = dict(
                                 num_output = 32, group = 1, neighborhood_size = 0, bias_term = True,
                                 norm_type = P.Permutohedral.AFTER,
                                 offset_type = P.Permutohedral.NONE),
                                 filter_filler = dict(type = 'gaussian', std = 0.01),
                                 bias_filler = dict(type = 'constant', value = 0),
                                 param=[{'lr_mult':1, 'decay_mult':1}, {'lr_mult':2, 'decay_mult':0}])

    n.out_seg2 = L.Permutohedral(n.unary, n.in_scaled_features2, n.out_scaled_features2,
                                 permutohedral_param = dict(
                                 num_output = 32, group = 1, neighborhood_size = 0, bias_term = True,
                                 norm_type = P.Permutohedral.AFTER,
                                 offset_type = P.Permutohedral.NONE),
                                 filter_filler = dict(type = 'gaussian', std = 0.01),
                                 bias_filler = dict(type = 'constant', value = 0),
                                 param=[{'lr_mult':1, 'decay_mult':1}, {'lr_mult':2, 'decay_mult':0}])

    n.concat_out_seg = L.Concat(n.out_seg1, n.out_seg2, concat_param=dict(axis=1))
    n.concat_out_relu = L.ReLU(n.concat_out_seg, in_place = True)

    n.spixel_out_seg1 = L.Convolution(n.concat_out_relu,
                                      convolution_param = dict(num_output = 2, kernel_size = 1, stride = 1,
                                                               weight_filler = dict(type = 'gaussian', std = 0.01),
                                                               bias_filler = dict(type = 'constant', value = 0)),
                                      param=[{'lr_mult':1, 'decay_mult':1}, {'lr_mult':2, 'decay_mult':0}])

    n.spixel_out_seg_final = normalize(n.spixel_out_seg1, 2)

    n.out_seg = L.Smear(n.spixel_out_seg_final, n.spixel_indices)

    return n.to_proto()


def load_bnn_cnn_deploy_net_fold_stage(num_input_frames, stage_id):

    # Create the prototxt
    if int(stage_id) == 0:
        net_proto = create_bnn_deploy_net(num_input_frames)
    else:
        net_proto = create_bnn_cnn_net_fold_stage(num_input_frames, fold_id = '0',
                                                  stage_id = '1', phase = None)

    # Save to temporary file and load
    f = tempfile.NamedTemporaryFile(mode='w+', delete=False)
    f.write(str(net_proto))
    f.close()
    return caffe.Net(f.name, caffe.TEST)

def get_bnn_cnn_train_net_fold_stage(num_input_frames, fold_id, stage_id, phase):

    # Create the prototxt
    net_proto = create_bnn_cnn_net_fold_stage(num_input_frames, fold_id, stage_id, phase)

    # Save to temporary file and load
    f = tempfile.NamedTemporaryFile(mode='w+', delete=False)
    f.write(str(net_proto))
    f.close()
    return f.name


########
# DeepLab Net specification
########

def deeplab(padimg, img, spixel_indices):

    # Conv1
    conv1_1 = L.Convolution(padimg, convolution_param = dict(num_output = 64, kernel_size = 3, stride = 1, pad = 1),
                            param=[{'lr_mult':0, 'decay_mult':0}, {'lr_mult':0, 'decay_mult':0}])
    conv1_1 = L.ReLU(conv1_1, in_place=True)
    conv1_2 = L.Convolution(conv1_1, convolution_param = dict(num_output = 64, kernel_size = 3, stride = 1, pad = 1),
                            param=[{'lr_mult':0, 'decay_mult':0}, {'lr_mult':0, 'decay_mult':0}])
    conv1_2 = L.ReLU(conv1_2, in_place=True)
    pool1 = L.Pooling(conv1_2, pooling_param = dict(kernel_size = 3, stride = 2, pad = 1, pool = P.Pooling.MAX))

    # Conv2
    conv2_1 = L.Convolution(pool1, convolution_param = dict(num_output = 128, kernel_size = 3, stride = 1, pad = 1),
                            param=[{'lr_mult':0, 'decay_mult':0}, {'lr_mult':0, 'decay_mult':0}])
    conv2_1 = L.ReLU(conv2_1, in_place=True)
    conv2_2 = L.Convolution(conv2_1, convolution_param = dict(num_output = 128, kernel_size = 3, stride = 1, pad = 1),
                            param=[{'lr_mult':0, 'decay_mult':0}, {'lr_mult':0, 'decay_mult':0}])
    conv2_2 = L.ReLU(conv2_2, in_place=True)
    pool2 = L.Pooling(conv2_2, pooling_param = dict(kernel_size = 3, stride = 2, pad = 1, pool = P.Pooling.MAX))

    # Conv3
    conv3_1 = L.Convolution(pool2, convolution_param = dict(num_output = 256, kernel_size = 3, stride = 1, pad = 1),
                            param=[{'lr_mult':0, 'decay_mult':0}, {'lr_mult':0, 'decay_mult':0}])
    conv3_1 = L.ReLU(conv3_1, in_place=True)
    conv3_2 = L.Convolution(conv3_1, convolution_param = dict(num_output = 256, kernel_size = 3, stride = 1, pad = 1),
                            param=[{'lr_mult':0, 'decay_mult':0}, {'lr_mult':0, 'decay_mult':0}])
    conv3_2 = L.ReLU(conv3_2, in_place=True)
    conv3_3 = L.Convolution(conv3_2, convolution_param = dict(num_output = 256, kernel_size = 3, stride = 1, pad = 1),
                            param=[{'lr_mult':0, 'decay_mult':0}, {'lr_mult':0, 'decay_mult':0}])
    conv3_3 = L.ReLU(conv3_3, in_place=True)
    pool3 = L.Pooling(conv3_3, pooling_param = dict(kernel_size = 3, stride = 2, pad = 1, pool = P.Pooling.MAX))

    # Conv4
    conv4_1 = L.Convolution(pool3, convolution_param = dict(num_output = 512, kernel_size = 3, stride = 1, pad = 1),
                            param=[{'lr_mult':0, 'decay_mult':0}, {'lr_mult':0, 'decay_mult':0}])
    conv4_1 = L.ReLU(conv4_1, in_place=True)
    conv4_2 = L.Convolution(conv4_1, convolution_param = dict(num_output = 512, kernel_size = 3, stride = 1, pad = 1),
                            param=[{'lr_mult':0, 'decay_mult':0}, {'lr_mult':0, 'decay_mult':0}])
    conv4_2 = L.ReLU(conv4_2, in_place=True)
    conv4_3 = L.Convolution(conv4_2, convolution_param = dict(num_output = 512, kernel_size = 3, stride = 1, pad = 1),
                            param=[{'lr_mult':0, 'decay_mult':0}, {'lr_mult':0, 'decay_mult':0}])
    conv4_3 = L.ReLU(conv4_3, in_place=True)
    pool4 = L.Pooling(conv4_3, pooling_param = dict(kernel_size = 3, stride = 1, pad = 1, pool = P.Pooling.MAX))

    # #Conv5
    conv5_1 = L.Convolution(pool4, convolution_param = dict(num_output = 512, kernel_size = 3, stride = 1,
                                                            pad = 2, dilation = 2, engine=1),
                            param=[{'lr_mult':0, 'decay_mult':0}, {'lr_mult':0, 'decay_mult':0}])
    conv5_1 = L.ReLU(conv5_1, in_place=True)
    conv5_2 = L.Convolution(conv5_1, convolution_param = dict(num_output = 512, kernel_size = 3, stride = 1,
                                                              pad = 2, dilation = 2, engine=1),
                            param=[{'lr_mult':0, 'decay_mult':0}, {'lr_mult':0, 'decay_mult':0}])
    conv5_2 = L.ReLU(conv5_2, in_place=True)
    conv5_3 = L.Convolution(conv5_2, convolution_param = dict(num_output = 512, kernel_size = 3, stride = 1,
                                                              pad = 2, dilation = 2, engine=1),
                            param=[{'lr_mult':0, 'decay_mult':0}, {'lr_mult':0, 'decay_mult':0}])
    conv5_3 = L.ReLU(conv5_3, in_place=True)
    pool5 = L.Pooling(conv5_3, pooling_param = dict(kernel_size = 3, stride = 1, pad = 1, pool = P.Pooling.MAX))
    pool5a = L.Pooling(pool5, pooling_param = dict(kernel_size = 3, stride = 1, pad = 1, pool = P.Pooling.MAX))

    #FC-6
    fc6 = L.Convolution(pool5a, convolution_param = dict(num_output = 1024, kernel_size = 3,
                                                         pad = 12, dilation = 12, engine=1),
                        param=[{'lr_mult':0, 'decay_mult':0}, {'lr_mult':0, 'decay_mult':0}])
    fc6 = L.ReLU(fc6, in_place=True)
    fc6 = L.Dropout(fc6, dropout_param = dict(dropout_ratio = 0.5), in_place = True)

    #FC-7
    fc7 = L.Convolution(fc6, convolution_param = dict(num_output = 1024, kernel_size = 1),
                        param=[{'lr_mult':0, 'decay_mult':0}, {'lr_mult':0, 'decay_mult':0}])
    fc7 = L.ReLU(fc7, in_place=True)
    fc7 = L.Dropout(fc7, dropout_param = dict(dropout_ratio = 0.5), in_place = True)

    #FC-8
    fc8_deeplab = L.Convolution(fc7, convolution_param = dict(num_output = 2, kernel_size = 1,
                                                              weight_filler = dict(type = 'gaussian', std = 0.01),
                                                              bias_filler = dict(type = 'constant', value = 0)),
                                param=[{'lr_mult':1, 'decay_mult':1}, {'lr_mult':2, 'decay_mult':0}])

    #Interpolate
    fc8_interp = L.Interp(fc8_deeplab, interp_param = dict(zoom_factor = 8))

    #Crop to match required dimensions
    fc8_crop = L.Crop(fc8_interp, img, crop_param = dict(axis = 2, offset = [0, 0]))

    return fc8_crop
