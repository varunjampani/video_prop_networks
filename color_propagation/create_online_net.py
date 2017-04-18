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
from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2
import tempfile


#####
# VPN (BNN + CNN) network
#####

def create_bnn_cnn_net(num_input_points, phase = None):

    n = caffe.NetSpec()

    n.input_color = L.Input(shape=[dict(dim=[1, 2, 1, num_input_points])])
    n.in_features = L.Input(shape=[dict(dim=[1, 4, 1, num_input_points])])
    n.out_features = L.Input(shape=[dict(dim=[1, 4, 480, 854])])
    n.scales = L.Input(shape=[dict(dim=[1, 4, 1, 1])])

    n.flatten_scales = L.Flatten(n.scales, flatten_param= dict(axis = 0))

    n.in_scaled_features = L.Scale(n.in_features, n.flatten_scales, scale_param= dict(axis = 1))
    n.out_scaled_features = L.Scale(n.out_features, n.flatten_scales, scale_param= dict(axis = 1))

    ### Start of BNN

    # BNN - stage - 1
    n.out_color1 = L.Permutohedral(n.input_color, n.in_scaled_features, n.out_scaled_features,
                                   permutohedral_param = dict(
                                   num_output = 32, group = 1, neighborhood_size = 0, bias_term = True,
                                   norm_type = P.Permutohedral.AFTER,
                                   offset_type = P.Permutohedral.NONE),
                                   filter_filler = dict(type = 'gaussian', std = 0.01),
                                   bias_filler = dict(type = 'constant', value = 0.5),
                                   param=[{'lr_mult':1, 'decay_mult':1}, {'lr_mult':2, 'decay_mult':0}])
    n.bnn_out_relu_1 = L.ReLU(n.out_color1, in_place = True)

    # BNN - stage - 2
    n.out_color2 = L.Permutohedral(n.bnn_out_relu_1, n.out_scaled_features, n.out_scaled_features,
                                   permutohedral_param = dict(
                                   num_output = 32, group = 1, neighborhood_size = 0, bias_term = True,
                                   norm_type = P.Permutohedral.AFTER,
                                   offset_type = P.Permutohedral.NONE),
                                   filter_filler = dict(type = 'gaussian', std = 0.01),
                                   bias_filler = dict(type = 'constant', value = 0),
                                   param=[{'lr_mult':1, 'decay_mult':1}, {'lr_mult':2, 'decay_mult':0}])
    n.bnn_out_relu_2 = L.ReLU(n.out_color2, in_place=True)

    # BNN - combination
    n.connection_out = L.Concat(n.bnn_out_relu_1, n.bnn_out_relu_2)
    n.out_color_bilateral = L.Convolution(n.connection_out,
                                          convolution_param = dict(num_output = 2, kernel_size = 1, stride = 1,
                                                                   weight_filler = dict(type = 'gaussian', std = 0.01),
                                                                   bias_filler = dict(type = 'constant', value = 0)),
                                          param=[{'lr_mult':1, 'decay_mult':1}, {'lr_mult':2, 'decay_mult':0}])
    n.out_color_bilateral_relu = L.ReLU(n.out_color_bilateral, in_place=True)

    ### Start of CNN

    # CNN - Stage 1
    n.out_color_spatial1 = L.Convolution(n.out_color_bilateral_relu,
                                      convolution_param = dict(num_output = 32, kernel_size = 3, stride = 1,
                                                               pad_h = 1, pad_w = 1,
                                                               weight_filler = dict(type = 'gaussian', std = 0.01),
                                                               bias_filler = dict(type = 'constant', value = 0)),
                                      param=[{'lr_mult':1, 'decay_mult':1}, {'lr_mult':2, 'decay_mult':0}])
    n.out_color_spatial_relu1 = L.ReLU(n.out_color_spatial1, in_place=True)

    # CNN - Stage 2
    n.out_color_spatial2 = L.Convolution(n.out_color_spatial_relu1,
                                      convolution_param = dict(num_output = 32, kernel_size = 3, stride = 1,
                                                               pad_h = 1, pad_w = 1,
                                                               weight_filler = dict(type = 'gaussian', std = 0.01),
                                                               bias_filler = dict(type = 'constant', value = 0)),
                                      param=[{'lr_mult':1, 'decay_mult':1}, {'lr_mult':2, 'decay_mult':0}])
    n.out_color_spatial_relu2 = L.ReLU(n.out_color_spatial2, in_place=True)

    # CNN - Stage 3
    n.out_color_spatial = L.Convolution(n.out_color_spatial_relu2,
                                        convolution_param = dict(num_output = 2, kernel_size = 3, stride = 1,
                                                                 pad_h = 1, pad_w = 1,
                                                                 weight_filler = dict(type = 'gaussian', std =  0.01),
                                                                 bias_filler = dict(type = 'constant', value = 0)),
                                        param=[{'lr_mult':1, 'decay_mult':1}, {'lr_mult':2, 'decay_mult':0}])
    n.out_color_spatial_relu = L.ReLU(n.out_color_spatial, in_place=True)

    n.final_connection_out = L.Concat(n.out_color_bilateral_relu, n.out_color_spatial_relu)
    n.out_color_result = L.Convolution(n.final_connection_out,
                                       convolution_param = dict(num_output = 2, kernel_size = 1, stride = 1,
                                                                weight_filler = dict(type = 'gaussian', std = 0.01),
                                                                bias_filler = dict(type = 'constant', value = 0.0)),
                                       param=[{'lr_mult':1, 'decay_mult':1}, {'lr_mult':2, 'decay_mult':0}])

    return n.to_proto()


def load_bnn_cnn_deploy_net(num_input_frames):
    # Create the prototxt
    net_proto = create_bnn_cnn_net(num_input_frames)

    # Save to temporary file and load
    f = tempfile.NamedTemporaryFile(mode='w+', delete=False)
    f.write(str(net_proto))
    f.close()
    return caffe.Net(f.name, caffe.TEST)


###############
# BNN network
###############

def create_bnn_deploy_net(num_input_points):

    n = caffe.NetSpec()
    n.input_color = L.Input(shape=[dict(dim=[1, 2, 1, num_input_points])])
    n.in_features = L.Input(shape=[dict(dim=[1, 4, 1, num_input_points])])
    n.out_features = L.Input(shape=[dict(dim=[1, 4, 480, 854])])
    n.scales = L.Input(shape=[dict(dim=[1, 4, 1, 1])])

    n.flatten_scales = L.Flatten(n.scales, flatten_param= dict(axis = 0))

    n.in_scaled_features = L.Scale(n.in_features, n.flatten_scales, scale_param= dict(axis = 1))
    n.out_scaled_features = L.Scale(n.out_features, n.flatten_scales, scale_param= dict(axis = 1))

    n.out_color_result = L.Permutohedral(n.input_color, n.in_scaled_features, n.out_scaled_features,
                                         permutohedral_param = dict(
                                         num_output = 2, group = 1, neighborhood_size = 0,
                                         norm_type = P.Permutohedral.AFTER,
                                         offset_type = P.Permutohedral.DIAG))
    return n.to_proto()


def load_bnn_deploy_net(num_input_points):
    # Create the prototxt
    net_proto = create_bnn_deploy_net(num_input_points)

    # Save to temporary file and load
    f = tempfile.NamedTemporaryFile(mode='w+', delete=False)
    f.write(str(net_proto))
    f.close()
    return caffe.Net(f.name, caffe.TEST)
