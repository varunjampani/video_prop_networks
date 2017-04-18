// Copyright 2015 MPI Tuebingen

#include <csignal>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include "caffe/malabar_layers.hpp"

namespace caffe {

// This layer computes a multiplication with a Gaussian kernel
//
// bottom[0] is of size N x C x H0 x W0 (input data)
// bottom[1] is of size N x F x H0 x W0 (input features)
// bottom[2] is of size N x F x H1 x W1 (output features)
// bottom[3] is of size K  (scales)
//
// top[0] is of size N x K*C x H1 x W1
//
// K(i,j) = 1/Z(i,theta) * \exp(- scale * dist( f-in(i), f-out(j)))
//
// with Z(j,theta) = \sum_{i} K(i,j)
//
// \forall j\in H1xW1
// val-out(j) = \sum_{i \in H0xW0}  K(i,j) * val-in(i)
//
//

template <typename Dtype>
void BilateralBruteForceLayer<Dtype>::Forward_gpu(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  for(int n = 0; n < num_;++n ){
    // we iterate and the temporary data has to be filled for each example
    caffe_copy(bb_bottom_data_.count(),
               bottom[0]->gpu_data()+bottom[0]->offset(n),
               bb_bottom_data_.mutable_gpu_data());
    caffe_copy(bb_bottom_f1_.count(),
               bottom[1]->gpu_data()+bottom[1]->offset(n),
               bb_bottom_f1_.mutable_gpu_data());
    caffe_copy(bb_bottom_f2_.count(),
               bottom[2]->gpu_data()+bottom[2]->offset(n),
               bb_bottom_f2_.mutable_gpu_data());

    pdist_layer_->Forward(tmp_bottom_pdist_, tmp_top_pdist_);

    // we need to store the output for the backward pass
    // still more efficient since we half the memory size
    // this should be removed if scalar 3 is not inplace 

    // scalar is not inplace
    // caffe_copy(bb_top_pdist_.count(),
    //            bb_top_pdist_.cpu_data(),
    //            bb_pdist_full_.mutable_cpu_data()+
    //            bb_pdist_full_.offset(n));

    for (int s = 0 ; s < scales_ ; ++s) {

      // Scalar Layer
      bb_bottom_s_.mutable_cpu_data()[0] = bottom[3]->cpu_data()[s];
      scalar_layer_->Forward(tmp_bottom_scalar_, tmp_top_scalar_);

      // we need to store the output for the backward pass
      // still more efficient since we half the memory size
      // this should be removed if scalar 3 is inplace 
      // IT IS BETTER TO NOT DO IT INPLACE
      caffe_copy(bb_top_scalar_.count(),
                 bb_top_scalar_.gpu_data(),
                 bb_scalar_full_.mutable_gpu_data()+
                 bb_scalar_full_.offset(n,s));

      // we need to store the output for the backward pass
      // still more efficient since we half the memory size
      softmax_layer_->Forward(tmp_bottom_softmax_, tmp_top_softmax_);
      caffe_copy(bb_top_softmax_.count(),
                 bb_top_softmax_.gpu_data(),
                 bb_softmax_full_.mutable_gpu_data()+
                 bb_softmax_full_.offset(n,s));

      matmul_layer_->Forward(tmp_bottom_matmul_, tmp_top_matmul_);

      caffe_copy(bb_top_matmul_.count(),
                 bb_top_matmul_.gpu_data(),
                 top[0]->mutable_gpu_data()+top[0]->offset(
                     n,s*channels_));
    }
  }
}


template <typename Dtype>
void BilateralBruteForceLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  vector<bool> pd_matmul(2);
  pd_matmul[0] = true;
  pd_matmul[1] = true;

  vector<bool> pd_softmax(1);
  pd_softmax[0] = true;

  vector<bool> pd_scalar(2);
  pd_scalar[0] = false; // pdist supports no backprop
  pd_scalar[1] = true;

  // vector<bool> pd_pdist(2);
  // pd_pdist[0] = false;
  // pd_pdist[1] = false;

  if(propagate_down[0]){
    caffe_gpu_set(bottom[0]->count(), Dtype(0), bottom[0]->mutable_gpu_diff());
  }
  if(propagate_down[1]){
    // not supported
    //caffe_gpu_set(bottom[1]->count(), Dtype(0), bottom[1]->mutable_gpu_diff());
  }
  if(propagate_down[2]){
    // not supported
    //caffe_gpu_set(bottom[2]->count(), Dtype(0), bottom[2]->mutable_gpu_diff());
  }
  if(propagate_down[3]){
    caffe_gpu_set(bottom[3]->count(), Dtype(0), bottom[3]->mutable_gpu_diff());
  }

  for(int n = 0; n < num_;++n ){

    // copy the botton data back
    caffe_copy(bb_bottom_data_.count(),
               bottom[0]->gpu_data()+bottom[0]->offset(n),
               bb_bottom_data_.mutable_gpu_data());
    // scalar is not inplace
    // if(propagate_down[3]){
    //   caffe_copy(bb_top_pdist_.count(),
    //              bb_pdist_full_.gpu_data()+bb_pdist_full_.offset(n),
    //              bb_top_pdist_.mutable_gpu_data());
    // }

    for (int s = 0 ; s < scales_ ; ++s) {

      // set the diff for the chain 
      caffe_copy(bb_top_matmul_.count(),
                 top[0]->gpu_diff()+top[0]->offset(
                     n,s*channels_),
                 bb_top_matmul_.mutable_gpu_diff());

      // this is the bottom layer of matmult we need that to backprop
      caffe_copy(bb_top_softmax_.count(),
                 bb_softmax_full_.gpu_data()+
                 bb_softmax_full_.offset(n,s),
                 bb_top_softmax_.mutable_gpu_data());

      matmul_layer_->Backward(tmp_top_matmul_,
                              pd_matmul,
                              tmp_bottom_matmul_);

      // if used we backprop this is an addition
      if(propagate_down[0]){
        caffe_gpu_axpy(bb_bottom_data_.count(),
                       Dtype(1),
                       bb_bottom_data_.gpu_diff(),
                       bottom[0]->mutable_gpu_diff()+bottom[0]->offset(n));
      }
      if(propagate_down[3]){

        // this is the bottom layer of softmax we need that to backprop
        caffe_copy(bb_top_scalar_.count(),
                   bb_scalar_full_.gpu_data()+
                   bb_scalar_full_.offset(n,s),
                   bb_top_scalar_.mutable_gpu_data());

        softmax_layer_->Backward(tmp_top_softmax_,
                                 pd_softmax,
                                 tmp_bottom_softmax_);

        // update to the latest scalar
        bb_bottom_s_.mutable_cpu_data()[0] = bottom[3]->cpu_data()[s];
        scalar_layer_->Backward(tmp_top_scalar_,
                                pd_scalar,
                                tmp_bottom_scalar_);

        bottom[3]->mutable_cpu_diff()[s] += bb_bottom_s_.cpu_diff()[0];
      }
      // pdist backward not implemented
      //pdist_layer_->Backward(tmp_top_pdist_, pd_pdist, tmp_bottom_pdist_);
      // now we add the temporary backward for this scale to the output for pdist
      if(propagate_down[1]){
        // not implemented in pdist 
        // caffe_gpu_axpy(bb_bottom_f1_.count(),
        //            Dtype(1),
        //            bb_bottom_f1_.gpu_diff(),
        //            bottom[1]->mutable_gpu_diff()+bottom[1]->offset(n));
      }
      if(propagate_down[2]){
        // not implemented in pdist 
        // caffe_gpu_axpy(bb_bottom_f2_.count(),
        //            Dtype(1),
        //            bb_bottom_f2_.gpu_diff(),
        //            bottom[2]->mutable_gpu_diff()+bottom[2]->offset(n));
      }
    }
  }
  // std::cout << "bottom 0 grad"  << std::endl;
  // for(int i = 0; i < bottom[0]->count(); ++i) {
  //   std::cout << " "  << bottom[0]->gpu_diff()[i];
  // }
  // std::cout << std::endl;

  // std::cout << "bottom 3 grad"  << std::endl;
  // for(int i = 0; i < bottom[3]->count(); ++i) {
  //   std::cout << " "  << bottom[3]->gpu_diff()[i];
  // }
  // std::cout << std::endl;
}

  INSTANTIATE_LAYER_GPU_FUNCS(BilateralBruteForceLayer);
}  // namespace caffe
