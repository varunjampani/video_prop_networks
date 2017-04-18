// Copyright 2015 MPI Tuebingen

#include <csignal>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include "boost/array.hpp"
#include "boost/make_shared.hpp"

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
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
void BilateralBruteForceLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  BilateralBruteForceParameter param_ =
    this->layer_param_.bilateral_brute_force_param();

  LayerParameter pdist_param;
  pdist_param.mutable_pdist_param()->set_ignore_value(255);
  pdist_param.mutable_pdist_param()->set_scale_value(-1);
  pdist_layer_.reset(new PdistLayer<Dtype>(pdist_param));

  LayerParameter scalar_param;
  scalar_layer_.reset(new Scalar3Layer<Dtype>(scalar_param));

  LayerParameter softmax_param;
  softmax_param.mutable_softmax_param()->set_axis(3);
  softmax_layer_.reset(new SoftmaxLayer<Dtype>(softmax_param));

  LayerParameter matmul_param;
  matmul_layer_.reset(new MatMul2Layer<Dtype>(matmul_param));

  Blob<Dtype>& data_Blob = *bottom[0];
  Blob<Dtype>& in_feature_Blob = *bottom[1];
  Blob<Dtype>& out_feature_Blob = *bottom[2];
  Blob<Dtype>& scales_Blob = *bottom[3];
  num_ = data_Blob.num(); 
  channels_ = data_Blob.channels();
  in_height_ = data_Blob.height();
  in_width_ = data_Blob.width();
  scales_ = scales_Blob.count();
  feature_size_ = in_feature_Blob.channels();
  out_height_ = out_feature_Blob.height();
  out_width_ = out_feature_Blob.width();
  CHECK_EQ(num_, in_feature_Blob.num());
  CHECK_EQ(num_, out_feature_Blob.num());
  CHECK_EQ(in_height_, in_feature_Blob.height());
  CHECK_EQ(in_width_, in_feature_Blob.width());
  CHECK_EQ(feature_size_, out_feature_Blob.channels());


  // this is for now only for one example
  vector<int> data_shape = bottom[0]->shape();
  data_shape[0] = 1;
  bb_bottom_data_.Reshape(data_shape);

  // this is for now only for one example
  vector<int> f1_shape = bottom[1]->shape();
  f1_shape[0] = 1;
  bb_bottom_f1_.Reshape(f1_shape);

  // this is for now only for one example
  vector<int> f2_shape = bottom[2]->shape();
  f2_shape[0] = 1;
  bb_bottom_f2_.Reshape(f2_shape);

  // we know we only want 1 scale
  bb_bottom_s_.Reshape(1,1,1,1);

  // this has to be initialized exactly one time
  tmp_bottom_pdist_.resize(2);
  tmp_top_pdist_.resize(1);
  tmp_bottom_scalar_.resize(2);
  tmp_top_scalar_.resize(1);
  tmp_bottom_softmax_.resize(1);
  tmp_top_softmax_.resize(1);
  tmp_bottom_matmul_.resize(2);
  tmp_top_matmul_.resize(1);
  
  // this mapping has to be done once
  tmp_bottom_pdist_[0] = &bb_bottom_f1_;
  tmp_bottom_pdist_[1] = &bb_bottom_f2_;
  tmp_top_pdist_[0] = &bb_top_pdist_;

  tmp_bottom_scalar_[0] = &bb_top_pdist_;
  tmp_bottom_scalar_[1] = &bb_bottom_s_;
  tmp_top_scalar_[0] = &bb_top_scalar_;

  tmp_bottom_softmax_[0] = &bb_top_scalar_;
  tmp_top_softmax_[0] = &bb_top_softmax_;

  tmp_bottom_matmul_[0] = &bb_bottom_data_;
  tmp_bottom_matmul_[1] = &bb_top_softmax_;
  tmp_top_matmul_[0] = &bb_top_matmul_;

  // the order is important otherwise the layers are not in the right shape
  pdist_layer_->LayerSetUp(tmp_bottom_pdist_, tmp_top_pdist_);
  scalar_layer_->LayerSetUp(tmp_bottom_scalar_, tmp_top_scalar_);
  softmax_layer_->LayerSetUp(tmp_bottom_softmax_, tmp_top_softmax_);
  matmul_layer_->LayerSetUp(tmp_bottom_matmul_, tmp_top_matmul_);

}

template <typename Dtype>
void BilateralBruteForceLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  vector<int> data_shape = bottom[0]->shape();
  data_shape[0] = 1;
  bb_bottom_data_.Reshape(data_shape);

  // this is for now only for one example
  vector<int> f1_shape = bottom[1]->shape();
  f1_shape[0] = 1;
  bb_bottom_f1_.Reshape(f1_shape);

  // this is for now only for one example
  vector<int> f2_shape = bottom[2]->shape();
  f2_shape[0] = 1;
  bb_bottom_f2_.Reshape(f2_shape);

  // we know we only want 1 scale
  bb_bottom_s_.Reshape(1,1,1,1);
  Blob<Dtype>& data_Blob = *bottom[0];
  Blob<Dtype>& in_feature_Blob = *bottom[1];
  Blob<Dtype>& out_feature_Blob = *bottom[2];
  Blob<Dtype>& scales_Blob = *bottom[3];
  num_ = data_Blob.num(); 
  channels_ = data_Blob.channels();
  in_height_ = data_Blob.height();
  in_width_ = data_Blob.width();
  scales_ = scales_Blob.count();
  feature_size_ = in_feature_Blob.channels();
  out_height_ = out_feature_Blob.height();
  out_width_ = out_feature_Blob.width();
  CHECK_EQ(num_, in_feature_Blob.num());
  CHECK_EQ(num_, out_feature_Blob.num());
  CHECK_EQ(in_height_, in_feature_Blob.height());
  CHECK_EQ(in_width_, in_feature_Blob.width());
  CHECK_EQ(feature_size_, out_feature_Blob.channels());

  // the order is important otherwise the layers are not in the right shape
  pdist_layer_->Reshape(tmp_bottom_pdist_, tmp_top_pdist_);
  scalar_layer_->Reshape(tmp_bottom_scalar_, tmp_top_scalar_);
  softmax_layer_->Reshape(tmp_bottom_softmax_, tmp_top_softmax_);
  matmul_layer_->Reshape(tmp_bottom_matmul_, tmp_top_matmul_);

  // n x shape 
  vector<int> pdist_full = bb_top_pdist_.shape();
  pdist_full[0] = num_;
  //bb_pdist_full_.Reshape(pdist_full);
  // n x k x shape 
  pdist_full[1] = scales_;
  bb_scalar_full_.Reshape(pdist_full);
  bb_softmax_full_.Reshape(pdist_full);

  top[0]->Reshape(num_, scales_ * channels_, out_height_, out_width_);
}

template <typename Dtype>
void BilateralBruteForceLayer<Dtype>::Forward_cpu(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {


  for(int n = 0; n < num_;++n ){
    // we iterate and the temporary data has to be filled for each example
    caffe_copy(bb_bottom_data_.count(),
               bottom[0]->cpu_data()+bottom[0]->offset(n),
               bb_bottom_data_.mutable_cpu_data());
    caffe_copy(bb_bottom_f1_.count(),
               bottom[1]->cpu_data()+bottom[1]->offset(n),
               bb_bottom_f1_.mutable_cpu_data());
    caffe_copy(bb_bottom_f2_.count(),
               bottom[2]->cpu_data()+bottom[2]->offset(n),
               bb_bottom_f2_.mutable_cpu_data());

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
                 bb_top_scalar_.cpu_data(),
                 bb_scalar_full_.mutable_cpu_data()+
                 bb_scalar_full_.offset(n,s));

      // we need to store the output for the backward pass
      // still more efficient since we half the memory size
      softmax_layer_->Forward(tmp_bottom_softmax_, tmp_top_softmax_);
      caffe_copy(bb_top_softmax_.count(),
                 bb_top_softmax_.cpu_data(),
                 bb_softmax_full_.mutable_cpu_data()+
                 bb_softmax_full_.offset(n,s));

      matmul_layer_->Forward(tmp_bottom_matmul_, tmp_top_matmul_);

      caffe_copy(bb_top_matmul_.count(),
                 bb_top_matmul_.cpu_data(),
                 top[0]->mutable_cpu_data()+top[0]->offset(
                     n,s*channels_));
    }
  }
}


template <typename Dtype>
void BilateralBruteForceLayer<Dtype>::Backward_cpu(
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
    caffe_set(bottom[0]->count(), Dtype(0), bottom[0]->mutable_cpu_diff());
  }
  if(propagate_down[1]){
    // not supported
    //caffe_set(bottom[1]->count(), Dtype(0), bottom[1]->mutable_cpu_diff());
  }
  if(propagate_down[2]){
    // not supported
    //caffe_set(bottom[2]->count(), Dtype(0), bottom[2]->mutable_cpu_diff());
  }
  if(propagate_down[3]){
    caffe_set(bottom[3]->count(), Dtype(0), bottom[3]->mutable_cpu_diff());
  }

  for(int n = 0; n < num_;++n ){

    // copy the botton data back
    caffe_copy(bb_bottom_data_.count(),
               bottom[0]->cpu_data()+bottom[0]->offset(n),
               bb_bottom_data_.mutable_cpu_data());
    // scalar is not inplace
    // if(propagate_down[3]){
    //   caffe_copy(bb_top_pdist_.count(),
    //              bb_pdist_full_.cpu_data()+bb_pdist_full_.offset(n),
    //              bb_top_pdist_.mutable_cpu_data());
    // }

    for (int s = 0 ; s < scales_ ; ++s) {

      // set the diff for the chain 
      caffe_copy(bb_top_matmul_.count(),
                 top[0]->cpu_diff()+top[0]->offset(
                     n,s*channels_),
                 bb_top_matmul_.mutable_cpu_diff());

      // this is the bottom layer of matmult we need that to backprop
      caffe_copy(bb_top_softmax_.count(),
                 bb_softmax_full_.cpu_data()+
                 bb_softmax_full_.offset(n,s),
                 bb_top_softmax_.mutable_cpu_data());

      matmul_layer_->Backward(tmp_top_matmul_,
                              pd_matmul,
                              tmp_bottom_matmul_);

      // if used we backprop this is an addition
      if(propagate_down[0]){
        caffe_axpy(bb_bottom_data_.count(),
                   Dtype(1),
                   bb_bottom_data_.cpu_diff(),
                   bottom[0]->mutable_cpu_diff()+bottom[0]->offset(n));
      }
      if(propagate_down[3]){

        // this is the bottom layer of softmax we need that to backprop
        caffe_copy(bb_top_scalar_.count(),
                   bb_scalar_full_.cpu_data()+
                   bb_scalar_full_.offset(n,s),
                   bb_top_scalar_.mutable_cpu_data());

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
        // caffe_axpy(bb_bottom_f1_.count(),
        //            Dtype(1),
        //            bb_bottom_f1_.cpu_diff(),
        //            bottom[1]->mutable_cpu_diff()+bottom[1]->offset(n));
      }
      if(propagate_down[2]){
        // not implemented in pdist 
        // caffe_axpy(bb_bottom_f2_.count(),
        //            Dtype(1),
        //            bb_bottom_f2_.cpu_diff(),
        //            bottom[2]->mutable_cpu_diff()+bottom[2]->offset(n));
      }
    }
  }
  // std::cout << "bottom 0 grad"  << std::endl;
  // for(int i = 0; i < bottom[0]->count(); ++i) {
  //   std::cout << " "  << bottom[0]->cpu_diff()[i];
  // }
  // std::cout << std::endl;

  // std::cout << "bottom 3 grad"  << std::endl;
  // for(int i = 0; i < bottom[3]->count(); ++i) {
  //   std::cout << " "  << bottom[3]->cpu_diff()[i];
  // }
  // std::cout << std::endl;
}

#ifdef CPU_ONLY
STUB_GPU(BilateralBruteForceLayer);
#endif

INSTANTIATE_CLASS(BilateralBruteForceLayer);
REGISTER_LAYER_CLASS(BilateralBruteForce);

}  // namespace caffe
