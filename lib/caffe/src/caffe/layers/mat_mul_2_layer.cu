#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/malabar_layers.hpp"

namespace caffe {

template <typename Dtype>
void MatMul2Layer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data_0 = bottom[0]->gpu_data();  // blob
  const Dtype* bottom_data_1 = bottom[1]->gpu_data();  // matrix
  Dtype* top_data = top[0]->mutable_gpu_data();

  vector<int> bottom_0_shape = bottom[0]->shape();
  vector<int> bottom_1_shape = bottom[1]->shape();

  const int num = bottom_0_shape[0];
  const int m_ = bottom_0_shape[1] * bottom_0_shape[2];
  const int k_ = bottom_0_shape[3];
  const int n_ = bottom_1_shape[2];

  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < num_kernels_; ++j) {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, m_, n_, k_, (Dtype)1.,
          bottom_data_0 + bottom[0]->offset(i),
          bottom_data_1 + bottom[1]->offset(i, j),
          (Dtype)0., top_data + top[0]->offset(i, channels_ * j));
    }
  }
}

template <typename Dtype>
void MatMul2Layer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {

    const Dtype* top_diff = top[0]->gpu_diff();

    vector<int> top_0_shape = top[0]->shape();
    vector<int> bottom_0_shape = bottom[0]->shape();
    vector<int> bottom_1_shape = bottom[1]->shape();

    const int num = bottom_0_shape[0];
    // const int chan = bottom_0_shape[1];

    // this should NOT be done since it would destroy gradients
    // of other layer, zeroing the diff is done by the solver
    caffe_gpu_set(bottom[0]->count(), (Dtype)0., bottom[0]->mutable_gpu_diff());

    for (int i = 0; i < num; ++i) {
      for (int j = 0; j < num_kernels_; ++j) {
        caffe_gpu_gemm<Dtype>(CblasNoTrans,
                              CblasNoTrans,
                              top[0]->shape()[2] * bottom[0]->shape()[1],
                              bottom[0]->shape()[3],
                              top[0]->shape()[3],
                              (Dtype)1.,
                              top_diff + top[0]->offset(i, channels_ * j),
                              bottom[1]->gpu_data() + bottom[1]->offset(i, j),
                              (Dtype)1.,
                              bottom[0]->mutable_gpu_diff() +
                              bottom[0]->offset(i));
      }
    }
  }
  if (propagate_down[1]) {
    // THIS HAS TO BE CHECKED, SOMEHOW IT SEEMS WEIRD THAT WE SET THIS TO
    // ZERO, WE MIGHT OVERWRITE ALL THE OTHER DIFFS
    // IT IS HOWEVER REQUIRED TO PASS THE GRADIENT TESTS
    // this should NOT be done since it would destroy gradients
    // of other layer, zeroing the diff is done by the solver
    caffe_gpu_set(bottom[1]->count(), (Dtype)0., bottom[1]->mutable_gpu_diff());
    const Dtype* top_diff = top[0]->gpu_diff();

    vector<int> top_0_shape = top[0]->shape();
    vector<int> bottom_0_shape = bottom[0]->shape();
    vector<int> bottom_1_shape = bottom[1]->shape();

    const int num = bottom_0_shape[0];

    for (int i = 0; i < num; ++i) {
      for (int j = 0; j < num_kernels_; ++j) {
        caffe_gpu_gemm<Dtype>(CblasTrans,
                              CblasNoTrans,
                              top[0]->shape()[3],
                              bottom[0]->shape()[3],
                              bottom[0]->shape()[2]*bottom[0]->shape()[1],
                              (Dtype)1.,
                              top_diff + top[0]->offset(i, channels_ * j),
                              bottom[0]->gpu_data() + bottom[0]->offset(i),
                              (Dtype)1.,
                              bottom[1]->mutable_gpu_diff() +
                              bottom[1]->offset(i, j));
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(MatMul2Layer);

}  // namespace caffe
