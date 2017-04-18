#include <vector>

#include "caffe/filler.hpp"
#include "caffe/malabar_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void Scale2Layer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int count = bottom[0]->count();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* scale_data = this->blobs_[0]->gpu_data();
  Dtype scale_values[1];
  cudaMemcpy(scale_values, scale_data, sizeof(Dtype),cudaMemcpyDeviceToHost);
  caffe_gpu_scale(count, scale_values[0], bottom_data, top_data);
}

template <typename Dtype>
void Scale2Layer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  // const Dtype* top_data = top[0]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();

  const int count = bottom[0]->count();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

  const Dtype* scale_data = this->blobs_[0]->gpu_data();
  Dtype* scale_diff = this->blobs_[0]->mutable_gpu_diff();

  Dtype scale_values[1];
  cudaMemcpy(scale_values, scale_data, sizeof(Dtype),cudaMemcpyDeviceToHost);

  if (this->param_propagate_down_[0]) {
    // caffe_gpu_mul(count, top_data, top_diff, bottom_diff);
    Dtype scale_diff_value;
    caffe_gpu_dot(count, top_diff, bottom_data, &scale_diff_value);
    cudaMemcpy(scale_diff, &scale_diff_value, sizeof(Dtype),cudaMemcpyHostToDevice);
  }

  if (!propagate_down[0]) { return; }
  const Dtype ct = 1.0;
  caffe_gpu_scale(count, ct, top_diff, bottom_diff);
  caffe_gpu_scal(count, scale_values[0], bottom_diff);
}

INSTANTIATE_LAYER_GPU_FUNCS(Scale2Layer);

}  // namespace caffe
