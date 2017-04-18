#include <cfloat>
#include <vector>

#include "caffe/malabar_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void Scalar3Layer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int count = bottom[0]->count();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* scale_data = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  caffe_gpu_scale(count, scale_data[0], bottom_data, top_data);
}

template <typename Dtype>
void Scalar3Layer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  if (propagate_down[0]) {
    int count = top[0]->count();
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* scalar_data = bottom[1]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    caffe_gpu_scale(count, scalar_data[0], top_diff, bottom_diff);
  }
  if (propagate_down[1]) {
    Dtype* top_diff = top[0]->mutable_gpu_diff();
    const Dtype* top_data = top[0]->gpu_data();
    Dtype* bottom_diff = bottom[1]->mutable_cpu_diff();

    Dtype tmp_value;
    caffe_gpu_dot(bottom[0]->count(),
                  top_diff,
                  top_data,
                  &tmp_value);
    bottom_diff[0] = tmp_value/bottom[1]->cpu_data()[0] ;

  }
}

INSTANTIATE_LAYER_GPU_FUNCS(Scalar3Layer);

}  // namespace caffe
