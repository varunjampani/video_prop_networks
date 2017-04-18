#include <algorithm>
#include <vector>

#include "caffe/malabar_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void Scalar3Layer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape(bottom[0]->shape());

}

template <typename Dtype>
void Scalar3Layer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int count = bottom[0]->count();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* scale_data = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_cpu_scale(count, scale_data[0], bottom_data, top_data);
}

template <typename Dtype>
void Scalar3Layer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    int count = top[0]->count();
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* scalar_data = bottom[1]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_cpu_scale(count, scalar_data[0], top_diff, bottom_diff);
  }

  if (propagate_down[1]) {
    Dtype* top_diff = top[0]->mutable_cpu_diff();
    const Dtype* top_data = top[0]->cpu_data();
    Dtype* bottom_diff = bottom[1]->mutable_cpu_diff();

    bottom_diff[0] = caffe_cpu_dot(bottom[0]->count(),
                                   top_diff,
                                   top_data);

    bottom_diff[0] /= bottom[1]->cpu_data()[0];
  }
}

#ifdef CPU_ONLY
STUB_GPU(Scalar3Layer);
#endif

INSTANTIATE_CLASS(Scalar3Layer);
REGISTER_LAYER_CLASS(Scalar3);

}  // namespace caffe
