#include <vector>

#include "caffe/filler.hpp"
#include "caffe/malabar_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void Scale2Layer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);

  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    // Intialize the weight
    this->blobs_.resize(1);
    vector<int> weight_shape(1);
    weight_shape[0] = 1;
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.scale2_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void Scale2Layer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int count = bottom[0]->count();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* scale_data = this->blobs_[0]->cpu_data();
  caffe_cpu_scale(count, scale_data[0], bottom_data, top_data);
}

template <typename Dtype>
void Scale2Layer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  // const Dtype* top_data = top[0]->cpu_data();
  const Dtype* top_diff = top[0]->cpu_diff();

  const int count = bottom[0]->count();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

  const Dtype* scale_data = this->blobs_[0]->cpu_data();
  Dtype* scale_diff = this->blobs_[0]->mutable_cpu_diff();

  if (this->param_propagate_down_[0]) {
    // caffe_mul(count, top_data, top_diff, bottom_diff);
    *scale_diff = caffe_cpu_dot(count, top_diff, bottom_data);
  }

  if (!propagate_down[0]) { return; }
  caffe_copy(count, top_diff, bottom_diff);
  caffe_scal(count, scale_data[0], bottom_diff);
}

#ifdef CPU_ONLY
STUB_GPU(Scale2Layer);
#endif

INSTANTIATE_CLASS(Scale2Layer);
REGISTER_LAYER_CLASS(Scale2);

}  // namespace caffe
