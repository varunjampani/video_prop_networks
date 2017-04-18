#include "caffe/util/math_functions.hpp"
#include "caffe/malabar_layers.hpp"


namespace caffe {

/*
Setup function
*/
template <typename Dtype>
void SmearLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  SmearParameter smear_param = this->layer_param_.smear_param();

  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();

  out_height_ = bottom[1]->height();
  out_width_ = bottom[1]->width();

  outer_num_ = num_;
  inner_num_ = out_height_ * out_width_;

  ignore_idx_value_ = smear_param.ignore_idx_value();
  ignore_feature_value_ = smear_param.ignore_feature_value();

  CHECK_EQ(bottom[1]->num(), num_)
    << "Blob dim-0 (num) should be same for bottom blobs.";

  CHECK_EQ(bottom[1]->channels(), 1)
    << "ID blob (bottom-2) has more than one channel.";

  top[0]->Reshape(num_, channels_, out_height_, out_width_);
}

template <typename Dtype>
void SmearLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
    top[0]->Reshape(num_, channels_, out_height_, out_width_);
}

/*
Forward CPU function
*/
template <typename Dtype>
void SmearLayer<Dtype>::Forward_cpu(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();

  for (int i = 0; i < num_; ++i) {
    for (int j = 0; j < channels_; j++) {
      for (int k = 0; k < out_height_; k++) {
        for (int l = 0; l < out_width_; l++) {
          const int reverse_idx =
            static_cast<int>(bottom[1]->data_at(i, 0, k, l));
          if (reverse_idx == ignore_idx_value_) {
            caffe_copy(1, &ignore_feature_value_,
              top_data + top[0]->offset(i, j, k, l));
          } else {
            const int bottom_offset = bottom[0]->offset(i, j) + reverse_idx;
            caffe_copy(1, bottom_data + bottom_offset,
              top_data + top[0]->offset(i, j, k, l));
          }
        }
      }
    }
  }
}

/*
Backward CPU function
 */
template <typename Dtype>
void SmearLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
    if (propagate_down[1]) {
      LOG(FATAL) << this->type()
                 << " Layer cannot backpropagate to spixel index inputs.";
    }
    if (propagate_down[0]) {
      caffe_set(bottom[0]->count(), (Dtype)0., bottom[0]->mutable_cpu_diff());

      Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
      const Dtype* top_diff = top[0]->cpu_diff();

      for (int i = 0; i < num_; ++i) {
        for (int j = 0; j < channels_; j++) {
          for (int k = 0; k < out_height_; k++) {
            for (int l = 0; l < out_width_; l++) {
              const int reverse_idx =
                static_cast<int>(bottom[1]->data_at(i, 0, k, l));
              if (reverse_idx != ignore_idx_value_) {
                const int bottom_offset = bottom[0]->offset(i, j) + reverse_idx;
                bottom_diff[bottom_offset] += top_diff[top[0]->offset(i, j, k, l)];
              }
            }
          }
        }
      }
    }
}

#ifdef CPU_ONLY
STUB_GPU(SmearLayer);
#endif

INSTANTIATE_CLASS(SmearLayer);
REGISTER_LAYER_CLASS(Smear);

}  // namespace caffe
