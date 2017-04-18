#include "caffe/util/math_functions.hpp"
#include "caffe/malabar_layers.hpp"

#include <cmath>


namespace caffe {

  // This layer computes pairwise eucledian squared distance between
  // two given bottom blobs.
  //
  // bottom[0] is of size NxCxH0xW0
  // bottom[1] is of size NxCxH1xW1
  // top[0] = pairwise(bottom[1], bottom[0]) is of size Nx1xH1W1xH0W0

/*
Setup function
*/
template <typename Dtype>
void PdistLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();

  out_height_ = bottom[1]->height() * bottom[1]->width();
  out_width_ = bottom[0]->height() * bottom[0]->width();

  ignore_value_ = this->layer_param_.pdist_param().ignore_value();
  scale_value_ = this->layer_param_.pdist_param().scale_value();

  CHECK_EQ(bottom[1]->num(), num_)
    << "dim-0 (num) should be same for bottom blobs.";

  CHECK_EQ(bottom[1]->channels(), channels_)
    << "dim-1 (channels) should be same for bottom blobs.";

  top[0]->Reshape(num_, 1, out_height_, out_width_);
}

template <typename Dtype>
void PdistLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape(num_, 1, out_height_, out_width_);
}

// ((n * channels() + c) * height() + h) * width() + w;

/*
Forward CPU function
*/
template <typename Dtype>
void PdistLayer<Dtype>::Forward_cpu(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_data_0 = bottom[0]->cpu_data();
  const Dtype* bottom_data_1 = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();

  for (int i = 0; i < num_; i++) {
    for (int j = 0; j < out_height_; j++) {
      for (int k = 0; k < out_width_; k++) {
        Dtype sq_dist = 0;
        for (int c = 0; c < channels_; c++) {
          Dtype bottom_1_value = bottom_data_1[(i * channels_ + c) * out_height_ + j];
          Dtype bottom_0_value = bottom_data_0[(i * channels_ + c) * out_width_ + k];
          if (bottom_1_value == ignore_value_ || bottom_0_value == ignore_value_) {
            sq_dist = 1e10;
            break;
          } else {
            sq_dist += pow(bottom_1_value - bottom_0_value, 2);
          }
        }
        top_data[top[0]->offset(i, 0, j, k)] = scale_value_ * sq_dist;
      }
    }
  }
}

/*
Backward CPU function
 */
template <typename Dtype>
void PdistLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

    if (propagate_down[0] || propagate_down[1]) {

      caffe_set(bottom[0]->count(), (Dtype)0., bottom[0]->mutable_cpu_diff());
      caffe_set(bottom[1]->count(), (Dtype)0., bottom[1]->mutable_cpu_diff());

      Dtype* bottom_diff_0 = bottom[0]->mutable_cpu_diff();
      Dtype* bottom_diff_1 = bottom[1]->mutable_cpu_diff();
      const Dtype* top_diff = top[0]->cpu_diff();

      const Dtype* bottom_data_0 = bottom[0]->cpu_data();
      const Dtype* bottom_data_1 = bottom[1]->cpu_data();
      const Dtype* top_data = top[0]->cpu_data();

      for (int i = 0; i < num_; i++) {
        for (int j = 0; j < out_height_; j++) {
          for (int k = 0; k < out_width_; k++) {
            if (top_data[top[0]->offset(i, 0, j, k)] != scale_value_ * 1e10) {
              for (int c = 0; c < channels_; c++) {
                int bottom_1_offset = (i * channels_ + c) * out_height_ + j;
                int bottom_0_offset = (i * channels_ + c) * out_width_ + k;
                int top_offset = top[0]->offset(i, 0, j, k);
                Dtype bottom_1_value = bottom_data_1[bottom_1_offset];
                Dtype bottom_0_value = bottom_data_0[bottom_0_offset];
                if (bottom_1_value == ignore_value_) {
                  bottom_diff_1[bottom_1_offset] = 0;
                } else {
                  bottom_diff_1[bottom_1_offset] += top_diff[top_offset] *
                    2 * scale_value_ * (bottom_1_value - bottom_0_value);
                }
                if (bottom_0_value == ignore_value_) {
                  bottom_diff_0[bottom_0_offset] = 0;
                } else {
                  bottom_diff_0[bottom_0_offset] += top_diff[top_offset] *
                    2 * scale_value_ * (bottom_0_value - bottom_1_value);
                }
              }
            }
          }
        }
      }
    }
}

#ifdef CPU_ONLY
STUB_GPU(PdistLayer);
#endif

INSTANTIATE_CLASS(PdistLayer);
REGISTER_LAYER_CLASS(Pdist);

}  // namespace caffe
