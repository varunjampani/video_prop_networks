#include "caffe/util/math_functions.hpp"
#include "caffe/malabar_layers.hpp"


namespace caffe {

/*
Setup function
*/
template <typename Dtype>
void TransposeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();

  top[0]->Reshape(num_, channels_, width_, height_);
}

template <typename Dtype>
void TransposeLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
    top[0]->Reshape(num_, channels_, width_, height_);
}

/*
Forward CPU function
*/
template <typename Dtype>
void TransposeLayer<Dtype>::Forward_cpu(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();

    for (int i = 0; i < num_; ++i) {
      for (int j = 0; j < channels_; j++) {
        for (int k = 0; k < height_; k++) {
          for (int l = 0; l < width_; l++) {
            caffe_copy(1, bottom_data + bottom[0]->offset(i, j, k, l),
              top_data + top[0]->offset(i, j, l, k));
          }
        }
      }
    }
}

/*
Backward CPU function
 */
template <typename Dtype>
void TransposeLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* top_diff = top[0]->cpu_diff();

    for (int i = 0; i < num_; ++i) {
      for (int j = 0; j < channels_; j++) {
        for (int k = 0; k < height_; k++) {
          for (int l = 0; l < width_; l++) {
            bottom_diff[bottom[0]->offset(i, j, k, l)] =
              top_diff[top[0]->offset(i, j, l, k)];
          }
        }
      }
    }
}

#ifdef CPU_ONLY
STUB_GPU(TransposeLayer);
#endif

INSTANTIATE_CLASS(TransposeLayer);
REGISTER_LAYER_CLASS(Transpose);

}  // namespace caffe
