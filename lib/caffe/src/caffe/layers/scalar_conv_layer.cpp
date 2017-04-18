#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/malabar_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

  template <typename Dtype>
  void ScalarConvLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {

    CHECK_NE(bottom[0], top[0]) << "ScalarConvLayer cannot be used in-place";

    axis_ = bottom[0]->CanonicalAxisIndex(this->layer_param_.scalar_conv_param().axis());

    outer_dim_ = bottom[0]->count(0, axis_);
    scalar_dim_ = bottom[0]->shape(axis_);
    inner_dim_ = bottom[0]->count(axis_ + 1);

    // Check if we need to set up the weights
    if (this->blobs_.size() > 0) {
      LOG(INFO) << "Skipping parameter initialization";
    } else {
      // Intialize the weight
      this->blobs_.resize(1);
      vector<int> weight_shape(1);
      weight_shape[0] = scalar_dim_;
      this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
      // fill the weights
      shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
          this->layer_param_.scalar_conv_param().weight_filler()));
      weight_filler->Fill(this->blobs_[0].get());
    }  // parameter initialization
    this->param_propagate_down_.resize(this->blobs_.size(), true);
  }

template <typename Dtype>
void ScalarConvLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // TODO: make ScalarConvLayer usable in-place.
  // Currently, in-place computation is broken during Backward with
  // propagate_down[0] && propagate_down[1], as bottom[0]'s diff is used for
  // temporary storage of an intermediate result, overwriting top[0]'s diff
  // if using in-place computation.

  top[0]->ReshapeLike(*bottom[0]);
  sum_result_.Reshape(vector<int>(1, outer_dim_ * scalar_dim_));
  const int sum_mult_size = std::max(outer_dim_, inner_dim_);
  sum_multiplier_.Reshape(vector<int>(1, sum_mult_size));
  if (sum_multiplier_.cpu_data()[sum_mult_size - 1] != Dtype(1)) {
    caffe_set(sum_mult_size, Dtype(1), sum_multiplier_.mutable_cpu_data());
  }
}

template <typename Dtype>
void ScalarConvLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* scalar_data = this->blobs_[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  for (int n = 0; n < outer_dim_; ++n) {
    for (int d = 0; d < scalar_dim_; ++d) {
      const Dtype factor = scalar_data[d];
      caffe_cpu_scale(inner_dim_, factor, bottom_data, top_data);
      bottom_data += inner_dim_;
      top_data += inner_dim_;
    }
  }
}

template <typename Dtype>
void ScalarConvLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    // Hack: store big eltwise product in bottom[0] diff, except in the special
    // case where this layer itself does the eltwise product, in which case we
    // can store it directly in the scalar diff, and we're done.
    const bool is_eltwise = (inner_dim_ == 1 && outer_dim_ == 1);
    Dtype* product = is_eltwise ?
        this->blobs_[0]->mutable_cpu_diff() : bottom[0]->mutable_cpu_diff();
    caffe_mul(top[0]->count(), top_diff, bottom_data, product);
    if (!is_eltwise) {
      Dtype* sum_result = NULL;
      if (inner_dim_ == 1) {
        sum_result = product;
      } else if (sum_result_.count() == 1) {
        const Dtype* sum_mult = sum_multiplier_.cpu_data();
        Dtype* scalar_diff = this->blobs_[0]->mutable_cpu_diff();
        *scalar_diff = caffe_cpu_dot(inner_dim_, product, sum_mult);
      } else {
        const Dtype* sum_mult = sum_multiplier_.cpu_data();
        sum_result = (outer_dim_ == 1) ?
            this->blobs_[0]->mutable_cpu_diff() : sum_result_.mutable_cpu_data();
        caffe_cpu_gemv(CblasNoTrans, sum_result_.count(), inner_dim_,
                       Dtype(1), product, sum_mult, Dtype(0), sum_result);
      }
      if (outer_dim_ != 1) {
        const Dtype* sum_mult = sum_multiplier_.cpu_data();
        Dtype* scalar_diff = this->blobs_[0]->mutable_cpu_diff();
        if (scalar_dim_ == 1) {
          *scalar_diff = caffe_cpu_dot(outer_dim_, sum_mult, sum_result);
        } else {
          caffe_cpu_gemv(CblasTrans, outer_dim_, scalar_dim_,
                         Dtype(1), sum_result, sum_mult, Dtype(0), scalar_diff);
        }
      }
    }
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* scalar_data = this->blobs_[0]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    for (int n = 0; n < outer_dim_; ++n) {
      for (int d = 0; d < scalar_dim_; ++d) {
        const Dtype factor = scalar_data[d];
        caffe_cpu_scale(inner_dim_, factor, top_diff, bottom_diff);
        bottom_diff += inner_dim_;
        top_diff += inner_dim_;
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(ScalarConvLayer);
#endif

INSTANTIATE_CLASS(ScalarConvLayer);
REGISTER_LAYER_CLASS(ScalarConv);

}  // namespace caffe
