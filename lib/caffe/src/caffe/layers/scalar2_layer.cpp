#include <algorithm>
#include <vector>

#include "caffe/malabar_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void Scalar2Layer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // TODO: make Scalar2Layer usable in-place.
  // Currently, in-place computation is broken during Backward with
  // propagate_down[0] && propagate_down[1], as bottom[0]'s diff is used for
  // temporary storage of an intermediate result, overwriting top[0]'s diff
  // if using in-place computation.
  CHECK_NE(bottom[0], top[0]) << "Scalar2Layer cannot be used in-place";

  vector<int> top_0_shape = bottom[0]->shape();
  top_0_shape[1] *= bottom[1]->count();
  // top blob shape bottom_num, bottom_channel* scalar_dim, bottom_remaining
  top[0]->Reshape(top_0_shape);

}

template <typename Dtype>
void Scalar2Layer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* scalar_data = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_set(top[0]->count(), Dtype(0), top_data);
  for (int n = 0; n < bottom[0]->shape(0); ++n) {
    for (int d = 0; d < bottom[1]->count(0); ++d) {
      caffe_axpy(bottom[0]->count(1),
                 scalar_data[d],
                 bottom_data + bottom[0]->offset(n),
                 top_data+top[0]->offset(n, d*bottom[0]->shape(1)));
    }
  }
}

template <typename Dtype>
void Scalar2Layer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* scalar_data = bottom[1]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
    for (int n = 0; n < bottom[0]->shape(0); ++n) {
      for (int d = 0; d < bottom[1]->count(0); ++d) {
        caffe_axpy(bottom[0]->count(1),
                   scalar_data[d],
                   top_diff+top[0]->offset(n, d*bottom[0]->shape(1)),
                   bottom_diff+bottom[0]->offset(n));
      }
    }
  }

  if (propagate_down[1]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* top_data = top[0]->cpu_data();
    tmp_diff_.Reshape(top[0]->shape());

    vector<int> tmp_sum_shape(1, bottom[0]->count(1));
    tmp_sum_.Reshape(tmp_sum_shape);
    caffe_set(bottom[0]->count(1), Dtype(1), tmp_sum_.mutable_cpu_data());

    // this will set the bottom diff to 0
    // important otherwise we have random values
    Dtype* bottom_diff = bottom[1]->mutable_cpu_diff();
    caffe_set(bottom[1]->count(), Dtype(0), bottom_diff);

    // ideally we would use the bottom[0] data for the right
    // part of the top data, however, that would require index things
    // this is much faster and we now the relationship is just the
    // additional scalar
    caffe_mul(top[0]->count(),top_diff,top_data,tmp_diff_.mutable_cpu_data());

    for (int n = 0; n < bottom[0]->shape(0); ++n) {
      for (int d = 0; d < bottom[1]->count(0); ++d) {
        // we sum all the diffs and divide by the current scalar
        // this has to be done since we use the top data instead
        // of the bottom data for the multiplication
        // it is easier though and faster
        bottom_diff[bottom[1]->offset(d)] += caffe_cpu_dot(
            bottom[0]->count(1),
            tmp_diff_.mutable_cpu_data()+top[0]->offset(
                n, d*bottom[0]->shape(1)),
            tmp_sum_.cpu_data())/bottom[1]->cpu_data()[d];
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(Scalar2Layer);
#endif

INSTANTIATE_CLASS(Scalar2Layer);
REGISTER_LAYER_CLASS(Scalar2);

}  // namespace caffe
