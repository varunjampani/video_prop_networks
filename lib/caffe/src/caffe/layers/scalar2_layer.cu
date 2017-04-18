#include <cfloat>
#include <vector>

#include "caffe/malabar_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void Scalar2Forward(const int max_index,
                               const int bottom_row_size,
                               const int top_row_size,
                               const int scalar_size,
                               const Dtype* in,
                               const Dtype* scalars,
                               Dtype* out) {
   CUDA_KERNEL_LOOP(index, max_index) {
     const int row_in_index = index / top_row_size; 
     const int col_in_index = index % bottom_row_size; 
     const int in_index = row_in_index * bottom_row_size + col_in_index;
     const int col_scalar_index = index % top_row_size; 
     const int scalar_index = col_scalar_index / bottom_row_size;
     out[index] = in[in_index] * scalars[scalar_index];
   }
}

template <typename Dtype>
void Scalar2Layer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
   const int count = top[0]->count();
   const Dtype* bottom_data = bottom[0]->gpu_data();
   const Dtype* scalar_data = bottom[1]->gpu_data();
   Dtype* top_data = top[0]->mutable_gpu_data();
   Scalar2Forward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
     <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
         count*bottom[1]->count(),
         bottom[0]->count(1),
         top[0]->count(1),
         bottom[1]->count(),
         bottom_data,
         scalar_data,
         top_data);
}

template <typename Dtype>
void Scalar2Layer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* scalar_data = bottom[1]->cpu_data(); // so I can de-reference
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    caffe_gpu_set(bottom[0]->count(), Dtype(0), bottom_diff);
    for (int n = 0; n < bottom[0]->shape(0); ++n) {
        for (int d = 0; d < bottom[1]->count(0); ++d) {
          caffe_gpu_axpy(bottom[0]->count(1),
                         scalar_data[d],
                         top_diff + top[0]->offset(n, d * bottom[0]->shape(1)),
                         bottom_diff + bottom[0]->offset(n));
        }
      }
  }
  if (propagate_down[1]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* top_data = top[0]->gpu_data();
    tmp_diff_.Reshape(top[0]->shape());

    // this is on the cpu much easier and it should only be in the
    // order of 10 numbers anyway
    // this will set the bottom diff to 0
    // important otherwise we have random values
    Dtype* bottom_diff = bottom[1]->mutable_cpu_diff();
    caffe_set(bottom[1]->count(), Dtype(0), bottom_diff);

    vector<int> tmp_sum_shape(1, bottom[0]->count(1));
    tmp_sum_.Reshape(tmp_sum_shape);
    caffe_gpu_set(bottom[0]->count(1), Dtype(1), tmp_sum_.mutable_gpu_data());

    // // ideally we would use the bottom[0] data for the right
    // // part of the top data, however, that would require index things
    // // this is much faster and we now the relationship is just the
    // // additional scalar
    caffe_gpu_mul(top[0]->count(),
                  top_diff,
                  top_data,
                  tmp_diff_.mutable_gpu_data());

    Dtype tmp_value;
    for (int n = 0; n < bottom[0]->shape(0); ++n) {

      for (int d = 0; d < bottom[1]->count(0); ++d) {
        // we sum all the diffs and divide by the current scalar
        // this has to be done since we use the top data instead
        // of the bottom data for the multiplication
        // it is easier though and faster
        caffe_gpu_dot(
            bottom[0]->count(1),
            tmp_diff_.mutable_gpu_data()+top[0]->offset(
                n, d*bottom[0]->shape(1)),
            tmp_sum_.gpu_data(),
            &tmp_value);// this fucking return value has to be on the host memory

        // this operation is easier on the host
        bottom_diff[bottom[1]->offset(d)] += tmp_value / bottom[1]->cpu_data()[d];
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(Scalar2Layer);

}  // namespace caffe
