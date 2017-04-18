#include "caffe/util/math_functions.hpp"
#include "caffe/malabar_layers.hpp"
#include "caffe/util/gpu_util.cuh"

namespace caffe {

template <typename Dtype>
__global__ void SmearForwardGPU(const int nthreads, const Dtype* bottom_data,
  const Dtype* index_data, const int num, const int dim, const int spatial_dim,
  const Dtype ignore_idx_value, const Dtype ignore_feature_value,
  const int data_spatial_dim, Dtype* top_data) {
    CUDA_KERNEL_LOOP(index, nthreads) {
      const int n = index / spatial_dim;
      const int s = index % spatial_dim;
      const int reverse_idx =
        static_cast<int>(index_data[n * spatial_dim + s]);

      for (int k = 0; k < dim; k++) {
        int top_offset = ((n * dim + k) * spatial_dim + s);
        int bottom_offset = ((n * dim + k) * data_spatial_dim + reverse_idx);
        if (reverse_idx == ignore_idx_value) {
          top_data[top_offset] = ignore_feature_value;
        } else {
          top_data[top_offset] = bottom_data[bottom_offset];
        }
      }
    }
}

template <typename Dtype>
__global__ void SmearBackwardGPU(const int nthreads, const Dtype* top_diff,
  const Dtype* index_data, const int num, const int dim, const int spatial_dim,
  const Dtype ignore_idx_value,
  const int data_spatial_dim, Dtype* bottom_diff) {
    CUDA_KERNEL_LOOP(index, nthreads) {
      const int n = index / spatial_dim;
      const int s = index % spatial_dim;
      const int reverse_idx =
        static_cast<int>(index_data[n * spatial_dim + s]);

      if (reverse_idx != ignore_idx_value) {
        for (int k = 0; k < dim; k++) {
          const int top_offset = ((n * dim + k) * spatial_dim + s);
          const int bottom_offset = ((n * dim + k) * data_spatial_dim + reverse_idx);

          caffe_gpu_atomic_add((Dtype) top_diff[top_offset],
            bottom_diff + bottom_offset);

        }
      }
    }
}

/*
Forward CPU function
*/
template <typename Dtype>
void SmearLayer<Dtype>::Forward_gpu(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* index_data = bottom[1]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();

    const int nthreads = outer_num_ * inner_num_;
    const int data_spatial_dim = bottom[0]->height() * bottom[0]->width();

    // NOLINT_NEXT_LINE(whitespace/operators)
    SmearForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, bottom_data, index_data,
                                outer_num_, channels_, inner_num_,
                                ignore_idx_value_, ignore_feature_value_,
                                data_spatial_dim, top_data);
}

/*
Backward GPU function
 */
template <typename Dtype>
void SmearLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
    if (propagate_down[1]) {
      LOG(FATAL) << this->type()
                 << " Layer cannot backpropagate to spixel index inputs.";
    }
    if (propagate_down[0]) {
      caffe_gpu_set(bottom[0]->count(), (Dtype)0.,
        bottom[0]->mutable_gpu_diff());
      Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
      const Dtype* top_diff = top[0]->gpu_diff();
      const Dtype* index_data = bottom[1]->gpu_data();

      const int nthreads = outer_num_ * inner_num_;
      const int data_spatial_dim = bottom[0]->height() * bottom[0]->width();

      // NOLINT_NEXT_LINE(whitespace/operators)
      SmearBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, top_diff, index_data,
                                  outer_num_, channels_, inner_num_,
                                  ignore_idx_value_,
                                  data_spatial_dim, bottom_diff);

    }
}

INSTANTIATE_LAYER_GPU_FUNCS(SmearLayer);

}  // namespace caffe
