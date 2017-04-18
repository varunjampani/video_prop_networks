#include "caffe/util/math_functions.hpp"
#include "caffe/malabar_layers.hpp"
#include "caffe/util/gpu_util.cuh"

#include <cmath>


namespace caffe {

template <typename Dtype>
__global__ void PdistForwardGPU(const int nthreads, const Dtype* bottom_data_1,
  const Dtype* bottom_data_0, const int dim, const int spatial_dim_1,
  const int spatial_dim_0, const float ignore_value, const float scale_value,
  Dtype* top_data) {
    CUDA_KERNEL_LOOP(index, nthreads) {
      const int n = index % spatial_dim_1;
      const int s = index / spatial_dim_1;

      Dtype sq_dist = 0;
      for (int c = 0; c < dim; c++) {
        Dtype bottom_1_value = bottom_data_1[c * spatial_dim_1 + n];
        Dtype bottom_0_value = bottom_data_0[c * spatial_dim_0 + s];
        if (bottom_1_value == ignore_value || bottom_0_value == ignore_value) {
          sq_dist = 1e10;
          break;
        } else {
          sq_dist += pow(bottom_1_value - bottom_0_value, 2);
        }
      }
      top_data[spatial_dim_0 * n + s] = scale_value * sq_dist;
    }
}

template <typename Dtype>
__global__ void PdistBackwardGPU(const int nthreads, const Dtype* bottom_data_1,
  const Dtype* bottom_data_0, const Dtype* top_data, const Dtype* top_diff,
  const int dim, const int spatial_dim_1, const int spatial_dim_0,
  const float ignore_value, const float scale_value,
  Dtype* bottom_diff_0, Dtype* bottom_diff_1) {
    CUDA_KERNEL_LOOP(index, nthreads) {
      const int n = index % spatial_dim_1;
      const int s = index / spatial_dim_1;

      int top_offset = spatial_dim_0 * n + s;
      if (top_data[spatial_dim_0 * n + s] != scale_value * 1e10) {
        for (int c = 0; c < dim; c++) {
          int bottom_1_offset = c * spatial_dim_1 + n;
          int bottom_0_offset = c * spatial_dim_0 + s;
          Dtype bottom_1_value = bottom_data_1[bottom_1_offset];
          Dtype bottom_0_value = bottom_data_0[bottom_0_offset];

          if (bottom_1_value == ignore_value) {
            bottom_diff_1[bottom_1_offset] = 0;
          } else {
            caffe_gpu_atomic_add((Dtype) top_diff[top_offset] *
              2 * scale_value * (bottom_1_value - bottom_0_value),
                bottom_diff_1 + bottom_1_offset);
          }
          if (bottom_0_value == ignore_value) {
            bottom_diff_0[bottom_0_offset] = 0;
          } else {
            caffe_gpu_atomic_add((Dtype) top_diff[top_offset] *
              2 * scale_value * (bottom_0_value - bottom_1_value),
                bottom_diff_0 + bottom_0_offset);
          }
        }
      }
    }
}

// ((n * channels() + c) * height() + h) * width() + w;

/*
Forward GPU function
*/
template <typename Dtype>
void PdistLayer<Dtype>::Forward_gpu(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

    const Dtype* bottom_data_0 = bottom[0]->gpu_data();
    const Dtype* bottom_data_1 = bottom[1]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();

    const int nthreads = out_height_ * out_width_;

    for (int i = 0; i < num_; i++) {
      const Dtype* b_data_1 = bottom_data_1 + bottom[1]->offset(i);
      const Dtype* b_data_0 = bottom_data_0 +bottom[0]->offset(i);
      Dtype* t_data = top_data + top[0]->offset(i);

      // NOLINT_NEXT_LINE(whitespace/operators)
      PdistForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, b_data_1, b_data_0,
                                  channels_, out_height_, out_width_,
                                  ignore_value_, scale_value_, t_data);
    }
}

/*
Backward GPU function
 */
template <typename Dtype>
void PdistLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

    if (propagate_down[0] || propagate_down[1]) {
      caffe_gpu_set(bottom[0]->count(), (Dtype)0., bottom[0]->mutable_gpu_diff());
      caffe_gpu_set(bottom[1]->count(), (Dtype)0., bottom[1]->mutable_gpu_diff());

      const Dtype* bottom_data_0 = bottom[0]->gpu_data();
      const Dtype* bottom_data_1 = bottom[1]->gpu_data();
      const Dtype* top_data = top[0]->gpu_data();

      const Dtype* top_diff = top[0]->gpu_diff();
      Dtype* bottom_diff_0 = bottom[0]->mutable_gpu_diff();
      Dtype* bottom_diff_1 = bottom[1]->mutable_gpu_diff();

      const int nthreads = out_height_ * out_width_;

      for (int i = 0; i < num_; i++) {
        const Dtype* b_data_1 = bottom_data_1 + bottom[1]->offset(i);
        const Dtype* b_data_0 = bottom_data_0 +bottom[0]->offset(i);
        const Dtype* t_data = top_data + top[0]->offset(i);

        Dtype* bdiff_data_1 = bottom_diff_1 + bottom[1]->offset(i);
        Dtype* bdiff_data_0 = bottom_diff_0 +bottom[0]->offset(i);
        const Dtype* tdiff_data = top_diff + top[0]->offset(i);

        // NOLINT_NEXT_LINE(whitespace/operators)
        PdistBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
          CAFFE_CUDA_NUM_THREADS>>>(nthreads, b_data_1, b_data_0, t_data,
                                    tdiff_data, channels_, out_height_,
                                    out_width_, ignore_value_, scale_value_,
                                    bdiff_data_0, bdiff_data_1);
      }
    }
}

INSTANTIATE_LAYER_GPU_FUNCS(PdistLayer);

}  // namespace caffe
