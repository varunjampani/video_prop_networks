#include "caffe/util/math_functions.hpp"
#include "caffe/malabar_layers.hpp"
#include "caffe/util/gpu_util.cuh"


namespace caffe {

template <typename Dtype>
__global__ void SpixelFeatureXYForwardGPU(const int nthreads,
  const Dtype* index_data, const Dtype ignore_idx_value,
  const int out_dim, const int height,
  const int width, const int max_spixels,
  const float xy_scale, Dtype* top_data, Dtype* count_data) {
    CUDA_KERNEL_LOOP(index, nthreads) {
      const int spatial_dim = height * width;
      const int n = index / spatial_dim;
      const int s = index % spatial_dim;
      const int idx = static_cast<int>(index_data[n * spatial_dim + s]);

      if (idx != ignore_idx_value) {
        const int y = s / width;
        const int x = s % width;

        int count_offset = (n * max_spixels + idx);
        for (int k = 0; k < out_dim; k++) {
          int top_offset = ((n * out_dim + k) * max_spixels + idx);
          if (k == 0) {
            caffe_gpu_atomic_add((Dtype) xy_scale * y, top_data + top_offset);
          } else if (k == 1) {
            caffe_gpu_atomic_add((Dtype) xy_scale * x, top_data + top_offset);
          }
        }
        caffe_gpu_atomic_add((Dtype) 1., count_data + count_offset);
      }
    }
}

template <typename Dtype>
__global__ void SpixelFeatureRGBXYForwardGPU(const int nthreads,
  const Dtype* bottom_data, const Dtype* index_data,
  const Dtype ignore_idx_value,
  const int out_dim, const int in_dim, const int height, const int width,
  const int max_spixels, const float rgbxy_rgb_scale,
  const float rgbxy_xy_scale, Dtype* top_data, Dtype* count_data) {
    CUDA_KERNEL_LOOP(index, nthreads) {
      const int spatial_dim = height * width;
      const int n = index / spatial_dim;
      const int s = index % spatial_dim;
      const int idx = static_cast<int>(index_data[n * spatial_dim + s]);

      if (idx != ignore_idx_value) {
        const int y = s / width;
        const int x = s % width;

        int count_offset = (n * max_spixels + idx);
        for (int k = 0; k < out_dim; k++) {
          int top_offset = ((n * out_dim + k) * max_spixels + idx);
          if (k < in_dim) {
            int bottom_offset = ((n * in_dim + k) * spatial_dim + s);
            caffe_gpu_atomic_add((Dtype) rgbxy_rgb_scale * bottom_data[bottom_offset],
              top_data + top_offset);
          } else if (k == in_dim) {
            caffe_gpu_atomic_add((Dtype) rgbxy_xy_scale * y, top_data + top_offset);
          } else if (k == in_dim + 1) {
            caffe_gpu_atomic_add((Dtype) rgbxy_xy_scale * x, top_data + top_offset);
          }
        }
        caffe_gpu_atomic_add((Dtype) 1., count_data + count_offset);
      }
    }
}

template <typename Dtype>
__global__ void SpixelFeatureRGBForwardGPU(const int nthreads,
  const Dtype* bottom_data, const Dtype* index_data,
  const Dtype ignore_idx_value,
  const int out_dim, const int in_dim, const int height, const int width,
  const int max_spixels, const float rgb_scale,
  Dtype* top_data, Dtype* count_data) {
    CUDA_KERNEL_LOOP(index, nthreads) {
      const int spatial_dim = height * width;
      const int n = index / spatial_dim;
      const int s = index % spatial_dim;
      const int idx = static_cast<int>(index_data[n * spatial_dim + s]);

      if (idx != ignore_idx_value) {
        int count_offset = (n * max_spixels + idx);
        for (int k = 0; k < out_dim; k++) {
          int top_offset = ((n * out_dim + k) * max_spixels + idx);
          if (k < in_dim) {
            int bottom_offset = ((n * in_dim + k) * spatial_dim + s);
            caffe_gpu_atomic_add((Dtype) rgb_scale * bottom_data[bottom_offset],
              top_data + top_offset);
          }
        }
        caffe_gpu_atomic_add((Dtype) 1., count_data + count_offset);
      }
    }
}

template <typename Dtype>
__global__ void SpixelFeatureXYRGBXYForwardGPU(const int nthreads,
  const Dtype* bottom_data, const Dtype* index_data,
  const Dtype ignore_idx_value,
  const int out_dim, const int in_dim, const int height, const int width,
  const int max_spixels, const float xy_scale, const float rgbxy_rgb_scale,
  const float rgbxy_xy_scale, Dtype* top_data, Dtype* count_data) {
    CUDA_KERNEL_LOOP(index, nthreads) {
      const int spatial_dim = height * width;
      const int n = index / spatial_dim;
      const int s = index % spatial_dim;
      const int idx = static_cast<int>(index_data[n * spatial_dim + s]);

      if (idx != ignore_idx_value) {
        const int y = s / width;
        const int x = s % width;

        int count_offset = (n * max_spixels + idx);
        for (int k = 0; k < out_dim; k++) {
          int top_offset = ((n * out_dim + k) * max_spixels + idx);
          if (k == 0) {
            caffe_gpu_atomic_add((Dtype) xy_scale * y, top_data + top_offset);
          } else if (k == 1) {
            caffe_gpu_atomic_add((Dtype) xy_scale * x, top_data + top_offset);
          } else if (k < in_dim + 2) {
            int bottom_offset = ((n * in_dim + (k-2)) * spatial_dim + s);
            caffe_gpu_atomic_add((Dtype) rgbxy_rgb_scale * bottom_data[bottom_offset],
              top_data + top_offset);
          } else if (k == in_dim + 2) {
            caffe_gpu_atomic_add((Dtype) rgbxy_xy_scale * y, top_data + top_offset);
          } else if (k == in_dim + 3) {
            caffe_gpu_atomic_add((Dtype) rgbxy_xy_scale * x, top_data + top_offset);
          }
        }
        caffe_gpu_atomic_add((Dtype) 1., count_data + count_offset);
      }
    }
}

template <typename Dtype>
__global__ void SpixelFeatureRGBXYRGBXYForwardGPU(const int nthreads,
  const Dtype* bottom_data, const Dtype* index_data,
  const Dtype ignore_idx_value,
  const int out_dim, const int in_dim, const int height, const int width,
  const int max_spixels, const float rgb_scale,
  const float xy_scale, const float rgbxy_rgb_scale,
  const float rgbxy_xy_scale, Dtype* top_data, Dtype* count_data) {
    CUDA_KERNEL_LOOP(index, nthreads) {
      const int spatial_dim = height * width;
      const int n = index / spatial_dim;
      const int s = index % spatial_dim;
      const int idx = static_cast<int>(index_data[n * spatial_dim + s]);

      if (idx != ignore_idx_value) {
        const int y = s / width;
        const int x = s % width;

        int count_offset = (n * max_spixels + idx);
        for (int k = 0; k < out_dim; k++) {
          int top_offset = ((n * out_dim + k) * max_spixels + idx);
          if (k < in_dim) {
            int bottom_offset = ((n * in_dim + k) * spatial_dim + s);
            caffe_gpu_atomic_add((Dtype) rgb_scale * bottom_data[bottom_offset],
              top_data + top_offset);
          } else if (k == in_dim) {
            caffe_gpu_atomic_add((Dtype) xy_scale * y, top_data + top_offset);
          } else if (k == in_dim + 1) {
            caffe_gpu_atomic_add((Dtype) xy_scale * x, top_data + top_offset);
          } else if (k < 2 * in_dim + 2) {
            int bottom_offset = ((n * in_dim + ((k-2) % in_dim)) * spatial_dim + s);
            caffe_gpu_atomic_add((Dtype) rgbxy_rgb_scale * bottom_data[bottom_offset],
              top_data + top_offset);
          } else if (k == 2 * in_dim + 2) {
            caffe_gpu_atomic_add((Dtype) rgbxy_xy_scale * y, top_data + top_offset);
          } else if (k == 2 * in_dim + 3) {
            caffe_gpu_atomic_add((Dtype) rgbxy_xy_scale * x, top_data + top_offset);
          }
        }
        caffe_gpu_atomic_add((Dtype) 1., count_data + count_offset);
      }
    }
}

template <typename Dtype>
__global__ void SpixelFeatureAverageForwardGPU(const int nthreads,
  const int max_spixels, const int out_dim, const float ignore_value,
  Dtype* top_data, Dtype* count_data) {
    CUDA_KERNEL_LOOP(index, nthreads) {
      const int n = index / max_spixels;
      const int s = index % max_spixels;

      const int count_offset = (n * max_spixels + s);
      for (int k = 0; k < out_dim; k++) {
        const int top_offset = ((n * out_dim + k) * max_spixels + s);
        if (count_data[count_offset] == 0) {
          top_data[top_offset] = ignore_value;
        } else {
          top_data[top_offset] /= count_data[count_offset];
        }
      }
    }
}

template <typename Dtype>
__global__ void SpixelFeatureCopyToPixelsGPU(const int nthreads,
  const Dtype* index_data, const Dtype ignore_idx_value,
  const int spatial_dim, const int max_spixels,
  const int out_dim, const float ignore_feature_value,
  Dtype* top_data, Dtype* top_data_2) {
    CUDA_KERNEL_LOOP(index, nthreads) {
      const int n = index / spatial_dim;
      const int s = index % spatial_dim;
      const int idx = static_cast<int>(index_data[n * spatial_dim + s]);

      if (idx != ignore_idx_value) {
        for (int k = 0; k < out_dim; k++) {
          int top_offset = ((n * out_dim + k) * max_spixels + idx);
          int top_offset_2 = ((n * out_dim + k) * spatial_dim + s);
          top_data_2[top_offset_2] = top_data[top_offset];
        }
      }
      else {
        for (int k = 0; k < out_dim; k++) {
          int top_offset_2 = ((n * out_dim + k) * spatial_dim + s);
          top_data_2[top_offset_2] = ignore_feature_value;
        }
      }
    }
}

/*
Forward GPU function
*/
template <typename Dtype>
void SpixelFeatureLayer<Dtype>::Forward_gpu(
const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  caffe_gpu_set(top[0]->count(), (Dtype)0., top[0]->mutable_gpu_data());
  caffe_gpu_set(spixel_counts_.count(), (Dtype)0.,
    spixel_counts_.mutable_gpu_data());

  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* index_data = bottom[1]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  Dtype* count_data = spixel_counts_.mutable_gpu_data();

  switch (this->layer_param_.spixel_feature_param().type()) {
  case SpixelFeatureParameter_Feature_AVGXY: {
    const int nthreads = num_ * height_ * width_;
    // NOLINT_NEXT_LINE(whitespace/operators)
    SpixelFeatureXYForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, index_data, ignore_idx_value_,
                                out_channels_, height_, width_,
                                max_spixels_, xy_scale_, top_data,
                                count_data);

    const int nthreads2 = num_ * max_spixels_;
    // NOLINT_NEXT_LINE(whitespace/operators)
    SpixelFeatureAverageForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads2),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads2, max_spixels_,
                                out_channels_, ignore_feature_value_,
                                top_data, count_data);
    break;
  }
  case SpixelFeatureParameter_Feature_AVGRGBXY: {
    const int nthreads = num_ * height_ * width_;
    // NOLINT_NEXT_LINE(whitespace/operators)
    SpixelFeatureRGBXYForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, bottom_data, index_data,
                                ignore_idx_value_,
                                out_channels_, in_channels_, height_, width_,
                                max_spixels_, rgbxy_rgb_scale_, rgbxy_xy_scale_,
                                top_data, count_data);

    const int nthreads2 = num_ * max_spixels_;
    // NOLINT_NEXT_LINE(whitespace/operators)
    SpixelFeatureAverageForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads2),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads2, max_spixels_,
                                out_channels_, ignore_feature_value_,
                                top_data, count_data);
    break;
  }
  case SpixelFeatureParameter_Feature_AVGXYRGBXY: {
    const int nthreads = num_ * height_ * width_;
    // NOLINT_NEXT_LINE(whitespace/operators)
    SpixelFeatureXYRGBXYForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, bottom_data, index_data,
                                ignore_idx_value_,
                                out_channels_, in_channels_, height_, width_,
                                max_spixels_, xy_scale_,
                                rgbxy_rgb_scale_, rgbxy_xy_scale_,
                                top_data, count_data);

    const int nthreads2 = num_ * max_spixels_;
    // NOLINT_NEXT_LINE(whitespace/operators)
    SpixelFeatureAverageForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads2),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads2, max_spixels_,
                                out_channels_, ignore_feature_value_,
                                top_data, count_data);
    break;
  }
  case SpixelFeatureParameter_Feature_AVGRGBXYRGBXY: {
    const int nthreads = num_ * height_ * width_;
    // NOLINT_NEXT_LINE(whitespace/operators)
    SpixelFeatureRGBXYRGBXYForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, bottom_data, index_data,
                                ignore_idx_value_,
                                out_channels_, in_channels_, height_, width_,
                                max_spixels_, rgb_scale_, xy_scale_,
                                rgbxy_rgb_scale_, rgbxy_xy_scale_,
                                top_data, count_data);

    const int nthreads2 = num_ * max_spixels_;
    // NOLINT_NEXT_LINE(whitespace/operators)
    SpixelFeatureAverageForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads2),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads2, max_spixels_,
                                out_channels_, ignore_feature_value_,
                                top_data, count_data);
    break;
  }
  case SpixelFeatureParameter_Feature_AVGRGB: {
    const int nthreads = num_ * height_ * width_;
    // NOLINT_NEXT_LINE(whitespace/operators)
    SpixelFeatureRGBForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, bottom_data, index_data,
                                ignore_idx_value_,
                                out_channels_, in_channels_, height_, width_,
                                max_spixels_, rgb_scale_,
                                top_data, count_data);

    const int nthreads2 = num_ * max_spixels_;
    // NOLINT_NEXT_LINE(whitespace/operators)
    SpixelFeatureAverageForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads2),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads2, max_spixels_,
                                out_channels_, ignore_feature_value_,
                                top_data, count_data);
    break;
  }
  default:
    LOG(FATAL) << "Undefined feature type of superpixel feature";
  }

  if (top.size() > 1) {
    caffe_gpu_set(top[1]->count(), (Dtype)0., top[1]->mutable_gpu_data());
    Dtype* top_data_2 = top[1]->mutable_gpu_data();
    const int nthreads = num_ * height_ * width_;
    // NOLINT_NEXT_LINE(whitespace/operators)
    SpixelFeatureCopyToPixelsGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, index_data, ignore_idx_value_,
                                height_ * width_, max_spixels_,
                                out_channels_, ignore_feature_value_,
                                top_data, top_data_2);
  }
}

/*
Backward GPU function (NOT_IMPLEMENTED for now)
 */
template <typename Dtype>
void SpixelFeatureLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
}

INSTANTIATE_LAYER_GPU_FUNCS(SpixelFeatureLayer);

}  // namespace caffe
