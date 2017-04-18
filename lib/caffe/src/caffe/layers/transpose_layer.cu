#include "caffe/util/math_functions.hpp"
#include "caffe/malabar_layers.hpp"


namespace caffe {

template <typename Dtype>
__global__ void TransposeForwardGPU(const int nthreads, const Dtype* bottom_data,
  const int channels, const int height, const int width,
  Dtype* top_data) {
    CUDA_KERNEL_LOOP(index, nthreads) {
      const int n = index / (channels * height * width);
      const int s = index % (channels * height * width);

      const int c = s / (height * width);
      const int t = s % (height * width);

      const int y = t / width;
      const int x = t % width;

      int top_offset = ((n * channels + c) * width + x) * height + y;
      int bottom_offset = ((n * channels + c) * height + y) * width + x;
      top_data[top_offset] = bottom_data[bottom_offset];
    }
}

template <typename Dtype>
__global__ void TransposeBackwardGPU(const int nthreads, const Dtype* top_diff,
  const int channels, const int height, const int width,
  Dtype* bottom_diff) {
    CUDA_KERNEL_LOOP(index, nthreads) {
      const int n = index / (channels * height * width);
      const int s = index % (channels * height * width);

      const int c = s / (height * width);
      const int t = s % (height * width);

      const int y = t / width;
      const int x = t % width;

      int top_offset = ((n * channels + c) * width + x) * height + y;
      int bottom_offset = ((n * channels + c) * height + y) * width + x;
      bottom_diff[bottom_offset] = top_diff[top_offset];
    }
}

/*
Forward GPU function
*/
template <typename Dtype>
void TransposeLayer<Dtype>::Forward_gpu(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();

    const int nthreads = num_ * channels_ * height_ * width_;

    // NOLINT_NEXT_LINE(whitespace/operators)
    TransposeForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, bottom_data,
                                channels_, height_, width_, top_data);
}

/*
Backward GPU function
 */
template <typename Dtype>
void TransposeLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

      Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
      const Dtype* top_diff = top[0]->gpu_diff();

      const int nthreads = num_ * channels_ * height_ * width_;

      // NOLINT_NEXT_LINE(whitespace/operators)
      TransposeBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, top_diff,
                                  channels_, height_, width_, bottom_diff);
}

INSTANTIATE_LAYER_GPU_FUNCS(TransposeLayer);

}  // namespace caffe
