#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/malabar_layers.hpp"

namespace caffe {

  template <typename Dtype>
  void MatMulLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top) {
    vector<Blob<Dtype>*> tmp_bottom;
    tmp_bottom.resize(bottom.size()-1);
    for(int i = 1; i < bottom.size(); ++i){
      tmp_bottom[i-1] = bottom[i];
    }
    vector<Blob<Dtype>*> tmp_top(1);
    tmp_top[0] = &tmp_k_;
    concat_layer_->Forward(tmp_bottom, tmp_top);

    const Dtype* bottom_data_0 = bottom[0]->gpu_data();  // blob
    const Dtype* bottom_data_1 = tmp_k_.gpu_data();  // matrix
    vector<int> bottom_0_shape = bottom[0]->shape();
    vector<int> bottom_1_shape = tmp_k_.shape();

    const int num = bottom_0_shape[0];
    const int m_ = bottom_0_shape[1] * bottom_0_shape[2];
    // const int m_ =bottom_0_shape[0]* bottom_0_shape[1] * bottom_0_shape[2];
    const int k_ = bottom_0_shape[3];
    const int n_ = bottom_1_shape[3];

    for (int i = 0; i < num; ++i) {
      caffe_gpu_gemm<Dtype>(CblasNoTrans,
                            CblasNoTrans,
                            m_,
                            n_,
                            k_,
                            (Dtype)1.,
                            bottom_data_0 + bottom[0]->offset(i),
                            bottom_data_1 + tmp_k_.offset(i),
                            (Dtype)0.,
                            top[0]->mutable_gpu_data()+top[0]->offset(i));
    }
  }

  template <typename Dtype>
  void MatMulLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                        const vector<bool>& propagate_down,
                                        const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();

    vector<int> top_0_shape = top[0]->shape();
    vector<int> bottom_0_shape = bottom[0]->shape();
    vector<int> bottom_1_shape = bottom[1]->shape();

    const int num = bottom_0_shape[0];
    // const int chan = bottom_0_shape[1];

    // this should NOT be done since it would destroy gradients
    // of other layer, zeroing the diff is done by the solver
    caffe_gpu_set(bottom[0]->count(), (Dtype)0., bottom[0]->mutable_gpu_diff());

    // FOR DEBUGGING
    // for(int b = 0; b < bottom[0]->count(); ++b){
    //     std::cout << " bottom 0 " << b
    //               << " : " << bottom[0]->gpu_data()[b] << std::endl;
    // }
    // for(int b = 0; b < tmp_k_.count(); ++b){
    //   if(tmp_k_.gpu_data()[b] != 0) {
    //     std::cout << " bottom 1 " << b
    //               << " : " << tmp_k_.gpu_data()[b] << std::endl;
    //   }
    // }

    for (int i = 0; i < num; ++i) {
        caffe_gpu_gemm<Dtype>(CblasNoTrans,
                              CblasTrans,
                              bottom[0]->shape()[1]*bottom[0]->shape()[2],
                              bottom[0]->shape()[3],
                              tmp_k_.shape()[3],
                              (Dtype)1.,
                              top_diff + top[0]->offset(i),
                              tmp_k_.gpu_data() + tmp_k_.offset(i),
                              (Dtype)1.,
                              bottom[0]->mutable_gpu_diff() +
                              bottom[0]->offset(i));
    }

    // FOR DEBUGGING
  }
  if (propagate_down[1]) {
    // THIS HAS TO BE CHECKED, SOMEHOW IT SEEMS WEIRD THAT WE SET THIS TO
    // ZERO, WE MIGHT OVERWRITE ALL THE OTHER DIFFS
    // IT IS HOWEVER REQUIRED TO PASS THE GRADIENT TESTS
    // this should NOT be done since it would destroy gradients
    // of other layer, zeroing the diff is done by the solver
    caffe_gpu_set(tmp_k_.count(), (Dtype)0., tmp_k_.mutable_gpu_diff());
    const Dtype* top_diff = top[0]->gpu_diff();

    vector<int> top_0_shape = top[0]->shape();
    vector<int> bottom_0_shape = bottom[0]->shape();
    vector<int> bottom_1_shape = bottom[1]->shape();

    const int num = bottom_0_shape[0];
    // const int chan = bottom_0_shape[1];

    // FOR DEBUGGING
    // std::cout << " A " << bottom[0]->shape()[2]
    //           << " " << bottom[0]->shape()[3] << std::endl;
    // std::cout << " B " << top[0]->shape()[2]
    //           << " " << top[0]->shape()[3] << std::endl;
    // std::cout << " C " << bottom[1]->shape()[2]
    //           << " " << bottom[1]->shape()[3] << std::endl;

    // for(int i = 0; i < top[0]->count(); ++i){
    //   std::cout << " top " << i
    //             << " : " << top[0]->gpu_diff()[i] << std::endl;
    // }
    // for(int b = 0; b < bottom[0]->count(); ++b){
    //     std::cout << " bottom 0 "
    //               << b << " : " << bottom[0]->gpu_data()[b] << std::endl;
    // }
    // for(int b = 0; b < bottom[1]->count(); ++b){
    //     std::cout << " bottom 1 "
    //               << b << " : " << bottom[1]->gpu_data()[b] << std::endl;
    // }

    for (int i = 0; i < num; ++i) {
        caffe_gpu_gemm<Dtype>(CblasTrans,
                              CblasNoTrans,
                              bottom[0]->shape(3),
                              tmp_k_.shape(3),
                              bottom[0]->shape(2)*bottom[0]->shape(1),
                              (Dtype)1.,
                              bottom[0]->gpu_data() + bottom[0]->offset(i),
                              top_diff + top[0]->offset(i),
                              (Dtype)1.,
                              tmp_k_.mutable_gpu_diff() +
                              tmp_k_.offset(i));
    }
    vector<Blob<Dtype>*> tmp_bottom;
    tmp_bottom.resize(bottom.size()-1);
    vector<bool> tmp_propagate_down(tmp_bottom.size());
    for(int i = 1; i < bottom.size(); ++i){
      tmp_bottom[i-1] = bottom[i];
      tmp_propagate_down[i-1] = true;
    }
    vector<Blob<Dtype>*> tmp_top(1);
    tmp_top[0] = &tmp_k_;

    concat_layer_->Backward(tmp_top, tmp_propagate_down, tmp_bottom);
  }
  }

  INSTANTIATE_LAYER_GPU_FUNCS(MatMulLayer);

}  // namespace caffe
