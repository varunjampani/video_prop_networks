#include <vector>

#include <csignal>
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/malabar_layers.hpp"

namespace caffe {

template <typename Dtype>
void MatMulLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) { }

// input / output as is now
// bottom 0 -> N x F x D x L (input blob)
// bottom 1 -> N x 1 x L x K (matrix to be multiplied)
// top -> N x F x D x K

// input / output as is now
// bottom 0 -> N x F x D x L (input blob)
// bottom 1 -> N x 1 x L x KT (T matrix kernels to be multiplied)
// top -> N x FT x D x K

// input / output as it should be
// bottom 0 -> N x F x D x L (input blob)
// bottom 1 -> N x {1,F} x L x K (matrix to be multiplied)
// top -> N x F x D x K

template <typename Dtype>
void MatMulLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  CHECK_GE(bottom.size(), 2);
  vector<int> bottom_0_shape = bottom[0]->shape();
  vector<int> bottom_1_shape = bottom[1]->shape();

  // matrix needs to match dimensions
  CHECK_EQ(bottom_0_shape[0], bottom_1_shape[0])
    << "Matrix dimensions need to match input blob.";
  // CHECK_EQ(1, bottom_1_shape[1])
  //   << "Matrix dimensions need to match input blob.";
  CHECK_EQ(bottom_0_shape[3], bottom_1_shape[2])
    << "Matrix dimensions need to match input blob.";

  for(int i = 1; i < bottom.size(); ++i){
    for(int j = 0; j < bottom_1_shape.size(); ++j){
      CHECK_EQ(bottom_1_shape[j], bottom[i]->shape()[j]);
    }
  }

  vector<int> top_shape = bottom[0]->shape();
  top_shape[3] = bottom_1_shape[3];
  top_shape[1] = bottom_1_shape[1] * bottom_0_shape[1] * (bottom.size()-1);
  top[0]->Reshape(top_shape);

  vector<Blob<Dtype>*> tmp_bottom;
  tmp_bottom.resize(bottom.size()-1);
  for(int i = 1; i < bottom.size(); ++i){
    tmp_bottom[i-1] = bottom[i];
  }
  vector<Blob<Dtype>*> tmp_top(1);
  tmp_top[0] = &tmp_k_;
  concat_layer_->Reshape(tmp_bottom,tmp_top);
}

template <typename Dtype>
void MatMulLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // Dtype* top_data = top[0]->mutable_cpu_data();

  // for(int i = 0; i < bottom[0]->shape().size(); ++i){
  //   std::cout << " shape " << i
  //             << " : " << bottom[0]->shape()[i] << std::endl;
  // }
  vector<Blob<Dtype>*> tmp_bottom;
  tmp_bottom.resize(bottom.size()-1);
  for(int i = 1; i < bottom.size(); ++i){
    tmp_bottom[i-1] = bottom[i];
  }
  vector<Blob<Dtype>*> tmp_top(1);
  tmp_top[0] = &tmp_k_;
  concat_layer_->Forward(tmp_bottom, tmp_top);

  const Dtype* bottom_data_0 = bottom[0]->cpu_data();  // blob
  const Dtype* bottom_data_1 = tmp_k_.cpu_data();  // matrix
  vector<int> bottom_0_shape = bottom[0]->shape();
  vector<int> bottom_1_shape = tmp_k_.shape();

  const int num = bottom_0_shape[0];
  const int m_ = bottom_0_shape[1] * bottom_0_shape[2];
  // const int m_ =bottom_0_shape[0]* bottom_0_shape[1] * bottom_0_shape[2];
  const int k_ = bottom_0_shape[3];
  const int n_ = bottom_1_shape[3];

  // std::cout << "m " << m_ << std::endl;
  // std::cout << "k " << k_ << std::endl;
  // std::cout << "n " << n_ << std::endl;

  // for(int i = 0; i < top[0]->shape().size(); ++i){
  //   std::cout << " shape " << i
  //             << " : " << top[0]->shape()[i] << std::endl;
  // }

  for (int i = 0; i < num; ++i) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans,
                          CblasNoTrans,
                          m_,
                          n_,
                          k_,
                          (Dtype)1.,
                          bottom_data_0 + bottom[0]->offset(i),
                          bottom_data_1 + tmp_k_.offset(i),
                          (Dtype)0.,
                          top[0]->mutable_cpu_data()+top[0]->offset(i));
  }
}

template <typename Dtype>
void MatMulLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();

    vector<int> top_0_shape = top[0]->shape();
    vector<int> bottom_0_shape = bottom[0]->shape();
    vector<int> bottom_1_shape = bottom[1]->shape();

    const int num = bottom_0_shape[0];
    // const int chan = bottom_0_shape[1];

    // this should NOT be done since it would destroy gradients
    // of other layer, zeroing the diff is done by the solver
    caffe_set(bottom[0]->count(), (Dtype)0., bottom[0]->mutable_cpu_diff());

    // FOR DEBUGGING
    // for(int i = 0; i < top[0]->count(); ++i){
    //   if(top[0]->cpu_diff()[i] != 0) {
    //   std::cout << " top " << i
    //             << " : " << top[0]->cpu_diff()[i] << std::endl;
    //   }
    // }
    // for(int b = 0; b < bottom[0]->count(); ++b){
    //     std::cout << " bottom 0 " << b
    //               << " : " << bottom[0]->cpu_data()[b] << std::endl;
    // }
    // for(int b = 0; b < tmp_k_.count(); ++b){
    //   if(tmp_k_.cpu_data()[b] != 0) {
    //     std::cout << " bottom 1 " << b
    //               << " : " << tmp_k_.cpu_data()[b] << std::endl;
    //   }
    // }

    for (int i = 0; i < num; ++i) {
        caffe_cpu_gemm<Dtype>(CblasNoTrans,
                              CblasTrans,
                              bottom[0]->shape()[1]*bottom[0]->shape()[2],
                              bottom[0]->shape()[3],
                              tmp_k_.shape()[3],
                              (Dtype)1.,
                              top_diff + top[0]->offset(i),
                              tmp_k_.cpu_data() + tmp_k_.offset(i),
                              (Dtype)1.,
                              bottom[0]->mutable_cpu_diff() +
                              bottom[0]->offset(i));
    }

    // FOR DEBUGGING
    // for(int b = 0; b < bottom[0]->count(); ++b){
    //   if(bottom[0]->cpu_diff()[b] != 0) {
    //     std::cout << " gradient " << b
    //               << " : " << bottom[0]->cpu_diff()[b] << std::endl;
    //   }
    // }
  }
  if (propagate_down[1]) {
    // THIS HAS TO BE CHECKED, SOMEHOW IT SEEMS WEIRD THAT WE SET THIS TO
    // ZERO, WE MIGHT OVERWRITE ALL THE OTHER DIFFS
    // IT IS HOWEVER REQUIRED TO PASS THE GRADIENT TESTS
    // this should NOT be done since it would destroy gradients
    // of other layer, zeroing the diff is done by the solver
    caffe_set(tmp_k_.count(), (Dtype)0., tmp_k_.mutable_cpu_diff());
    const Dtype* top_diff = top[0]->cpu_diff();

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
    //             << " : " << top[0]->cpu_diff()[i] << std::endl;
    // }
    // for(int b = 0; b < bottom[0]->count(); ++b){
    //     std::cout << " bottom 0 "
    //               << b << " : " << bottom[0]->cpu_data()[b] << std::endl;
    // }
    // for(int b = 0; b < bottom[1]->count(); ++b){
    //     std::cout << " bottom 1 "
    //               << b << " : " << bottom[1]->cpu_data()[b] << std::endl;
    // }

    for (int i = 0; i < num; ++i) {
        caffe_cpu_gemm<Dtype>(CblasTrans,
                              CblasNoTrans,
                              bottom[0]->shape(3),
                              tmp_k_.shape(3),
                              bottom[0]->shape(2)*bottom[0]->shape(1),
                              (Dtype)1.,
                              bottom[0]->cpu_data() + bottom[0]->offset(i),
                              top_diff + top[0]->offset(i),
                              (Dtype)1.,
                              tmp_k_.mutable_cpu_diff() +
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

#ifdef CPU_ONLY
STUB_GPU(MatMulLayer);
#endif

INSTANTIATE_CLASS(MatMulLayer);
REGISTER_LAYER_CLASS(MatMul);

}  // namespace caffe
