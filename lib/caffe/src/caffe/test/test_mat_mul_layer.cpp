#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/malabar_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

#ifndef CPU_ONLY
extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;
#endif

template <typename TypeParam>
class MatMulLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  MatMulLayerTest()
      : blob_bottom_0_(new Blob<Dtype>(3, 2, 3, 4)),
        blob_bottom_1_(new Blob<Dtype>(3, 1, 4, 2)),
        blob_bottom_2_(new Blob<Dtype>(3, 1, 4, 2)),
        blob_bottom_0_d_(new Blob<Dtype>(2, 3, 2, 3)),
        blob_bottom_1_d_(new Blob<Dtype>(2, 1, 3, 3)),
        blob_bottom_2_d_(new Blob<Dtype>(2, 1, 3, 3)),
        blob_top_(new Blob<Dtype>()),
        blob_top_d_(new Blob<Dtype>()) {
    Caffe::set_random_seed(1701);
    // fill the values
    FillerParameter filler_param;
    filler_param.set_value(1.);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_0_);
    filler.Fill(this->blob_bottom_1_);
    filler.Fill(this->blob_bottom_2_);
    blob_bottom_vec_.push_back(blob_bottom_0_);
    blob_bottom_vec_.push_back(blob_bottom_1_);
    blob_bottom_vec_.push_back(blob_bottom_2_);
    blob_top_vec_.push_back(blob_top_);
    blob_top_vec_d_.push_back(blob_top_d_);
    for (int i = 0; i < blob_bottom_0_d_->count(); i++){
      blob_bottom_0_d_->mutable_cpu_data()[i] = i;
    }
    for (int i = 0; i < blob_bottom_1_d_->count(); i++){
      blob_bottom_1_d_->mutable_cpu_data()[i] = blob_bottom_0_d_->count() + i;
    }
    for (int i = 0; i < blob_bottom_2_d_->count(); i++){
      blob_bottom_2_d_->mutable_cpu_data()[i] =
        blob_bottom_0_d_->count() + blob_bottom_1_d_->count() + i;
    }
    blob_bottom_vec_d_.push_back(blob_bottom_0_d_);
    blob_bottom_vec_d_.push_back(blob_bottom_1_d_);
    blob_bottom_vec_d_.push_back(blob_bottom_2_d_);
    // numpy data :
    // A = np.array(range(2*3*2*3)).reshape(2,3,2,3)
    // B = np.zeros(range(2*2*3*3)).reshape(2,2,3,3)
    // B[:,0,:,:] =  np.array([36 + val for val in range(2*3*3)]).reshape(2,3,3)
    // B[:,1,:,:] =  np.array([54 + val for val in range(2*3*3)]).reshape(2,3,3)
    // B = B.reshape(2,1,3,6)
  }
  virtual ~MatMulLayerTest() {
    delete blob_bottom_0_;
    delete blob_bottom_1_;
    delete blob_bottom_2_;
    delete blob_bottom_0_d_;
    delete blob_bottom_1_d_;
    delete blob_bottom_2_d_;
    delete blob_top_;
    delete blob_top_d_;
  }
  Blob<Dtype>* const blob_bottom_0_;
  Blob<Dtype>* const blob_bottom_1_;
  Blob<Dtype>* const blob_bottom_2_;
  Blob<Dtype>* const blob_bottom_0_d_;
  Blob<Dtype>* const blob_bottom_1_d_;
  Blob<Dtype>* const blob_bottom_2_d_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const blob_top_d_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_bottom_vec_d_;
  vector<Blob<Dtype>*> blob_top_vec_;
  vector<Blob<Dtype>*> blob_top_vec_d_;
};

TYPED_TEST_CASE(MatMulLayerTest, TestDtypesAndDevices);

TYPED_TEST(MatMulLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  MatMulLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_d_, this->blob_top_vec_d_);
  EXPECT_EQ(this->blob_top_d_->num(), 2);
  EXPECT_EQ(this->blob_top_d_->channels(), 6);
  EXPECT_EQ(this->blob_top_d_->height(), 2);
  EXPECT_EQ(this->blob_top_d_->width(), 3);
}

TYPED_TEST(MatMulLayerTest, TestForwardBackward) {
  typedef typename TypeParam::Dtype Dtype;
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    shared_ptr<MatMulLayer<Dtype> > layer(
        new MatMulLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_d_, this->blob_top_vec_d_);
    layer->Forward(this->blob_bottom_vec_d_, this->blob_top_vec_d_);

    vector<bool> propagate_down(this->blob_bottom_vec_d_.size(), true);
    layer->Backward(this->blob_top_vec_d_, propagate_down, this->blob_bottom_vec_d_);
  }
}

//
TYPED_TEST(MatMulLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    shared_ptr<MatMulLayer<Dtype> > layer(
        new MatMulLayer<Dtype>(layer_param));

    layer->SetUp(this->blob_bottom_vec_d_, this->blob_top_vec_d_);
    layer->Forward(this->blob_bottom_vec_d_, this->blob_top_vec_d_);
    const Dtype* data = this->blob_top_d_->cpu_data();

    const vector<int> bottom_0_shape = this->blob_bottom_0_d_->shape();
    const vector<int> bottom_1_shape = this->blob_bottom_1_d_->shape();

    const Dtype* data_0 = this->blob_bottom_0_d_->cpu_data();

    //    std::cout << "out_channel_size " << this->blob_top_d_[0].shape()[1] << std::endl; // 
    // for every top blob element ...
    for (int n = 0; n < bottom_0_shape[0]; ++n) {
      int out_channel = -1;
      int layer_channels = -1;
      bool update_c = false;
        for (int c = 0; c < bottom_0_shape[1]; ++c) {
          out_channel += 1;
          update_c = true;

          for (int y = 0; y < bottom_0_shape[2]; ++y) {
            for(int kk = 1; kk < this->blob_bottom_vec_d_.size();++kk){
              layer_channels += 1;
              if (!update_c){
                if (layer_channels % bottom_0_shape[2] == 0){
                    out_channel += 1;
                }
              }else{
                update_c = false;
              }
            const Dtype* data_1 = this->blob_bottom_vec_d_[kk]->cpu_data();
              for (int kx = 0; kx < bottom_1_shape[3]; ++kx) {
                // ... compute the result by looping

                // blob N x C x Y x X
                // mat  N x  x X x K

                // out N x C x Y x K

                Dtype res = 0;
                for (int x = 0; x < bottom_0_shape[3]; ++x) {
                  const int i = this->blob_bottom_0_d_->offset(n, c, y, x);
                  const int j = this->blob_bottom_1_d_->offset(n, 0, x, kx);
                  // std::cout << "(n,c,y,x) = ()" << n << ","
                  //           << c << "," << y << "," << x
                  //           << " = " <<  data_0[i] << std::endl;
                  // std::cout << "(n,0,x,kx) = ()" << n << ",0,"
                  //           << x << "," << kx << " = " <<  data_1[j] << std::endl;
                  // std::cout << "i,j = " << i << "," << j << std::endl;
                  res += data_0[i] * data_1[j];
                }

                // std::cout << "c , kk " << c << "," << kk << std::endl;
                const int i = this->blob_top_d_->offset(n, out_channel, layer_channels%bottom_0_shape[2], kx);
                EXPECT_NEAR(data[i], res, 1e-4);
                // std::cout << "index = " << out_channel << "," << layer_channels%bottom_0_shape[2] << "," << kx << std::endl;
                // std::cout << "res,data = " << res << "," << data[i] << std::endl;
              }
            }
          }
      }
    }
  } else {
    LOG(ERROR) << "test failed!.";
  }
}
//
TYPED_TEST(MatMulLayerTest, TestForwardRandom) {
  typedef typename TypeParam::Dtype Dtype;
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    shared_ptr<MatMulLayer<Dtype> > layer(
        new MatMulLayer<Dtype>(layer_param));

    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype* data = this->blob_top_->cpu_data();
    

    const vector<int> bottom_0_shape = this->blob_bottom_0_->shape();
    const vector<int> bottom_1_shape = this->blob_bottom_1_->shape();

    const Dtype* data_0 = this->blob_bottom_0_->cpu_data();

    //    std::cout << "out_channel_size " << this->blob_top_[0].shape()[1] << std::endl;
    // for every top blob element ...
    for (int n = 0; n < bottom_0_shape[0]; ++n) {
      int out_channel = -1;
      int layer_channels = -1;
      bool update_c = false;
      for (int c = 0; c < bottom_0_shape[1]; ++c) {
        out_channel += 1;
        update_c = true;

        for (int y = 0; y < bottom_0_shape[2]; ++y) {
          for(int kk = 1; kk < this->blob_bottom_vec_.size();++kk){
            layer_channels += 1;
            if (!update_c){
              if (layer_channels % bottom_0_shape[2] == 0){
                out_channel += 1;
              }
            }else{
              update_c = false;
            }
            const Dtype* data_1 = this->blob_bottom_vec_[kk]->cpu_data();
            for (int kx = 0; kx < bottom_1_shape[3]; ++kx) {
              // ... compute the result by looping

              // blob N x C x Y x X
              // mat  N x  x X x K

              // out N x C x Y x K

              Dtype res = 0;
              for (int x = 0; x < bottom_0_shape[3]; ++x) {
                const int i = this->blob_bottom_0_->offset(n, c, y, x);
                const int j = this->blob_bottom_1_->offset(n, 0, x, kx);
                // std::cout << "(n,c,y,x) = ()" << n << ","
                //           << c << "," << y << "," << x
                //           << " = " <<  data_0[i] << std::endl;
                // std::cout << "(n,0,x,kx) = ()" << n << ",0,"
                //           << x << "," << kx << " = " <<  data_1[j] << std::endl;
                // std::cout << "i,j = " << i << "," << j << std::endl;
                res += data_0[i] * data_1[j];
              }

              const int i = this->blob_top_->offset(n, out_channel, layer_channels%bottom_0_shape[2], kx);
              EXPECT_NEAR(data[i], res, 1e-4);
            }
          }
        }
      }
    }
  } else {
    LOG(ERROR) << "test failed!.";
  }
}

TYPED_TEST(MatMulLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    shared_ptr<MatMulLayer<Dtype> > layer(
        new MatMulLayer<Dtype>(layer_param));

    GradientChecker<Dtype> checker(1e-1, 1e-2);
    checker.CheckGradientExhaustive(layer.get(),
                                    this->blob_bottom_vec_d_,
                                    this->blob_top_vec_d_);
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

TYPED_TEST(MatMulLayerTest, TestGradientRandom) {
  typedef typename TypeParam::Dtype Dtype;
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    shared_ptr<MatMulLayer<Dtype> > layer(
        new MatMulLayer<Dtype>(layer_param));

    GradientChecker<Dtype> checker(1e-1, 1e-2);
    checker.CheckGradientExhaustive(layer.get(),
                                    this->blob_bottom_vec_,
                                    this->blob_top_vec_);
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}




}  // namespace caffe
