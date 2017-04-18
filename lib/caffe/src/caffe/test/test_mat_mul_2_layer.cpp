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
class MatMul2LayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  MatMul2LayerTest()
      : blob_bottom_0_(new Blob<Dtype>(8, 16, 3, 4)),
        blob_bottom_1_(new Blob<Dtype>(8, 1, 2, 4)),
        blob_bottom_0_d_(new Blob<Dtype>(1, 1, 2, 3)),
        blob_bottom_1_d_(new Blob<Dtype>(1, 1, 3, 3)),
        blob_top_(new Blob<Dtype>()),
        blob_top_d_(new Blob<Dtype>()) {
    Caffe::set_random_seed(1701);
    // fill the values
    FillerParameter filler_param;
    filler_param.set_value(1.);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_0_);
    filler.Fill(this->blob_bottom_1_);
    blob_bottom_vec_.push_back(blob_bottom_0_);
    blob_bottom_vec_.push_back(blob_bottom_1_);
    blob_top_vec_.push_back(blob_top_);
    blob_top_vec_d_.push_back(blob_top_d_);
    for (int i = 0; i < blob_bottom_0_d_->count(); i++){
      blob_bottom_0_d_->mutable_cpu_data()[i] = i;
    }
    for (int i = 0; i < blob_bottom_1_d_->count(); i++){
      blob_bottom_1_d_->mutable_cpu_data()[i] = blob_bottom_0_d_->count() + i;
    }

    blob_bottom_vec_d_.push_back(blob_bottom_0_d_);
    blob_bottom_vec_d_.push_back(blob_bottom_1_d_);
  }
  virtual ~MatMul2LayerTest() {
    delete blob_bottom_0_;
    delete blob_bottom_1_;
    delete blob_bottom_0_d_;
    delete blob_bottom_1_d_;
    delete blob_top_;
    delete blob_top_d_;
  }
  Blob<Dtype>* const blob_bottom_0_;
  Blob<Dtype>* const blob_bottom_1_;
  Blob<Dtype>* const blob_bottom_0_d_;
  Blob<Dtype>* const blob_bottom_1_d_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const blob_top_d_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_bottom_vec_d_;
  vector<Blob<Dtype>*> blob_top_vec_;
  vector<Blob<Dtype>*> blob_top_vec_d_;
};

TYPED_TEST_CASE(MatMul2LayerTest, TestDtypesAndDevices);

TYPED_TEST(MatMul2LayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  shared_ptr<MatMul2Layer<Dtype> > layer(
      new MatMul2Layer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 8);
  EXPECT_EQ(this->blob_top_->channels(), 16);
  EXPECT_EQ(this->blob_top_->height(), 3);
  EXPECT_EQ(this->blob_top_->width(), 2);
}

//
TYPED_TEST(MatMul2LayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    shared_ptr<MatMul2Layer<Dtype> > layer(
        new MatMul2Layer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_d_, this->blob_top_vec_d_);
    layer->Forward(this->blob_bottom_vec_d_, this->blob_top_vec_d_);
    const Dtype* data = this->blob_top_d_->cpu_data();

    const int num = this->blob_top_d_->num();
    const int channels = this->blob_top_d_->channels();
    const int height = this->blob_top_d_->height();
    const int width = this->blob_top_d_->width();

    const Dtype* data_0 = this->blob_bottom_0_d_->cpu_data();
    const Dtype* data_1 = this->blob_bottom_1_d_->cpu_data();
    const int in_x = this->blob_bottom_0_d_->width();

    // for every top blob element ...
    for (int n = 0; n < num; ++n) {
      for (int c = 0; c < channels; ++c) {
        for (int y = 0; y < height; ++y) {
          for (int k = 0; k < width; ++k) {
            // ... compute the result by looping

            Dtype res = 0;
            for (int x = 0; x < in_x; ++x) {
              const int i = this->blob_bottom_0_d_->offset(n, c, y, x);
              const int j = this->blob_bottom_1_d_->offset(n, 0, k, x);
              res += data_0[i] * data_1[j];
            }

            const int i = this->blob_top_d_->offset(n, c, y, k);
            EXPECT_NEAR(data[i], res, 1e-4);
            // std::cout << "res,data = " << res << "," << data[i] << std::endl;
          }
        }
      }
    }
  } else {
    LOG(ERROR) << "test failed!.";
  }
}
//
TYPED_TEST(MatMul2LayerTest, TestForwardRandom) {
  typedef typename TypeParam::Dtype Dtype;
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    shared_ptr<MatMul2Layer<Dtype> > layer(
        new MatMul2Layer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype* data = this->blob_top_->cpu_data();

    const int num = this->blob_top_->num();
    const int channels = this->blob_top_->channels();
    const int height = this->blob_top_->height();
    const int width = this->blob_top_->width();

    const Dtype* data_0 = this->blob_bottom_0_->cpu_data();
    const Dtype* data_1 = this->blob_bottom_1_->cpu_data();
    const int in_x = this->blob_bottom_0_->width();

    // for every top blob element ...
    for (int n = 0; n < num; ++n) {
      for (int c = 0; c < channels; ++c) {
        for (int y = 0; y < height; ++y) {
          for (int k = 0; k < width; ++k) {
            // ... compute the result by looping
            Dtype res = 0;
            for (int x = 0; x < in_x; ++x) {
              const int i = this->blob_bottom_0_->offset(n, c, y, x);
              const int j = this->blob_bottom_1_->offset(n, 0, k, x);
              res += data_0[i] * data_1[j];
            }

            const int i = this->blob_top_->offset(n, c, y, k);
            EXPECT_NEAR(data[i], res, 1e-4);
          }
        }
      }
    }
  } else {
    LOG(ERROR) << "test failed!.";
  }
}

TYPED_TEST(MatMul2LayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    shared_ptr<MatMul2Layer<Dtype> > layer(
        new MatMul2Layer<Dtype>(layer_param));

    GradientChecker<Dtype> checker(1e-2, 1e-3);
    checker.CheckGradientExhaustive(layer.get(), this->blob_bottom_vec_d_,
                                    this->blob_top_vec_d_);
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

TYPED_TEST(MatMul2LayerTest, TestGradientRandom) {
  typedef typename TypeParam::Dtype Dtype;
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    shared_ptr<MatMul2Layer<Dtype> > layer(
        new MatMul2Layer<Dtype>(layer_param));

    GradientChecker<Dtype> checker(1e-2, 1e-3);
    checker.CheckGradientExhaustive(layer.get(), this->blob_bottom_vec_,
                                    this->blob_top_vec_);
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}




}  // namespace caffe
