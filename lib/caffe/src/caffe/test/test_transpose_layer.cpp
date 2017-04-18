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
class TransposeLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  TransposeLayerTest()
      : blob_bottom_0_(new Blob<Dtype>(1, 2, 4, 3)),
        blob_bottom_1_(new Blob<Dtype>(1, 3, 1, 4)),
        blob_top_(new Blob<Dtype>()) {
    blob_bottom_vec_0_.push_back(blob_bottom_0_);
    blob_bottom_vec_1_.push_back(blob_bottom_1_);
    blob_top_vec_.push_back(blob_top_);

    for (int i = 0; i < blob_bottom_0_->count(); i++){
      blob_bottom_0_->mutable_cpu_data()[i] = i * 23;
    }

    for (int i = 0; i < blob_bottom_1_->count(); i++){
      blob_bottom_1_->mutable_cpu_data()[i] = i;
    }
  }
  virtual ~TransposeLayerTest() {
    delete blob_bottom_0_;
    delete blob_bottom_1_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_0_;
  Blob<Dtype>* const blob_bottom_1_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_0_;
  vector<Blob<Dtype>*> blob_bottom_vec_1_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(TransposeLayerTest, TestDtypesAndDevices);

TYPED_TEST(TransposeLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  shared_ptr<TransposeLayer<Dtype> > layer(
      new TransposeLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_0_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 2);
  EXPECT_EQ(this->blob_top_->height(), 3);
  EXPECT_EQ(this->blob_top_->width(), 4);
}

TYPED_TEST(TransposeLayerTest, TestSetUp2) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  shared_ptr<TransposeLayer<Dtype> > layer(
      new TransposeLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_1_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 3);
  EXPECT_EQ(this->blob_top_->height(), 4);
  EXPECT_EQ(this->blob_top_->width(), 1);
}

//
TYPED_TEST(TransposeLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    shared_ptr<TransposeLayer<Dtype> > layer(
        new TransposeLayer<Dtype>(layer_param));

    layer->SetUp(this->blob_bottom_vec_0_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_0_, this->blob_top_vec_);
    const Dtype* data = this->blob_top_->cpu_data();

    // for every top blob element ...
    for (int i = 0; i < 2; i++){
      for (int j = 0; j < 4; j++) {
        for (int k = 0; k < 3; k++) {
          EXPECT_NEAR(data[(i * 3 + k) * 4 + j],
            this->blob_bottom_0_->mutable_cpu_data()[(i * 4 + j) * 3 + k] , 1e-4);
        }
      }
    }
}

TYPED_TEST(TransposeLayerTest, TestForward2) {
  typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    shared_ptr<TransposeLayer<Dtype> > layer(
        new TransposeLayer<Dtype>(layer_param));

    layer->SetUp(this->blob_bottom_vec_1_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_1_, this->blob_top_vec_);
    const Dtype* data = this->blob_top_->cpu_data();

    // for every top blob element ...
    for (int i = 0; i < 3; i++){
      for (int j = 0; j < 1; j++) {
        for (int k = 0; k < 4; k++) {
          EXPECT_NEAR(data[(i * 4 + k) * 1 + j],
            this->blob_bottom_1_->mutable_cpu_data()[(i * 1 + j) * 4 + k] , 1e-4);
        }
      }
    }
}

TYPED_TEST(TransposeLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  TransposeLayer<Dtype> layer(layer_param);

  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_1_,
      this->blob_top_vec_);
}

}  // namespace caffe
