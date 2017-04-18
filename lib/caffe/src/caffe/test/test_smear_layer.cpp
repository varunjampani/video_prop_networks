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
class SmearLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  SmearLayerTest()
      : blob_bottom_0_(new Blob<Dtype>(1, 2, 1, 10)),
        blob_bottom_1_(new Blob<Dtype>(1, 1, 3, 3)),
        blob_bottom_0_2_(new Blob<Dtype>(2, 1, 1, 10)),
        blob_bottom_1_2_(new Blob<Dtype>(2, 1, 3, 3)),
        blob_top_(new Blob<Dtype>()) {
    blob_bottom_vec_.push_back(blob_bottom_0_);
    blob_bottom_vec_.push_back(blob_bottom_1_);
    blob_bottom_vec_2_.push_back(blob_bottom_0_2_);
    blob_bottom_vec_2_.push_back(blob_bottom_1_2_);
    blob_top_vec_.push_back(blob_top_);

    for (int i = 0; i < blob_bottom_0_->count(); i++){
      blob_bottom_0_->mutable_cpu_data()[i] = i * 23;
    }

    blob_bottom_1_->mutable_cpu_data()[0] = 3;
    blob_bottom_1_->mutable_cpu_data()[1] = 2;
    blob_bottom_1_->mutable_cpu_data()[2] = 2;
    blob_bottom_1_->mutable_cpu_data()[3] = 1;
    blob_bottom_1_->mutable_cpu_data()[4] = 0;
    blob_bottom_1_->mutable_cpu_data()[5] = 9;
    blob_bottom_1_->mutable_cpu_data()[6] = 4;
    blob_bottom_1_->mutable_cpu_data()[7] = 1;
    blob_bottom_1_->mutable_cpu_data()[8] = 5;

    for (int i = 0; i < blob_bottom_0_2_->count(); i++){
      blob_bottom_0_2_->mutable_cpu_data()[i] = i * 23;
    }

    blob_bottom_1_2_->mutable_cpu_data()[0] = 3;
    blob_bottom_1_2_->mutable_cpu_data()[1] = 2;
    blob_bottom_1_2_->mutable_cpu_data()[2] = 2;
    blob_bottom_1_2_->mutable_cpu_data()[3] = 1;
    blob_bottom_1_2_->mutable_cpu_data()[4] = 0;
    blob_bottom_1_2_->mutable_cpu_data()[5] = 9;
    blob_bottom_1_2_->mutable_cpu_data()[6] = 4;
    blob_bottom_1_2_->mutable_cpu_data()[7] = 1;
    blob_bottom_1_2_->mutable_cpu_data()[8] = 5;

    blob_bottom_1_2_->mutable_cpu_data()[9] = 1;
    blob_bottom_1_2_->mutable_cpu_data()[10] = 0;
    blob_bottom_1_2_->mutable_cpu_data()[11] = 0;
    blob_bottom_1_2_->mutable_cpu_data()[12] = 2;
    blob_bottom_1_2_->mutable_cpu_data()[13] = 1;
    blob_bottom_1_2_->mutable_cpu_data()[14] = 3;
    blob_bottom_1_2_->mutable_cpu_data()[15] = 5;
    blob_bottom_1_2_->mutable_cpu_data()[16] = 4;
    blob_bottom_1_2_->mutable_cpu_data()[17] = 4;
  }
  virtual ~SmearLayerTest() {
    delete blob_bottom_0_;
    delete blob_bottom_1_;
    delete blob_bottom_0_2_;
    delete blob_bottom_1_2_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_0_;
  Blob<Dtype>* const blob_bottom_1_;
  Blob<Dtype>* const blob_bottom_0_2_;
  Blob<Dtype>* const blob_bottom_1_2_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_bottom_vec_2_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(SmearLayerTest, TestDtypesAndDevices);

TYPED_TEST(SmearLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  shared_ptr<SmearLayer<Dtype> > layer(
      new SmearLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 2);
  EXPECT_EQ(this->blob_top_->height(), 3);
  EXPECT_EQ(this->blob_top_->width(), 3);
}

//
TYPED_TEST(SmearLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    shared_ptr<SmearLayer<Dtype> > layer(
        new SmearLayer<Dtype>(layer_param));

    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype* data = this->blob_top_->cpu_data();

    // for every top blob element ...
    EXPECT_NEAR(data[0], 3 * 23, 1e-4);
    EXPECT_NEAR(data[1], 2 * 23, 1e-4);
    EXPECT_NEAR(data[2], 2 * 23, 1e-4);
    EXPECT_NEAR(data[3], 1 * 23, 1e-4);
    EXPECT_NEAR(data[4], 0 * 23, 1e-4);
    EXPECT_NEAR(data[5], 9 * 23, 1e-4);
    EXPECT_NEAR(data[6], 4 * 23, 1e-4);
    EXPECT_NEAR(data[7], 1 * 23, 1e-4);
    EXPECT_NEAR(data[8], 5 * 23, 1e-4);

    EXPECT_NEAR(data[9], 13 * 23, 1e-4);
    EXPECT_NEAR(data[10], 12 * 23, 1e-4);
    EXPECT_NEAR(data[11], 12 * 23, 1e-4);
    EXPECT_NEAR(data[12], 11 * 23, 1e-4);
    EXPECT_NEAR(data[13], 10 * 23, 1e-4);
    EXPECT_NEAR(data[14], 19 * 23, 1e-4);
    EXPECT_NEAR(data[15], 14 * 23, 1e-4);
    EXPECT_NEAR(data[16], 11 * 23, 1e-4);
    EXPECT_NEAR(data[17], 15 * 23, 1e-4);
}

TYPED_TEST(SmearLayerTest, TestForward2) {
  typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    shared_ptr<SmearLayer<Dtype> > layer(
        new SmearLayer<Dtype>(layer_param));

    layer->SetUp(this->blob_bottom_vec_2_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_2_, this->blob_top_vec_);
    const Dtype* data = this->blob_top_->cpu_data();

    // for every top blob element ...
    EXPECT_NEAR(data[0], 3 * 23, 1e-4);
    EXPECT_NEAR(data[1], 2 * 23, 1e-4);
    EXPECT_NEAR(data[2], 2 * 23, 1e-4);
    EXPECT_NEAR(data[3], 1 * 23, 1e-4);
    EXPECT_NEAR(data[4], 0 * 23, 1e-4);
    EXPECT_NEAR(data[5], 9 * 23, 1e-4);
    EXPECT_NEAR(data[6], 4 * 23, 1e-4);
    EXPECT_NEAR(data[7], 1 * 23, 1e-4);
    EXPECT_NEAR(data[8], 5 * 23, 1e-4);

    EXPECT_NEAR(data[9], 11 * 23, 1e-4);
    EXPECT_NEAR(data[10], 10 * 23, 1e-4);
    EXPECT_NEAR(data[11], 10 * 23, 1e-4);
    EXPECT_NEAR(data[12], 12 * 23, 1e-4);
    EXPECT_NEAR(data[13], 11 * 23, 1e-4);
    EXPECT_NEAR(data[14], 13 * 23, 1e-4);
    EXPECT_NEAR(data[15], 15 * 23, 1e-4);
    EXPECT_NEAR(data[16], 14 * 23, 1e-4);
    EXPECT_NEAR(data[17], 14 * 23, 1e-4);
}

TYPED_TEST(SmearLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SmearLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

TYPED_TEST(SmearLayerTest, TestGradient2) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SmearLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_2_,
      this->blob_top_vec_, 0);
}

}  // namespace caffe
