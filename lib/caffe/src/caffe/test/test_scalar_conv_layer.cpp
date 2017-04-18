#include <algorithm>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/malabar_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class ScalarConvLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  ScalarConvLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 4, 5)),
        blob_top_(new Blob<Dtype>()) {
    Caffe::set_random_seed(1701);
    FillerParameter filler_param;
    filler_param.set_min(1);
    filler_param.set_max(10);
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~ScalarConvLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(ScalarConvLayerTest, TestDtypesAndDevices);

TYPED_TEST(ScalarConvLayerTest, TestForwardScalar) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ScalarConvParameter* scalar_conv_param =
      layer_param.mutable_scalar_conv_param();
  scalar_conv_param->set_axis(2);
  scalar_conv_param->mutable_weight_filler()->set_type("constant");
  scalar_conv_param->mutable_weight_filler()->set_value(3);
  shared_ptr<ScalarConvLayer<Dtype> > layer(
      new ScalarConvLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_EQ(this->blob_bottom_->shape(), this->blob_top_->shape());
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* data = this->blob_top_->cpu_data();
  const int count = this->blob_top_->count();
  const Dtype* in_data = this->blob_bottom_->cpu_data();
  for (int i = 0; i < count; ++i) {
    EXPECT_EQ(data[i], in_data[i] * 3);
  }
}

TYPED_TEST(ScalarConvLayerTest, TestForwardScalarAxis2) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ScalarConvParameter* scalar_conv_param =
      layer_param.mutable_scalar_conv_param();
  scalar_conv_param->set_axis(2);
  scalar_conv_param->mutable_weight_filler()->set_type("constant");
  scalar_conv_param->mutable_weight_filler()->set_value(5);
  shared_ptr<ScalarConvLayer<Dtype> > layer(
      new ScalarConvLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_EQ(this->blob_bottom_->shape(), this->blob_top_->shape());
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* data = this->blob_top_->cpu_data();
  const int count = this->blob_top_->count();
  const Dtype* in_data = this->blob_bottom_->cpu_data();
  for (int i = 0; i < count; ++i) {
    EXPECT_EQ(data[i], in_data[i] * 5);
  }
}

TYPED_TEST(ScalarConvLayerTest, TestGradientScalar) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ScalarConvLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(ScalarConvLayerTest, TestGradientScalarAxis2) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_scalar_param()->set_axis(2);
  ScalarConvLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

}  // namespace caffe
