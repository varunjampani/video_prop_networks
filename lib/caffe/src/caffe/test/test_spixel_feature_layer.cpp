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
class SpixelFeatureLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  SpixelFeatureLayerTest()
      : blob_bottom_0_(new Blob<Dtype>(1, 2, 4, 3)),
        blob_bottom_1_(new Blob<Dtype>(1, 1, 4, 3)),
        blob_bottom_0_2_(new Blob<Dtype>(1, 1, 3, 3)),
        blob_bottom_1_2_(new Blob<Dtype>(1, 1, 3, 3)),
        blob_bottom_0_3_(new Blob<Dtype>(2, 2, 4, 3)),
        blob_bottom_1_3_(new Blob<Dtype>(2, 1, 4, 3)),
        blob_top_(new Blob<Dtype>()),
        blob_top_2_(new Blob<Dtype>()) {
    blob_bottom_vec_.push_back(blob_bottom_0_);
    blob_bottom_vec_.push_back(blob_bottom_1_);
    blob_bottom_vec_2_.push_back(blob_bottom_0_2_);
    blob_bottom_vec_2_.push_back(blob_bottom_1_2_);
    blob_bottom_vec_3_.push_back(blob_bottom_0_3_);
    blob_bottom_vec_3_.push_back(blob_bottom_1_3_);
    blob_top_vec_.push_back(blob_top_);
    blob_top_vec_.push_back(blob_top_2_);

    for (int i = 0; i < blob_bottom_0_->count(); i++){
      blob_bottom_0_->mutable_cpu_data()[i] = i * 10;
    }

    blob_bottom_1_->mutable_cpu_data()[0] = 1;
    blob_bottom_1_->mutable_cpu_data()[1] = 0;
    blob_bottom_1_->mutable_cpu_data()[2] = 2;
    blob_bottom_1_->mutable_cpu_data()[3] = 1;
    blob_bottom_1_->mutable_cpu_data()[4] = 0;
    blob_bottom_1_->mutable_cpu_data()[5] = 0;
    blob_bottom_1_->mutable_cpu_data()[6] = 3;
    blob_bottom_1_->mutable_cpu_data()[7] = 3;
    blob_bottom_1_->mutable_cpu_data()[8] = 3;
    blob_bottom_1_->mutable_cpu_data()[9] = 0;
    blob_bottom_1_->mutable_cpu_data()[10] = 1;
    blob_bottom_1_->mutable_cpu_data()[11] = 1;


    for (int i = 0; i < blob_bottom_0_2_->count(); i++){
      blob_bottom_0_2_->mutable_cpu_data()[i] = 10;
    }

    blob_bottom_1_2_->mutable_cpu_data()[0] = 0;
    blob_bottom_1_2_->mutable_cpu_data()[1] = 1;
    blob_bottom_1_2_->mutable_cpu_data()[2] = -1;
    blob_bottom_1_2_->mutable_cpu_data()[3] = 2;
    blob_bottom_1_2_->mutable_cpu_data()[4] = 3;
    blob_bottom_1_2_->mutable_cpu_data()[5] = -1;
    blob_bottom_1_2_->mutable_cpu_data()[6] = -1;
    blob_bottom_1_2_->mutable_cpu_data()[7] = -1;
    blob_bottom_1_2_->mutable_cpu_data()[8] = -1;


    for (int i = 0; i < blob_bottom_0_3_->count(); i++){
      blob_bottom_0_3_->mutable_cpu_data()[i] = (i % 24) * 10;
    }

    blob_bottom_1_3_->mutable_cpu_data()[0] = 1;
    blob_bottom_1_3_->mutable_cpu_data()[1] = 0;
    blob_bottom_1_3_->mutable_cpu_data()[2] = 2;
    blob_bottom_1_3_->mutable_cpu_data()[3] = 1;
    blob_bottom_1_3_->mutable_cpu_data()[4] = 0;
    blob_bottom_1_3_->mutable_cpu_data()[5] = 0;
    blob_bottom_1_3_->mutable_cpu_data()[6] = 3;
    blob_bottom_1_3_->mutable_cpu_data()[7] = 3;
    blob_bottom_1_3_->mutable_cpu_data()[8] = 3;
    blob_bottom_1_3_->mutable_cpu_data()[9] = 0;
    blob_bottom_1_3_->mutable_cpu_data()[10] = 1;
    blob_bottom_1_3_->mutable_cpu_data()[11] = 1;

    blob_bottom_1_3_->mutable_cpu_data()[12] = 1;
    blob_bottom_1_3_->mutable_cpu_data()[13] = 0;
    blob_bottom_1_3_->mutable_cpu_data()[14] = 2;
    blob_bottom_1_3_->mutable_cpu_data()[15] = 1;
    blob_bottom_1_3_->mutable_cpu_data()[16] = 0;
    blob_bottom_1_3_->mutable_cpu_data()[17] = 0;
    blob_bottom_1_3_->mutable_cpu_data()[18] = 3;
    blob_bottom_1_3_->mutable_cpu_data()[19] = 3;
    blob_bottom_1_3_->mutable_cpu_data()[20] = 3;
    blob_bottom_1_3_->mutable_cpu_data()[21] = 0;
    blob_bottom_1_3_->mutable_cpu_data()[22] = 1;
    blob_bottom_1_3_->mutable_cpu_data()[23] = 1;
  }
  virtual ~SpixelFeatureLayerTest() {
    delete blob_bottom_0_;
    delete blob_bottom_1_;
    delete blob_bottom_0_2_;
    delete blob_bottom_1_2_;
    delete blob_bottom_0_3_;
    delete blob_bottom_1_3_;
    delete blob_top_;
    delete blob_top_2_;
  }
  Blob<Dtype>* const blob_bottom_0_;
  Blob<Dtype>* const blob_bottom_1_;
  Blob<Dtype>* const blob_bottom_0_2_;
  Blob<Dtype>* const blob_bottom_1_2_;
  Blob<Dtype>* const blob_bottom_0_3_;
  Blob<Dtype>* const blob_bottom_1_3_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const blob_top_2_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_bottom_vec_2_;
  vector<Blob<Dtype>*> blob_bottom_vec_3_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(SpixelFeatureLayerTest, TestDtypesAndDevices);

TYPED_TEST(SpixelFeatureLayerTest, TestSetUp1) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SpixelFeatureParameter* spixel_param =
    layer_param.mutable_spixel_feature_param();
  spixel_param->set_max_spixels(10);
  shared_ptr<SpixelFeatureLayer<Dtype> > layer(
      new SpixelFeatureLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 2);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 10);

  EXPECT_EQ(this->blob_top_2_->num(), 1);
  EXPECT_EQ(this->blob_top_2_->channels(), 2);
  EXPECT_EQ(this->blob_top_2_->height(), 4);
  EXPECT_EQ(this->blob_top_2_->width(), 3);
}

TYPED_TEST(SpixelFeatureLayerTest, TestSetUp2) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SpixelFeatureParameter* spixel_param =
    layer_param.mutable_spixel_feature_param();
  spixel_param->set_max_spixels(10);
  spixel_param->set_type(SpixelFeatureParameter_Feature_AVGXYRGBXY);
  shared_ptr<SpixelFeatureLayer<Dtype> > layer(
      new SpixelFeatureLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 6);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 10);

  EXPECT_EQ(this->blob_top_2_->num(), 1);
  EXPECT_EQ(this->blob_top_2_->channels(), 6);
  EXPECT_EQ(this->blob_top_2_->height(), 4);
  EXPECT_EQ(this->blob_top_2_->width(), 3);
}

TYPED_TEST(SpixelFeatureLayerTest, TestSetUp3) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SpixelFeatureParameter* spixel_param =
    layer_param.mutable_spixel_feature_param();
  spixel_param->set_max_spixels(10);
  spixel_param->set_type(SpixelFeatureParameter_Feature_AVGXYRGBXY);
  shared_ptr<SpixelFeatureLayer<Dtype> > layer(
      new SpixelFeatureLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_3_, this->blob_top_vec_);

  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 6);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 10);

  EXPECT_EQ(this->blob_top_2_->num(), 2);
  EXPECT_EQ(this->blob_top_2_->channels(), 6);
  EXPECT_EQ(this->blob_top_2_->height(), 4);
  EXPECT_EQ(this->blob_top_2_->width(), 3);
}

//
TYPED_TEST(SpixelFeatureLayerTest, TestForwardXY) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SpixelFeatureParameter* spixel_param =
    layer_param.mutable_spixel_feature_param();
  spixel_param->set_max_spixels(5);
  shared_ptr<SpixelFeatureLayer<Dtype> > layer(
      new SpixelFeatureLayer<Dtype>(layer_param));

  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* data = this->blob_top_->cpu_data();

  // for every top blob element ...
  EXPECT_NEAR(data[0], 5./4., 1e-4);
  EXPECT_NEAR(data[1], 7./4., 1e-4);
  EXPECT_NEAR(data[2], 0./1., 1e-4);
  EXPECT_NEAR(data[3], 6./3., 1e-4);
  EXPECT_NEAR(data[4], -1000, 1e-4);

  EXPECT_NEAR(data[5], 4./4., 1e-4);
  EXPECT_NEAR(data[6], 3./4., 1e-4);
  EXPECT_NEAR(data[7], 2./1., 1e-4);
  EXPECT_NEAR(data[8], 3./3., 1e-4);
  EXPECT_NEAR(data[9], -1000, 1e-4);
}


TYPED_TEST(SpixelFeatureLayerTest, TestForwardRGBXY) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SpixelFeatureParameter* spixel_param =
    layer_param.mutable_spixel_feature_param();
  spixel_param->set_max_spixels(5);
  spixel_param->set_type(SpixelFeatureParameter_Feature_AVGRGBXY);
  shared_ptr<SpixelFeatureLayer<Dtype> > layer(
      new SpixelFeatureLayer<Dtype>(layer_param));

  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* data = this->blob_top_->cpu_data();

  // for every top blob element ...
  EXPECT_NEAR(data[0], 190./4., 1e-4);
  EXPECT_NEAR(data[1], 240./4., 1e-4);
  EXPECT_NEAR(data[2], 20./1., 1e-4);
  EXPECT_NEAR(data[3], 210./3., 1e-4);
  EXPECT_NEAR(data[4], -1000, 1e-4);

  EXPECT_NEAR(data[5], 670./4., 1e-4);
  EXPECT_NEAR(data[6], 720./4., 1e-4);
  EXPECT_NEAR(data[7], 140./1., 1e-4);
  EXPECT_NEAR(data[8], 570./3., 1e-4);
  EXPECT_NEAR(data[9], -1000, 1e-4);

  EXPECT_NEAR(data[10], 5./4., 1e-4);
  EXPECT_NEAR(data[11], 7./4., 1e-4);
  EXPECT_NEAR(data[12], 0./1., 1e-4);
  EXPECT_NEAR(data[13], 6./3., 1e-4);
  EXPECT_NEAR(data[14], -1000, 1e-4);

  EXPECT_NEAR(data[15], 4./4., 1e-4);
  EXPECT_NEAR(data[16], 3./4., 1e-4);
  EXPECT_NEAR(data[17], 2./1., 1e-4);
  EXPECT_NEAR(data[18], 3./3., 1e-4);
  EXPECT_NEAR(data[19], -1000, 1e-4);

  const Dtype* data2 = this->blob_top_2_->cpu_data();

  EXPECT_NEAR(data2[3], 240./4., 1e-4);
  EXPECT_NEAR(data2[8], 210./3., 1e-4);
  EXPECT_NEAR(data2[20], 570./3. , 1e-4);
  EXPECT_NEAR(data2[34], 7./4., 1e-4);
  EXPECT_NEAR(data2[45], 4./4., 1e-4);

}

TYPED_TEST(SpixelFeatureLayerTest, TestForwardRGBXYScale) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SpixelFeatureParameter* spixel_param =
    layer_param.mutable_spixel_feature_param();
  spixel_param->set_max_spixels(5);
  spixel_param->set_rgbxy_rgb_scale(-3);
  spixel_param->set_rgbxy_xy_scale(7);
  spixel_param->set_type(SpixelFeatureParameter_Feature_AVGRGBXY);
  shared_ptr<SpixelFeatureLayer<Dtype> > layer(
      new SpixelFeatureLayer<Dtype>(layer_param));

  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* data = this->blob_top_->cpu_data();

  // for every top blob element ...
  EXPECT_NEAR(data[0], -3 * 190./4., 1e-4);
  EXPECT_NEAR(data[1], -3 * 240./4., 1e-4);
  EXPECT_NEAR(data[2], -3 * 20./1., 1e-4);
  EXPECT_NEAR(data[3], -3 * 210./3., 1e-4);
  EXPECT_NEAR(data[4], -1000, 1e-4);

  EXPECT_NEAR(data[5], -3 * 670./4., 1e-4);
  EXPECT_NEAR(data[6], -3 * 720./4., 1e-4);
  EXPECT_NEAR(data[7], -3 * 140./1., 1e-4);
  EXPECT_NEAR(data[8], -3 * 570./3., 1e-4);
  EXPECT_NEAR(data[9], -1000, 1e-4);

  EXPECT_NEAR(data[10], 7 * 5./4., 1e-4);
  EXPECT_NEAR(data[11], 7 * 7./4., 1e-4);
  EXPECT_NEAR(data[12], 7 * 0./1., 1e-4);
  EXPECT_NEAR(data[13], 7 * 6./3., 1e-4);
  EXPECT_NEAR(data[14], -1000, 1e-4);

  EXPECT_NEAR(data[15], 7 * 4./4., 1e-4);
  EXPECT_NEAR(data[16], 7 * 3./4., 1e-4);
  EXPECT_NEAR(data[17], 7 * 2./1., 1e-4);
  EXPECT_NEAR(data[18], 7 * 3./3., 1e-4);
  EXPECT_NEAR(data[19], -1000, 1e-4);
}

TYPED_TEST(SpixelFeatureLayerTest, TestForwardRGBScale) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SpixelFeatureParameter* spixel_param =
    layer_param.mutable_spixel_feature_param();
  spixel_param->set_max_spixels(5);
  spixel_param->set_rgb_scale(-3);
  spixel_param->set_type(SpixelFeatureParameter_Feature_AVGRGB);
  shared_ptr<SpixelFeatureLayer<Dtype> > layer(
      new SpixelFeatureLayer<Dtype>(layer_param));

  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* data = this->blob_top_->cpu_data();

  // for every top blob element ...
  EXPECT_NEAR(data[0], -3 * 190./4., 1e-4);
  EXPECT_NEAR(data[1], -3 * 240./4., 1e-4);
  EXPECT_NEAR(data[2], -3 * 20./1., 1e-4);
  EXPECT_NEAR(data[3], -3 * 210./3., 1e-4);
  EXPECT_NEAR(data[4], -1000, 1e-4);

  EXPECT_NEAR(data[5], -3 * 670./4., 1e-4);
  EXPECT_NEAR(data[6], -3 * 720./4., 1e-4);
  EXPECT_NEAR(data[7], -3 * 140./1., 1e-4);
  EXPECT_NEAR(data[8], -3 * 570./3., 1e-4);
  EXPECT_NEAR(data[9], -1000, 1e-4);

}

TYPED_TEST(SpixelFeatureLayerTest, TestForwardRGBScale2) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SpixelFeatureParameter* spixel_param =
    layer_param.mutable_spixel_feature_param();
  spixel_param->set_max_spixels(5);
  spixel_param->set_rgb_scale(-3);
  spixel_param->set_type(SpixelFeatureParameter_Feature_AVGRGB);
  shared_ptr<SpixelFeatureLayer<Dtype> > layer(
      new SpixelFeatureLayer<Dtype>(layer_param));

  layer->SetUp(this->blob_bottom_vec_3_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_3_, this->blob_top_vec_);
  const Dtype* data = this->blob_top_->cpu_data();

  // for every top blob element ...
  EXPECT_NEAR(data[0], -3 * 190./4., 1e-4);
  EXPECT_NEAR(data[1], -3 * 240./4., 1e-4);
  EXPECT_NEAR(data[2], -3 * 20./1., 1e-4);
  EXPECT_NEAR(data[3], -3 * 210./3., 1e-4);
  EXPECT_NEAR(data[4], -1000, 1e-4);

  EXPECT_NEAR(data[5], -3 * 670./4., 1e-4);
  EXPECT_NEAR(data[6], -3 * 720./4., 1e-4);
  EXPECT_NEAR(data[7], -3 * 140./1., 1e-4);
  EXPECT_NEAR(data[8], -3 * 570./3., 1e-4);
  EXPECT_NEAR(data[9], -1000, 1e-4);

  EXPECT_NEAR(data[10], -3 * 190./4., 1e-4);
  EXPECT_NEAR(data[11], -3 * 240./4., 1e-4);
  EXPECT_NEAR(data[12], -3 * 20./1., 1e-4);
  EXPECT_NEAR(data[13], -3 * 210./3., 1e-4);
  EXPECT_NEAR(data[14], -1000, 1e-4);

  EXPECT_NEAR(data[15], -3 * 670./4., 1e-4);
  EXPECT_NEAR(data[16], -3 * 720./4., 1e-4);
  EXPECT_NEAR(data[17], -3 * 140./1., 1e-4);
  EXPECT_NEAR(data[18], -3 * 570./3., 1e-4);
  EXPECT_NEAR(data[19], -1000, 1e-4);

}

TYPED_TEST(SpixelFeatureLayerTest, TestForwardXYScale) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SpixelFeatureParameter* spixel_param =
    layer_param.mutable_spixel_feature_param();
  spixel_param->set_max_spixels(5);
  spixel_param->set_xy_scale(-4.2);
  shared_ptr<SpixelFeatureLayer<Dtype> > layer(
      new SpixelFeatureLayer<Dtype>(layer_param));

  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* data = this->blob_top_->cpu_data();

  // for every top blob element ...
  EXPECT_NEAR(data[0], -4.2 * 5./4., 1e-4);
  EXPECT_NEAR(data[1], -4.2 * 7./4., 1e-4);
  EXPECT_NEAR(data[2], -4.2 * 0./1., 1e-4);
  EXPECT_NEAR(data[3], -4.2 * 6./3., 1e-4);
  EXPECT_NEAR(data[4], -1000, 1e-4);

  EXPECT_NEAR(data[5], -4.2 * 4./4., 1e-4);
  EXPECT_NEAR(data[6], -4.2 * 3./4., 1e-4);
  EXPECT_NEAR(data[7], -4.2 * 2./1., 1e-4);
  EXPECT_NEAR(data[8], -4.2 * 3./3., 1e-4);
  EXPECT_NEAR(data[9], -1000, 1e-4);
}

TYPED_TEST(SpixelFeatureLayerTest, TestForwardXYRGBXYScale) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SpixelFeatureParameter* spixel_param =
    layer_param.mutable_spixel_feature_param();
  spixel_param->set_max_spixels(6);
  spixel_param->set_ignore_feature_value(255);
  spixel_param->set_xy_scale(0.01);
  spixel_param->set_rgbxy_rgb_scale(-3);
  spixel_param->set_rgbxy_xy_scale(7);
  spixel_param->set_type(SpixelFeatureParameter_Feature_AVGXYRGBXY);
  shared_ptr<SpixelFeatureLayer<Dtype> > layer(
      new SpixelFeatureLayer<Dtype>(layer_param));

  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* data = this->blob_top_->cpu_data();

  // for every top blob element ...
  EXPECT_NEAR(data[0], 0.01 * 5./4., 1e-4);
  EXPECT_NEAR(data[1], 0.01 * 7./4., 1e-4);
  EXPECT_NEAR(data[2], 0.01 * 0./1., 1e-4);
  EXPECT_NEAR(data[3], 0.01 * 6./3., 1e-4);
  EXPECT_NEAR(data[4], 255, 1e-4);
  EXPECT_NEAR(data[5], 255, 1e-4);

  EXPECT_NEAR(data[6], 0.01 * 4./4., 1e-4);
  EXPECT_NEAR(data[7], 0.01 * 3./4., 1e-4);
  EXPECT_NEAR(data[8], 0.01 * 2./1., 1e-4);
  EXPECT_NEAR(data[9], 0.01 * 3./3., 1e-4);
  EXPECT_NEAR(data[10], 255, 1e-4);
  EXPECT_NEAR(data[11], 255, 1e-4);

  EXPECT_NEAR(data[12], -3 * 190./4., 1e-4);
  EXPECT_NEAR(data[13], -3 * 240./4., 1e-4);
  EXPECT_NEAR(data[14], -3 * 20./1., 1e-4);
  EXPECT_NEAR(data[15], -3 * 210./3., 1e-4);
  EXPECT_NEAR(data[16], 255, 1e-4);
  EXPECT_NEAR(data[17], 255, 1e-4);

  EXPECT_NEAR(data[18], -3 * 670./4., 1e-4);
  EXPECT_NEAR(data[19], -3 * 720./4., 1e-4);
  EXPECT_NEAR(data[20], -3 * 140./1., 1e-4);
  EXPECT_NEAR(data[21], -3 * 570./3., 1e-4);
  EXPECT_NEAR(data[22], 255, 1e-4);
  EXPECT_NEAR(data[23], 255, 1e-4);

  EXPECT_NEAR(data[24], 7 * 5./4., 1e-4);
  EXPECT_NEAR(data[25], 7 * 7./4., 1e-4);
  EXPECT_NEAR(data[26], 7 * 0./1., 1e-4);
  EXPECT_NEAR(data[27], 7 * 6./3., 1e-4);
  EXPECT_NEAR(data[28], 255, 1e-4);
  EXPECT_NEAR(data[29], 255, 1e-4);

  EXPECT_NEAR(data[30], 7 * 4./4., 1e-4);
  EXPECT_NEAR(data[31], 7 * 3./4., 1e-4);
  EXPECT_NEAR(data[32], 7 * 2./1., 1e-4);
  EXPECT_NEAR(data[33], 7 * 3./3., 1e-4);
  EXPECT_NEAR(data[34], 255, 1e-4);
  EXPECT_NEAR(data[35], 255, 1e-4);
}

TYPED_TEST(SpixelFeatureLayerTest, TestForwardRGBXYRGBXYScale) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SpixelFeatureParameter* spixel_param =
    layer_param.mutable_spixel_feature_param();
  spixel_param->set_max_spixels(6);
  spixel_param->set_ignore_feature_value(255);
  spixel_param->set_xy_scale(0.01);
  spixel_param->set_rgb_scale(-5);
  spixel_param->set_rgbxy_rgb_scale(-3);
  spixel_param->set_rgbxy_xy_scale(7);
  spixel_param->set_type(SpixelFeatureParameter_Feature_AVGRGBXYRGBXY);
  shared_ptr<SpixelFeatureLayer<Dtype> > layer(
      new SpixelFeatureLayer<Dtype>(layer_param));

  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* data = this->blob_top_->cpu_data();

  // for every top blob element ...
  EXPECT_NEAR(data[0], -5 * 190./4., 1e-4);
  EXPECT_NEAR(data[1], -5 * 240./4., 1e-4);
  EXPECT_NEAR(data[2], -5 * 20./1., 1e-4);
  EXPECT_NEAR(data[3], -5 * 210./3., 1e-4);
  EXPECT_NEAR(data[4], 255, 1e-4);
  EXPECT_NEAR(data[5], 255, 1e-4);

  EXPECT_NEAR(data[6], -5 * 670./4., 1e-4);
  EXPECT_NEAR(data[7], -5 * 720./4., 1e-4);
  EXPECT_NEAR(data[8], -5 * 140./1., 1e-4);
  EXPECT_NEAR(data[9], -5 * 570./3., 1e-4);
  EXPECT_NEAR(data[10], 255, 1e-4);
  EXPECT_NEAR(data[11], 255, 1e-4);


  EXPECT_NEAR(data[12], 0.01 * 5./4., 1e-4);
  EXPECT_NEAR(data[13], 0.01 * 7./4., 1e-4);
  EXPECT_NEAR(data[14], 0.01 * 0./1., 1e-4);
  EXPECT_NEAR(data[15], 0.01 * 6./3., 1e-4);
  EXPECT_NEAR(data[16], 255, 1e-4);
  EXPECT_NEAR(data[17], 255, 1e-4);

  EXPECT_NEAR(data[18], 0.01 * 4./4., 1e-4);
  EXPECT_NEAR(data[19], 0.01 * 3./4., 1e-4);
  EXPECT_NEAR(data[20], 0.01 * 2./1., 1e-4);
  EXPECT_NEAR(data[21], 0.01 * 3./3., 1e-4);
  EXPECT_NEAR(data[22], 255, 1e-4);
  EXPECT_NEAR(data[23], 255, 1e-4);

  EXPECT_NEAR(data[24], -3 * 190./4., 1e-4);
  EXPECT_NEAR(data[25], -3 * 240./4., 1e-4);
  EXPECT_NEAR(data[26], -3 * 20./1., 1e-4);
  EXPECT_NEAR(data[27], -3 * 210./3., 1e-4);
  EXPECT_NEAR(data[28], 255, 1e-4);
  EXPECT_NEAR(data[29], 255, 1e-4);

  EXPECT_NEAR(data[30], -3 * 670./4., 1e-4);
  EXPECT_NEAR(data[31], -3 * 720./4., 1e-4);
  EXPECT_NEAR(data[32], -3 * 140./1., 1e-4);
  EXPECT_NEAR(data[33], -3 * 570./3., 1e-4);
  EXPECT_NEAR(data[34], 255, 1e-4);
  EXPECT_NEAR(data[35], 255, 1e-4);

  EXPECT_NEAR(data[36], 7 * 5./4., 1e-4);
  EXPECT_NEAR(data[37], 7 * 7./4., 1e-4);
  EXPECT_NEAR(data[38], 7 * 0./1., 1e-4);
  EXPECT_NEAR(data[39], 7 * 6./3., 1e-4);
  EXPECT_NEAR(data[40], 255, 1e-4);
  EXPECT_NEAR(data[41], 255, 1e-4);

  EXPECT_NEAR(data[42], 7 * 4./4., 1e-4);
  EXPECT_NEAR(data[43], 7 * 3./4., 1e-4);
  EXPECT_NEAR(data[44], 7 * 2./1., 1e-4);
  EXPECT_NEAR(data[45], 7 * 3./3., 1e-4);
  EXPECT_NEAR(data[46], 255, 1e-4);
  EXPECT_NEAR(data[47], 255, 1e-4);
}

TYPED_TEST(SpixelFeatureLayerTest, TestForwardRGBXYScaleIgnore) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SpixelFeatureParameter* spixel_param =
    layer_param.mutable_spixel_feature_param();
  spixel_param->set_max_spixels(5);
  spixel_param->set_ignore_idx_value(-1);
  spixel_param->set_ignore_feature_value(255);
  spixel_param->set_rgbxy_rgb_scale(-3);
  spixel_param->set_rgbxy_xy_scale(7);
  spixel_param->set_type(SpixelFeatureParameter_Feature_AVGRGBXY);
  shared_ptr<SpixelFeatureLayer<Dtype> > layer(
      new SpixelFeatureLayer<Dtype>(layer_param));

  layer->SetUp(this->blob_bottom_vec_2_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_2_, this->blob_top_vec_);
  const Dtype* data = this->blob_top_->cpu_data();

  // for every top blob element ...
  EXPECT_NEAR(data[0], -3 * 10., 1e-4);
  EXPECT_NEAR(data[1], -3 * 10., 1e-4);
  EXPECT_NEAR(data[2], -3 * 10., 1e-4);
  EXPECT_NEAR(data[3], -3 * 10., 1e-4);
  EXPECT_NEAR(data[4], 255, 1e-4);

  EXPECT_NEAR(data[5], 7 * 0., 1e-4);
  EXPECT_NEAR(data[6], 7 * 0., 1e-4);
  EXPECT_NEAR(data[7], 7 * 1., 1e-4);
  EXPECT_NEAR(data[8], 7 * 1., 1e-4);
  EXPECT_NEAR(data[9], 255, 1e-4);

  EXPECT_NEAR(data[10], 7 * 0., 1e-4);
  EXPECT_NEAR(data[11], 7 * 1., 1e-4);
  EXPECT_NEAR(data[12], 7 * 0., 1e-4);
  EXPECT_NEAR(data[13], 7 * 1., 1e-4);
  EXPECT_NEAR(data[14], 255, 1e-4);

  const Dtype* data2 = this->blob_top_2_->cpu_data();

  EXPECT_NEAR(data2[0], -3 * 10., 1e-4);
  EXPECT_NEAR(data2[1], -3 * 10., 1e-4);
  EXPECT_NEAR(data2[2], 255, 1e-4);
  EXPECT_NEAR(data2[3], -3 * 10., 1e-4);
  EXPECT_NEAR(data2[4], -3 * 10., 1e-4);
  EXPECT_NEAR(data2[5], 255, 1e-4);
  EXPECT_NEAR(data2[6], 255, 1e-4);
  EXPECT_NEAR(data2[7], 255, 1e-4);
  EXPECT_NEAR(data2[8], 255, 1e-4);

  EXPECT_NEAR(data2[9], 0., 1e-4);
  EXPECT_NEAR(data2[10], 0., 1e-4);
  EXPECT_NEAR(data2[11], 255, 1e-4);
  EXPECT_NEAR(data2[12], 7., 1e-4);
  EXPECT_NEAR(data2[13], 7., 1e-4);
  EXPECT_NEAR(data2[14], 255, 1e-4);
  EXPECT_NEAR(data2[15], 255, 1e-4);
  EXPECT_NEAR(data2[16], 255, 1e-4);
  EXPECT_NEAR(data2[17], 255, 1e-4);

  EXPECT_NEAR(data2[18], 0., 1e-4);
  EXPECT_NEAR(data2[19], 7., 1e-4);
  EXPECT_NEAR(data2[20], 255, 1e-4);
  EXPECT_NEAR(data2[21], 0., 1e-4);
  EXPECT_NEAR(data2[22], 7., 1e-4);
  EXPECT_NEAR(data2[23], 255, 1e-4);
  EXPECT_NEAR(data2[24], 255, 1e-4);
  EXPECT_NEAR(data2[25], 255, 1e-4);
  EXPECT_NEAR(data2[26], 255, 1e-4);

}

}  // namespace caffe
