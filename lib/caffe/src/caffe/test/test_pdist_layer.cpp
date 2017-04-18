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
class PdistLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  PdistLayerTest()
      : blob_bottom_0_(new Blob<Dtype>(1, 2, 3, 2)),
        blob_bottom_1_(new Blob<Dtype>(1, 2, 1, 3)),
        blob_top_(new Blob<Dtype>()) {
    blob_bottom_vec_.push_back(blob_bottom_0_);
    blob_bottom_vec_.push_back(blob_bottom_1_);
    blob_top_vec_.push_back(blob_top_);

    for (int i = 0; i < blob_bottom_0_->count(); i++){
      blob_bottom_0_->mutable_cpu_data()[i] = i;
    }

    for (int i = 0; i < blob_bottom_1_->count(); i++){
      blob_bottom_1_->mutable_cpu_data()[i] = i;
    }
  }
  virtual ~PdistLayerTest() {
    delete blob_bottom_0_;
    delete blob_bottom_1_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_0_;
  Blob<Dtype>* const blob_bottom_1_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(PdistLayerTest, TestDtypesAndDevices);

TYPED_TEST(PdistLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  shared_ptr<PdistLayer<Dtype> > layer(
      new PdistLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 1);
  EXPECT_EQ(this->blob_top_->height(), 3);
  EXPECT_EQ(this->blob_top_->width(), 6);
}

TYPED_TEST(PdistLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    shared_ptr<PdistLayer<Dtype> > layer(
        new PdistLayer<Dtype>(layer_param));

    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype* data = this->blob_top_->cpu_data();

    // for every top blob element ...
    for (int t = 0; t < 3; t++) {
      for (int s = 0; s < 6; s++) {
        EXPECT_NEAR(data[t * 6 + s], pow(s - t, 2) + pow(6 + s - (3 + t), 2) , 1e-4);
      }
    }
}

TYPED_TEST(PdistLayerTest, TestForwardScale) {
  typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    PdistParameter* pdist_param =
      layer_param.mutable_pdist_param();
    pdist_param->set_scale_value(-2.0);

    shared_ptr<PdistLayer<Dtype> > layer(
        new PdistLayer<Dtype>(layer_param));

    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype* data = this->blob_top_->cpu_data();

    // for every top blob element ...
    for (int t = 0; t < 3; t++) {
      for (int s = 0; s < 6; s++) {
        EXPECT_NEAR(data[t * 6 + s], -2.0 * (pow(s - t, 2) + pow(6 + s - (3 + t), 2)) , 1e-4);
      }
    }
}

TYPED_TEST(PdistLayerTest, TestForwardIgnore) {
  typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    PdistParameter* pdist_param =
      layer_param.mutable_pdist_param();
    pdist_param->set_scale_value(-2.0);
    pdist_param->set_ignore_value(5.0);
    shared_ptr<PdistLayer<Dtype> > layer(
        new PdistLayer<Dtype>(layer_param));

    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype* data = this->blob_top_->cpu_data();

    // for every top blob element ...
    for (int t = 0; t < 3; t++) {
      for (int s = 0; s < 6; s++) {
        if (s == 5 || 6 + s == 5 || t == 5 || 3 + t == 5) {
          EXPECT_NEAR(data[t * 6 + s], -2e10 , 1e-4);
        } else {
          EXPECT_NEAR(data[t * 6 + s],
            -2.0 * (pow(s - t, 2) + pow(6 + s - (3 + t), 2)) , 1e-4);
        }
      }
    }
}

TYPED_TEST(PdistLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  PdistLayer<Dtype> layer(layer_param);

  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(PdistLayerTest, TestGradientScale) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  PdistParameter* pdist_param =
    layer_param.mutable_pdist_param();
  pdist_param->set_scale_value(-2.0);
  PdistLayer<Dtype> layer(layer_param);

  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

// TYPED_TEST(PdistLayerTest, TestGradientIgnore) {
//   typedef typename TypeParam::Dtype Dtype;
//   LayerParameter layer_param;
//   PdistParameter* pdist_param =
//     layer_param.mutable_pdist_param();
//   pdist_param->set_scale_value(-2.0);
//   pdist_param->set_ignore_value(5.0);
//   PdistLayer<Dtype> layer(layer_param);
//
//   GradientChecker<Dtype> checker(1e-2, 1e-3);
//   checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
//       this->blob_top_vec_);
// }

}  // namespace caffe
