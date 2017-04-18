#include <algorithm>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/malabar_layers.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class Scalar3LayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  Scalar3LayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 4, 5)),
        blob_bottom_scalar1_(new Blob<Dtype>(1,1,1,1)),
        blob_bottom_scalar2_(new Blob<Dtype>(1,1,1,1)),
        blob_top_(new Blob<Dtype>()) {
    Caffe::set_random_seed(1701);
    FillerParameter filler_param;
    filler_param.set_min(1);
    filler_param.set_max(10);
    for(int i = 0; i < blob_bottom_scalar1_->count();++i){
      blob_bottom_scalar1_->mutable_cpu_data()[i] = i+1;
    }
    for(int i = 0; i < blob_bottom_->count();++i){
      blob_bottom_->mutable_cpu_data()[i] = i-blob_bottom_->count()/2;
    }
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(blob_bottom_scalar2_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
    blob_top_in_place_vec_.push_back(blob_bottom_);
  }
  virtual ~Scalar3LayerTest() {
    delete blob_bottom_;
    delete blob_bottom_scalar1_;
    delete blob_bottom_scalar2_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_bottom_scalar1_;
  Blob<Dtype>* const blob_bottom_scalar2_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  vector<Blob<Dtype>*> blob_top_in_place_vec_;
};

TYPED_TEST_CASE(Scalar3LayerTest, TestDtypesAndDevices);

TYPED_TEST(Scalar3LayerTest, TestForwardScalar1) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_scalar1_);
  LayerParameter layer_param;
  shared_ptr<Scalar3Layer<Dtype> > layer(new Scalar3Layer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  int scales = this->blob_bottom_vec_[1]->count();
  for (int n = 0; n < this->blob_bottom_->num(); ++n) {
    for (int k = 0; k < scales; ++k) {
      for (int c = 0; c < this->blob_bottom_->channels(); ++c) {
        for (int h = 0; h < this->blob_bottom_->height(); ++h) {
          for (int w = 0; w < this->blob_bottom_->width(); ++w) {
            // std::cout << n << " " << c << " " << h << " " << w << " " << k << std::endl;
            // std::cout << "top " << this->blob_top_->data_at(n, k*scales+c, h, w)<< std::endl;
            // std::cout << "ref " << this->blob_bottom_vec_[0]->data_at(n, c, h, w) *
            //           this->blob_bottom_vec_[1]->data_at(k,0,0,0) << std::endl;
            // std::cout << "ref " << this->blob_bottom_vec_[0]->data_at(n, c, h, w) << std::endl;
            // std::cout << "ref " << this->blob_bottom_vec_[1]->data_at(k,0,0,0) << std::endl;
            EXPECT_EQ(this->blob_top_->data_at(n, k*this->blob_bottom_vec_[0]->shape(1)+c, h, w),
                      this->blob_bottom_vec_[0]->data_at(n, c, h, w) *
                      this->blob_bottom_vec_[1]->data_at(k,0,0,0));
          }
        }
      }
    }
  }
}

  TYPED_TEST(Scalar3LayerTest, TestForwardScalar1InPlace) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_scalar1_);
  LayerParameter layer_param;
  shared_ptr<Scalar3Layer<Dtype> > layer(new Scalar3Layer<Dtype>(layer_param));
  shared_ptr<Scalar3Layer<Dtype> > layer2(new Scalar3Layer<Dtype>(layer_param));
  // Blob<Dtype>* bottom_copy;
  // bottom_copy->ReshapeLike(*this->blob_bottom_);
  // caffe_copy(this->blob_bottom_->count(), this->blob_bottom_, bottom_copy);

  layer2->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer2->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_in_place_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_in_place_vec_);
  for (int n = 0; n < this->blob_bottom_->num(); ++n) {
      for (int c = 0; c < this->blob_bottom_->channels(); ++c) {
        for (int h = 0; h < this->blob_bottom_->height(); ++h) {
          for (int w = 0; w < this->blob_bottom_->width(); ++w) {
            EXPECT_EQ(this->blob_bottom_->data_at(n, c, h, w),
                      this->blob_top_in_place_vec_[0]->data_at(n, c, h, w));
            EXPECT_EQ(this->blob_top_vec_[0]->data_at(n, c, h, w),
                      this->blob_top_in_place_vec_[0]->data_at(n, c, h, w));
          }
      }
    }
  }
}

  TYPED_TEST(Scalar3LayerTest, TestGradientScalar1) {
    typedef typename TypeParam::Dtype Dtype;
    this->blob_bottom_vec_.push_back(this->blob_bottom_scalar1_);
    LayerParameter layer_param;
    Scalar3Layer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 1e-3);
    // layer.Forward(this->blob_bottom_vec_,
    //                this->blob_top_vec_);
    // std::vector<bool>prop(2);
    // prop[0]= true;
    // prop[1]= true;
    // layer.Backward(this->blob_top_vec_,prop,
    //                this->blob_bottom_vec_);
    checker.CheckGradientExhaustive(&layer,
                                    this->blob_bottom_vec_,
                                    this->blob_top_vec_,
                                    1);
  }

  TYPED_TEST(Scalar3LayerTest, TestGradientScalar2) {
    typedef typename TypeParam::Dtype Dtype;
    this->blob_bottom_vec_.push_back(this->blob_bottom_scalar2_);
    LayerParameter layer_param;
    Scalar3Layer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 1e-3);
    checker.CheckGradientExhaustive(&layer,
                                    this->blob_bottom_vec_,
                                    this->blob_top_vec_,
                                    1);
  }

  TYPED_TEST(Scalar3LayerTest, TestGradientScalar2InPlace) {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    Scalar3Layer<Dtype> layer(layer_param);
    Scalar3Layer<Dtype> layer2(layer_param);

    // Set up blobs
    vector<Blob<Dtype>*> blob_bottom_vec_2;
    vector<Blob<Dtype>*> blob_top_vec_2;
    shared_ptr<Blob<Dtype> > blob_bottom_2(new Blob<Dtype>());
    shared_ptr<Blob<Dtype> > blob_top_2(new Blob<Dtype>());
    blob_bottom_vec_2.push_back(blob_bottom_2.get());
    blob_top_vec_2.push_back(blob_top_2.get());
    blob_bottom_2->CopyFrom(*this->blob_bottom_, false, true);

    this->blob_bottom_vec_.push_back(this->blob_bottom_scalar2_);
    blob_bottom_vec_2.push_back(this->blob_bottom_scalar2_);

    // SetUp layers
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_in_place_vec_);
    layer2.SetUp(blob_bottom_vec_2, blob_top_vec_2);

    // Check forward
    layer.Forward(this->blob_bottom_vec_, this->blob_top_in_place_vec_);
    layer2.Forward(blob_bottom_vec_2, blob_top_vec_2);
    for (int s = 0; s < blob_top_2->count(); ++s) {
      EXPECT_EQ(this->blob_bottom_->cpu_data()[s], blob_top_2->cpu_data()[s]);
    }

     // Fill top diff with random numbers
    shared_ptr<Blob<Dtype> > tmp_blob(new Blob<Dtype>());
    tmp_blob->ReshapeLike(*blob_top_2.get());
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(tmp_blob.get());
    caffe_copy(blob_top_2->count(), tmp_blob->cpu_data(),
               this->blob_bottom_->mutable_cpu_diff());
    caffe_copy(blob_top_2->count(), tmp_blob->cpu_data(),
               blob_top_2->mutable_cpu_diff());
    // Backward in-place
    vector<bool> propagate_down;
    propagate_down.push_back(true);
    propagate_down.push_back(true);
    layer.Backward(this->blob_top_in_place_vec_, propagate_down, this->blob_bottom_vec_);
    // Backward non-in-place
    layer2.Backward(blob_top_vec_2, propagate_down, blob_bottom_vec_2);
    // Check numbers
    for (int s = 0; s < blob_bottom_2->count(); ++s) {
      EXPECT_EQ(this->blob_bottom_->cpu_diff()[s], blob_bottom_2->cpu_diff()[s]);
    }

  }


}  // namespace caffe
