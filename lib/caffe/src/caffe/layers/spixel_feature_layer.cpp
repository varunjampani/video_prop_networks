#include "caffe/util/math_functions.hpp"
#include "caffe/malabar_layers.hpp"


namespace caffe {

/*
Setup function
*/
template <typename Dtype>
void SpixelFeatureLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  num_ = bottom[0]->num();
  in_channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();

  SpixelFeatureParameter spixel_param = this->layer_param_.spixel_feature_param();

  rgbxy_rgb_scale_ = spixel_param.rgbxy_rgb_scale();
  rgbxy_xy_scale_ = spixel_param.rgbxy_xy_scale();
  xy_scale_ = spixel_param.xy_scale();
  rgb_scale_ = spixel_param.rgb_scale();

  if (spixel_param.has_max_spixels()) {
    max_spixels_ = this->layer_param_.spixel_feature_param().max_spixels();
  } else {
    LOG(FATAL) << "Undefined maximum number of superpixels";
  }

  ignore_idx_value_ = spixel_param.ignore_idx_value();
  ignore_feature_value_ = spixel_param.ignore_feature_value();

  switch (spixel_param.type()) {
    case SpixelFeatureParameter_Feature_AVGXY:
      out_channels_ = 2;
      break;
    case SpixelFeatureParameter_Feature_AVGRGB:
      out_channels_ = bottom[0]->channels();
      break;
    case SpixelFeatureParameter_Feature_AVGRGBXY:
      out_channels_ = bottom[0]->channels() + 2;
      break;
    case SpixelFeatureParameter_Feature_AVGXYRGBXY:
      out_channels_ = bottom[0]->channels() + 4;
      break;
    case SpixelFeatureParameter_Feature_AVGRGBXYRGBXY:
      out_channels_ = 2 * bottom[0]->channels() + 4;
      break;
  }

  CHECK_EQ(bottom[1]->num(), num_)
    << "Blob dim-0 (num) should be same for bottom blobs.";

  CHECK_EQ(bottom[1]->channels(), 1)
    << "Index blob has more than one channel.";

  top[0]->Reshape(num_, out_channels_, 1, max_spixels_);
  if (top.size() > 1) {
    top[1]->Reshape(num_, out_channels_, height_, width_);
  }
  spixel_counts_.Reshape(num_, 1, 1, max_spixels_);
}

template <typename Dtype>
void SpixelFeatureLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
    top[0]->Reshape(num_, out_channels_, 1, max_spixels_);
    if (top.size() > 1) {
      top[1]->Reshape(num_, out_channels_, height_, width_);
    }
}

/*
Forward CPU function
*/
template <typename Dtype>
void SpixelFeatureLayer<Dtype>::Forward_cpu(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  caffe_set(top[0]->count(), (Dtype)0., top[0]->mutable_cpu_data());
  caffe_set(spixel_counts_.count(), (Dtype)0.,
    spixel_counts_.mutable_cpu_data());

  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* count_data = spixel_counts_.mutable_cpu_data();

  switch (this->layer_param_.spixel_feature_param().type()) {
  case SpixelFeatureParameter_Feature_AVGXY: {
      for (unsigned int n = 0; n < num_; ++n) {
        // Compute the unique superpixel features
        for (unsigned int y = 0; y < height_; ++y) {
          for (unsigned int x = 0; x < width_; ++x) {
            const int idx = static_cast<int>(bottom[1]->data_at(n, 0, y, x));
            if (idx != ignore_idx_value_) {
              if (idx > max_spixels_) {
                LOG(FATAL) << "Pixel ID is greater than max. superpixels";
              }
              top_data[top[0]->offset(n, 0, 0, idx)] += xy_scale_ * y;
              top_data[top[0]->offset(n, 1, 0, idx)] += xy_scale_ * x;
              count_data[spixel_counts_.offset(n, 0, 0, idx)] += 1;
            }
          }
        }
        // Average the superpixel features
        for (unsigned int t = 0; t < max_spixels_; ++t) {
          if (count_data[spixel_counts_.offset(n, 0, 0, t)] == 0) {
            for (unsigned int k = 0; k < out_channels_; ++k) {
              top_data[top[0]->offset(n, k, 0, t)] = ignore_feature_value_;
            }
          } else {
            for (unsigned int k = 0; k < out_channels_; ++k) {
              top_data[top[0]->offset(n, k, 0, t)] /=
                count_data[spixel_counts_.offset(n, 0, 0, t)];
            }
          }
        }
      }
    break;
  }
  case SpixelFeatureParameter_Feature_AVGRGBXY: {
    for (unsigned int n = 0; n < num_; ++n) {
      // Compute the unique superpixel features
      for (unsigned int y = 0; y < height_; ++y) {
        for (unsigned int x = 0; x < width_; ++x) {
          const int idx = static_cast<int>(bottom[1]->data_at(n, 0, y, x));
          if (idx != ignore_idx_value_) {
            if (idx > max_spixels_) {
              LOG(FATAL) << "Pixel ID is greater than max. superpixels";
            }
            for (unsigned int c = 0; c < in_channels_; ++c) {
              top_data[top[0]->offset(n, c, 0, idx)] +=
                rgbxy_rgb_scale_ * bottom[0]->data_at(n, c, y, x);
            }
            top_data[top[0]->offset(n, in_channels_, 0, idx)] +=
              rgbxy_xy_scale_ * y;
            top_data[top[0]->offset(n, in_channels_ + 1, 0, idx)] +=
              rgbxy_xy_scale_ * x;
            count_data[spixel_counts_.offset(n, 0, 0, idx)] += 1;
          }
        }
      }
      // Average the superpixel features
      for (unsigned int t = 0; t < max_spixels_; ++t) {
        if (count_data[spixel_counts_.offset(n, 0, 0, t)] == 0) {
          for (unsigned int k = 0; k < out_channels_; ++k) {
            top_data[top[0]->offset(n, k, 0, t)] = ignore_feature_value_;
          }
        } else {
          for (unsigned int k = 0; k < out_channels_; ++k) {
            top_data[top[0]->offset(n, k, 0, t)] /=
              count_data[spixel_counts_.offset(n, 0, 0, t)];
          }
        }
      }
    }
    break;
  }
  case SpixelFeatureParameter_Feature_AVGRGB: {
    for (unsigned int n = 0; n < num_; ++n) {
      // Compute the unique superpixel features
      for (unsigned int y = 0; y < height_; ++y) {
        for (unsigned int x = 0; x < width_; ++x) {
          const int idx = static_cast<int>(bottom[1]->data_at(n, 0, y, x));
          if (idx != ignore_idx_value_) {
            if (idx > max_spixels_) {
              LOG(FATAL) << "Pixel ID is greater than max. superpixels";
            }
            for (unsigned int c = 0; c < in_channels_; ++c) {
              top_data[top[0]->offset(n, c, 0, idx)] +=
                rgb_scale_ * bottom[0]->data_at(n, c, y, x);
            }
            count_data[spixel_counts_.offset(n, 0, 0, idx)] += 1;
          }
        }
      }
      // Average the superpixel features
      for (unsigned int t = 0; t < max_spixels_; ++t) {
        if (count_data[spixel_counts_.offset(n, 0, 0, t)] == 0) {
          for (unsigned int k = 0; k < out_channels_; ++k) {
            top_data[top[0]->offset(n, k, 0, t)] = ignore_feature_value_;
          }
        } else {
          for (unsigned int k = 0; k < out_channels_; ++k) {
            top_data[top[0]->offset(n, k, 0, t)] /=
              count_data[spixel_counts_.offset(n, 0, 0, t)];
          }
        }
      }
    }
    break;
  }
  case SpixelFeatureParameter_Feature_AVGXYRGBXY: {
    for (unsigned int n = 0; n < num_; ++n) {
      // Compute the unique superpixel features
      for (unsigned int y = 0; y < height_; ++y) {
        for (unsigned int x = 0; x < width_; ++x) {
          const int idx = static_cast<int>(bottom[1]->data_at(n, 0, y, x));
          if (idx != ignore_idx_value_) {
            if (idx > max_spixels_) {
              LOG(FATAL) << "Pixel ID is greater than max. superpixels";
            }
            top_data[top[0]->offset(n, 0, 0, idx)] += xy_scale_ * y;
            top_data[top[0]->offset(n, 1, 0, idx)] += xy_scale_ * x;
            for (unsigned int c = 2; c < in_channels_ + 2; ++c) {
              top_data[top[0]->offset(n, c, 0, idx)] +=
                rgbxy_rgb_scale_ * bottom[0]->data_at(n, c - 2, y, x);
            }
            top_data[top[0]->offset(n, in_channels_ + 2, 0, idx)] +=
              rgbxy_xy_scale_ * y;
            top_data[top[0]->offset(n, in_channels_ + 3, 0, idx)] +=
              rgbxy_xy_scale_ * x;
            count_data[spixel_counts_.offset(n, 0, 0, idx)] += 1;
          }
        }
      }
      // Average the superpixel features
      for (unsigned int t = 0; t < max_spixels_; ++t) {
        if (count_data[spixel_counts_.offset(n, 0, 0, t)] == 0) {
          for (unsigned int k = 0; k < out_channels_; ++k) {
            top_data[top[0]->offset(n, k, 0, t)] = ignore_feature_value_;
          }
        } else {
          for (unsigned int k = 0; k < out_channels_; ++k) {
            top_data[top[0]->offset(n, k, 0, t)] /=
              count_data[spixel_counts_.offset(n, 0, 0, t)];
          }
        }
      }
    }
    break;
  }
  case SpixelFeatureParameter_Feature_AVGRGBXYRGBXY: {
    for (unsigned int n = 0; n < num_; ++n) {
      // Compute the unique superpixel features
      for (unsigned int y = 0; y < height_; ++y) {
        for (unsigned int x = 0; x < width_; ++x) {
          const int idx = static_cast<int>(bottom[1]->data_at(n, 0, y, x));
          if (idx != ignore_idx_value_) {
            if (idx > max_spixels_) {
              LOG(FATAL) << "Pixel ID is greater than max. superpixels";
            }
            // RGB features
            for (unsigned int c = 0; c < in_channels_; ++c) {
              top_data[top[0]->offset(n, c, 0, idx)] +=
                rgb_scale_ * bottom[0]->data_at(n, c, y, x);
            }

            // XY features
            top_data[top[0]->offset(n, in_channels_, 0, idx)] +=
              xy_scale_ * y;
            top_data[top[0]->offset(n, in_channels_ + 1, 0, idx)] +=
              xy_scale_ * x;

            // RGBXY features
            for (unsigned int c = 0; c < in_channels_; ++c) {
              top_data[top[0]->offset(n, in_channels_ + 2 + c, 0, idx)] +=
                rgbxy_rgb_scale_ * bottom[0]->data_at(n, c, y, x);
            }
            top_data[top[0]->offset(n, 2 * in_channels_ + 2, 0, idx)] +=
              rgbxy_xy_scale_ * y;
            top_data[top[0]->offset(n, 2 * in_channels_ + 3, 0, idx)] +=
              rgbxy_xy_scale_ * x;
            count_data[spixel_counts_.offset(n, 0, 0, idx)] += 1;
          }
        }
      }
      // Average the superpixel features
      for (unsigned int t = 0; t < max_spixels_; ++t) {
        if (count_data[spixel_counts_.offset(n, 0, 0, t)] == 0) {
          for (unsigned int k = 0; k < out_channels_; ++k) {
            top_data[top[0]->offset(n, k, 0, t)] = ignore_feature_value_;
          }
        } else {
          for (unsigned int k = 0; k < out_channels_; ++k) {
            top_data[top[0]->offset(n, k, 0, t)] /=
              count_data[spixel_counts_.offset(n, 0, 0, t)];
          }
        }
      }
    }
    break;
  }
  default:
    LOG(FATAL) << "Undefined feature type of superpixel feature";
  }

  // Copy the superpixel feature values to pixels in top[1]
  if (top.size() > 1) {
    caffe_set(top[1]->count(), (Dtype)0., top[1]->mutable_cpu_data());
    Dtype* top_data_2 = top[1]->mutable_cpu_data();
    for (unsigned int n = 0; n < num_; ++n) {
      for (unsigned int y = 0; y < height_; ++y) {
        for (unsigned int x = 0; x < width_; ++x) {
          const int idx = static_cast<int>(bottom[1]->data_at(n, 0, y, x));
          if (idx != ignore_idx_value_) {
            for (unsigned int k = 0; k < out_channels_; ++k) {
              top_data_2[top[1]->offset(n, k, y, x)] =
                top_data[top[0]->offset(n, k, 0, idx)];
            }
          }
          else {
            for (unsigned int k = 0; k < out_channels_; ++k) {
              top_data_2[top[1]->offset(n, k, y, x)] =
                ignore_feature_value_;
            }
          }
        }
      }
    }
  }

}

/*
Backward CPU function (NOT_IMPLEMENTED for now)
 */
template <typename Dtype>
void SpixelFeatureLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
}

#ifdef CPU_ONLY
STUB_GPU(SpixelFeatureLayer);
#endif

INSTANTIATE_CLASS(SpixelFeatureLayer);
REGISTER_LAYER_CLASS(SpixelFeature);

}  // namespace caffe
