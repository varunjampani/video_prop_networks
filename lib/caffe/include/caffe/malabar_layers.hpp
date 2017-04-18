#ifndef MALABAR_LAYERS_HPP_
#define MALABAR_LAYERS_HPP_

/**
 *  @file
 *  @brief This header contains all malabar specific layers which have not
 *         yet merged back to upstream caffe.
 *
 *  Layers which are already available in caffe and have been changed
 *  or improved for the malabar project should be mentioned here as well.
 *
 *  Caffe Layer Improvements:
 *   TODO files and layers which have been improved should be listed here
 *
 */

#include <string>
#include <utility>
#include <vector>

#include <boost/random/mersenne_twister.hpp>

#include "caffe/layer.hpp"
#include "caffe/layers/loss_layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/concat_layer.hpp"

#include "caffe/util/db.hpp"

#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"
#include "caffe/layers/softmax_layer.hpp"


namespace caffe {

  /**
   * Permutohedral Feature layer
   *
   */
  template <typename Dtype>
  class PixelFeatureLayer : public Layer<Dtype> {
  public:
    explicit PixelFeatureLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                         const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const { return "PixelFeature"; }
    virtual inline int ExactNumBottomBlobs() const { return 1; }
    virtual inline int ExactNumTopBlobs() const { return 1; }

  protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

    // Permutohedral Feature parameters
    int count_;
    int num_;
    int channels_;
    int height_, width_;
    bool ran_once;
  };

  /**
   * Matrix multiplication layer.
   *
   */
  template <typename Dtype>
  class MatMulLayer : public Layer<Dtype> {
  public:
    explicit MatMulLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {
      LayerParameter tmp_param;
      tmp_param.mutable_concat_param()->set_concat_dim(3);
      concat_layer_.reset(new ConcatLayer<Dtype>(tmp_param));
    }
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                         const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const { return "MatMul"; }
    virtual inline int MinNumBottomBlobs() const { return 2; }
    virtual inline int ExactNumTopBlobs() const { return 1; }

  protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down,
                              const vector<Blob<Dtype>*>& bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down,
                              const vector<Blob<Dtype>*>& bottom);
    Blob<Dtype> tmp_k_;
    boost::shared_ptr<ConcatLayer<Dtype> > concat_layer_;
  };


  /**
   * Matrix multiplication layer - 2.
   *
   */
  template <typename Dtype>
  class MatMul2Layer : public Layer<Dtype> {
  public:
    explicit MatMul2Layer(const LayerParameter& param)
      : Layer<Dtype>(param) {};
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                         const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const { return "MatMul2"; }
    virtual inline int ExactNumBottomBlobs() const { return 2; }
    virtual inline int ExactNumTopBlobs() const { return 1; }

  protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down,
                              const vector<Blob<Dtype>*>& bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down,
                              const vector<Blob<Dtype>*>& bottom);
    int M_;
    int K_;
    int N_;
    int num_kernels_;
    int channels_;
    bool bias_term_;
    Blob<Dtype> bias_multiplier_;
  };


/**
 * @brief Computes a product of two input Blobs, with the shape of the
 *        latter Blob "broadcast" to match the shape of the former.
 *        Equivalent to tiling the latter Blob, then computing the elementwise
 *        product.
 * Taken from - caffe ://github.com/jeffdonahue/caffe/commit/b4a5b6abb6272207da83bacde448fa3b2d4c7793?diff=split
 */
template <typename Dtype>
class ScalarLayer: public Layer<Dtype> {
 public:
  explicit ScalarLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Scalar"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  /**
   * In the below shape specifications, @f$ i @f$ denotes the value of the
   * `axis` field given by `this->layer_param_.scalar_param().axis()`, after
   * canonicalization (i.e., conversion from negative to positive index,
   * if applicable).
   *
   * @param bottom input Blob vector (length 2)
   *   -# @f$ (d_0 \times ... \times
   *           d_i \times ... \times d_j \times ... \times d_n) @f$
   *      the first factor @f$ x @f$
   *   -# @f$ (d_i \times ... \times d_j) @f$
   *      the second factor @f$ y @f$
   * @param top output Blob vector (length 1)
   *   -# @f$ (d_0 \times ... \times
   *           d_i \times ... \times d_j \times ... \times d_n) @f$
   *      the product @f$ z = x y @f$ computed after "broadcasting" y.
   *      Equivalent to tiling @f$ y @f$ to have the same shape as @f$ x @f$,
   *      then computing the elementwise product.
   */
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Blob<Dtype> sum_multiplier_;
  Blob<Dtype> sum_result_;
  int axis_;
  int outer_dim_, scalar_dim_, inner_dim_;
};

template <typename Dtype>
class Scalar2Layer: public Layer<Dtype> {
 public:
  explicit Scalar2Layer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Scalar2"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Blob<Dtype> tmp_diff_;
  Blob<Dtype> tmp_sum_;
};

template <typename Dtype>
class Scalar3Layer: public Layer<Dtype> {
 public:
  explicit Scalar3Layer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Scalar3"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Blob<Dtype> tmp_sum_;
};

/**
 * @brief Computes a product of input Blob and parameter blob, with the shape of the
 *        parameter Blob "broadcast" to match the shape of the former.
 *        Equivalent to tiling the parameter, then computing the elementwise
 *        product.
 * Adapted from - caffe ://github.com/jeffdonahue/caffe/commit/b4a5b6abb6272207da83bacde448fa3b2d4c7793?diff=split
 */
template <typename Dtype>
class ScalarConvLayer: public Layer<Dtype> {
 public:
  explicit ScalarConvLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
          const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ScalarConv"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Blob<Dtype> sum_multiplier_;
  Blob<Dtype> sum_result_;
  int axis_;
  int outer_dim_, scalar_dim_, inner_dim_;
};


/**
 * @brief Computes @f$ y = exp ^ {\alpha x + \beta} @f$,
 *        as specified by the parameter @f$ \alpha @f$, shift @f$ \beta @f$.
 */
template <typename Dtype>
class ExpScaleLayer : public NeuronLayer<Dtype> {
 public:
  /**
   * @param param provides ExpParameter exp_scale_param,
   *     with ExpScaleLayer options:
   *   - shift (\b optional, default 0) the shift @f$ \beta @f$
   */
  explicit ExpScaleLayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ExpScale"; }

 protected:
  /**
   * @param bottom input Blob vector (length 1)
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the inputs @f$ x @f$
   * @param top output Blob vector (length 1)
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the computed outputs @f$
   *        y = e ^ {\alpha x + \beta}
   *      @f$
   */
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  /**
   * @brief Computes the error gradient w.r.t. the exp inputs and
   * scale parameter.
   */
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Dtype outer_scale_;
};


/**
 * @brief Computes @f$ y = \alpha x  @f$,
 *        as specified by the learnable parameter @f$ \alpha @f$.
 */
template <typename Dtype>
class Scale2Layer : public NeuronLayer<Dtype> {
 public:
  explicit Scale2Layer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Scale2"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  /**
   * @brief Computes the error gradient w.r.t. the inputs and
   * scale parameter.
   */
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
};


template <typename Dtype>
class SmearLayer : public Layer<Dtype> {
 public:
  explicit SmearLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Smear"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int num_;
  int channels_;

  int out_height_;
  int out_width_;

  int outer_num_;
  int inner_num_;

  Dtype ignore_idx_value_;
  Dtype ignore_feature_value_;
};


template <typename Dtype>
class PdistLayer : public Layer<Dtype> {
 public:
  explicit PdistLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Pdist"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int num_;
  int channels_;

  int out_height_;
  int out_width_;

  float ignore_value_;
  float scale_value_;
};


template <typename Dtype>
class SpixelFeatureLayer : public Layer<Dtype> {
 public:
  explicit SpixelFeatureLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SpixelFeature"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int num_;
  int in_channels_;
  int height_;
  int width_;
  int out_channels_;
  int max_spixels_;

  float rgbxy_rgb_scale_;
  float rgbxy_xy_scale_;
  float xy_scale_;
  float rgb_scale_;
  float ignore_idx_value_;
  float ignore_feature_value_;

  Blob<Dtype> spixel_counts_;
};


template <typename Dtype>
class TransposeLayer: public Layer<Dtype> {
 public:
  explicit TransposeLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
          const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Transpose"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int num_, channels_, height_, width_;
};

  /**
   * @brief Implements the Bilateral Convolution as brute-force but exact.
   *        K*x with K=exp(-D)
   *
   * TODO(pgehler): thorough documentation for Forward, Backward, and proto params.
   */
  template <typename Dtype>
  class BilateralBruteForceLayer : public Layer<Dtype> {
   public:
    explicit BilateralBruteForceLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {};
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const { return "BilateralBruteForce"; }
    virtual inline int ExactNumBottomBlobs() const { return 4; }
    virtual inline int ExactNumTopBlobs() const { return 1; }

   protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

    int num_;
    int channels_;
    int scales_;
    int feature_size_;
    int in_height_, in_width_;
    int out_height_, out_width_;

    // we need this since we have to accumulate gradients for several scalse
    Blob<Dtype> bb_bottom_data_; // pixel
    Blob<Dtype> bb_bottom_f1_;   // pixel
    Blob<Dtype> bb_bottom_f2_;   // superpixel
    Blob<Dtype> bb_bottom_s_;    // scales

    Blob<Dtype> bb_top_pdist_;
    Blob<Dtype> bb_top_scalar_;
    Blob<Dtype> bb_top_softmax_;
    Blob<Dtype> bb_top_matmul_;

    // scalar needs only the top for the backward we can exploit that
    // Blob<Dtype> bb_pdist_full_;
    Blob<Dtype> bb_scalar_full_;
    Blob<Dtype> bb_softmax_full_;

    std::vector<Blob<Dtype>*> tmp_bottom_pdist_;
    std::vector<Blob<Dtype>*> tmp_top_pdist_;
    boost::shared_ptr<PdistLayer<Dtype> > pdist_layer_;
    std::vector<Blob<Dtype>*> tmp_bottom_scalar_;
    std::vector<Blob<Dtype>*> tmp_top_scalar_;
    boost::shared_ptr<Scalar3Layer<Dtype> > scalar_layer_;
    std::vector<Blob<Dtype>*> tmp_bottom_softmax_;
    std::vector<Blob<Dtype>*> tmp_top_softmax_;
    boost::shared_ptr<SoftmaxLayer<Dtype> > softmax_layer_;
    std::vector<Blob<Dtype>*> tmp_bottom_matmul_;
    std::vector<Blob<Dtype>*> tmp_top_matmul_;
    boost::shared_ptr<MatMul2Layer<Dtype> > matmul_layer_;
  };

}  // namespace caffe

#endif  // MALABAR_LAYERS_HPP_
