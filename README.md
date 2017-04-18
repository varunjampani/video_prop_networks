# Video Propagation Networks

This is the code accompanying the **CVPR 2017** publication on **Video Propagation Networks**.

This is developed and maintained by
[Varun Jampani](https://varunjampani.github.io),
[Raghudeep Gadde](https://ps.is.tuebingen.mpg.de/person/rgadde),
and
[Peter V. Gehler](https://ps.is.tuebingen.mpg.de/person/pgehler).

Please visit the project [website](http://varunjampani.github.io/vpn) for more details about the paper and overall methodology.

### Download and Installation

Use the following command to fetch this code repository. Use `--recursive` option to fetch the git submodules that this code uses.

```
git clone --recursive https://github.com/varunjampani/video_prop_networks.git
```

The code provided in this repository uses a modified [Caffe](http://caffe.berkeleyvision.org/) neural network library and also uses [gSLICr](https://github.com/carlren/gSLICr) to compute SLIC superpixels.

##### Caffe Installation
For convenience, we already included the modified Caffe in the `lib` folder. Go to `lib/caffe` folder and follow the standard Caffe installation [instructions](http://caffe.berkeleyvision.org/installation.html) to build Caffe in `lib/caffe/build` folder:

```
cd $video_prop_networks/lib/caffe
mkdir build
cd build
cmake ..
make
```

You may need to install some pre-requisites for Caffe. Follow the instructions [here](http://caffe.berkeleyvision.org/installation.html).

##### gSLICr installation

If you clone this repository with `--recursive` option, gSLICr library will be cloned in `lib` folder. Additionally, we provide superpixel computation script in `lib` folder. For installation, check the pre-requisites from gSLICr [page](https://github.com/carlren/gSLICr) and do the following:

```
cd $video_prop_networks/lib/gSLICr
cp ../CMakeLists.txt ../compute_superpixels.cpp .
mkdir build
cd build
cmake ..
make
```

This will create a `compute_superpixels` binary in 'lib/gSLICr/build/` folder which we use for computing superpixels.


### Data Preparation

For the experiments on both object segmentation propagation and color propagation, we use the [DAVIS](http://davischallenge.org) dataset. To download DAVIS dataset in the `data` folder:

```
cd $video_prop_networks/data
sh get_davis.sh
```

## Example 1: Segmentation Propagation

Here, the task is to propagate a given object segmentation from the first frame to the remaining video frames. For speed reasons, we use _superpixel_ sampling for segmentation propagation. Before diving into training or testing, we need to compute superpixels and prepare some training data files.

##### Superpixel Computation

Compute the SLIC superpixels using the following command
```
cd $video_prop_networks
./lib/gSLICr/build/compute_superpixels $IMAGE_DIR $IMAGE_LIST $SUPERPIXEL_DIR $NUM_SPIXELS
```

To extract superpixels on DAVIS data set images:
```
cd $video_prop_networks
./lib/gSLICr/build/compute_superpixels ./data/DAVIS/JPEGImages/480p/ ./data/fold_list/img_list.txt ./data/gslic_spixels/ 12000
```

This will create superpixel indices in `data/gslic_spixels` folder.

##### Training/Testing Data Preparation

For the ease of training, we bundle superpixels, their features and also GT for all the dataset files into single files. Use `prepare_train_data.py` for creating these data files:

```
cd $video_prop_networks/seg_propagation/
python prepare_train_data.py
```

#### BNN-Identity

In our experiments, we first obtain initial propagated masks using BNN-Identity. Use the `run_segmentation.py` script to extract initial BNN-Identity masks for all the videos in the DAVIS dataset:

```
cd $video_prop_networks/seg_propagation/
python run_segmentation.py $STAGE_ID $FOLD_ID (optional)
```

We refer to BNN-Identity as stage-0 (`$STAGE_ID=0`) and since we want to run BNN-Identity on all the dataset images, we will omit the $FOLD_ID:

```
python run_segmentation.py 0
```

This will run BNN-Identity model and save the segementation results in `data/seg_results/STAGE0_RESULT/` folder. This will also save the corresponding superpixel segmentation probabilities in `data/training_data/stage_unaries/STAGE0_UNARY/` folder. These superpixel probabilities are used for training the VPN models.

#### Testing (VPN-DeepLab)

For directly testing the VPN-DeepLab model, we first download the trained segmentation models using the `get_seg_models.sh` script in the `data` folder:

```
cd $video_prop_networks/data/
sh get_seg_models.sh
```

This will download the trained segmentation models corresponding to all 5 folds in to `data/seg_models/` directory.

Then, to get segmentation results on different folds. We can use the same `run_segmentation.py` (used for BNN-Identity) script to run VPN-DeepLab on different dataset folds:

```
cd $video_prop_networks/seg_propagation/
python run_segmentation.py $STAGE_ID $FOLD_ID (optional)
```

where `$STAGE_ID=1` and `$FOLD_ID={0,1,2,3,4}`. Thus, to obtain VPN-DeepLab results on fold-4:

```
cd $video_prop_networks/seg_propagation/
python run_segmentation.py 1 4
```

This will save the segmentation results in `data/seg_results/STAGE1_RESULT/` directory.

#### Evaluation

For segmentation evaluation, we use standard evaluation scripts provided with DAVIS dataset. We slightly modified them to point to right data paths and kept them in the `seg_propagation` folder.

First, install the `davis` evaluation scripts with the following:

```
cd $video_prop_networks/lib/davis/
./configure.sh && make -C build/release
```

To evaluate the VPN-DeepLab stage-1 results:

```
cd $video_prop_networks/seg_propagation/
python eval.py ../data/seg_results/STAGE1_RESULT/ ../data/seg_results/
```

This will save result file in `data/seg_results/STAGE1_RESULT.h5`.

To print different performance metrics:

```
cd $video_prop_networks/seg_propagation/
python eval_view.py ../data/seg_results/STAGE1_RESULT.h5
```

This will print an average IoU score of 75.0 and other metrics mentioned in the paper. Numbers may slightly differ due to the randomization, if you re-trained (below) the models. Replace `STAGE1` with `STAGE0` in the above commands to evaluate BNN-Identity results.

#### Training (VPN-DeepLab)

If we want to re-train the VPN-DeepLab model, we can use `train_fold.sh` script in the `seg_propagation` folder. To run, training on fold-3 data:

```
cd $video_prop_networks/seg_propagation/
sh train_fold.sh 3
```

Similarly, one could do the training for all the 5 folds. Once the training is done (which takes around 1 day for 15000 iterations), the final models are saved in `data/seg_models/` directory.

## Example 2: Color Propagation

To demonstrate the use of VPN for regression, we experimented with color propagation, where the task is to propagate color from the first frame to the remaining video frames in a given grayscale video. We again use DAVIS dataset videos for this example.

In contrast to the above experiments on segmentation propagation, for color propagation, we used random sampling of 300K points from the input video frames instead of superpixels. Also, we use single train and validation split of 35 and 15 videos respectively.

##### Training/Testing Data Preparation

For the ease of training/testing, we pre-computed YCbCrXYT features for each video frame. To compute and save features:

```
cd $video_prop_networks/color_propagation/
python prepare_feature_data.py
```

This will save the features in `data/color_feature_folder/`.

#### BNN-Identity

Like in segmentation propagation, we first obtain initial color propagated videos using BNN-Identity. Use the `do_color_propagation.py` script to get initial BNN-Identity color videos for the DAVIS validation set:

```
cd $video_prop_networks/color_propagation/
python do_color_propagation.py $STAGE_ID
```

We refer to BNN-Identity as stage-0 (`$STAGE_ID=0`):

```
python do_color_propagation.py  0
```

This will run BNN-Identity model and save the color results in `data/color_results/STAGE0_RESULT/` folder.

#### Testing (VPN)

We first download the trained color propagation model using the `get_color_models.sh` script in the `data` folder:

```
cd $video_prop_networks/data/
sh get_color_models.sh
```
This will download the trained VPN model  to `data/color_models/` directory.

Then, to do VPN color propagation, we use the same `do_color_propagation.py ` (used for BNN-Identity) script:

```
cd $video_prop_networks/color_propagation/
python do_color_propagation.py 1
```

where 1 refers to `$STAGE_ID`.

This will save the color propagation results in `data/color_results/STAGE1_RESULT/` directory.

#### Evaluation

To evaluate the color propagation results, we use `compute_rmse.py` script that computes RMSE.

To evaluate BNN-Identity results:

```
cd $video_prop_networks/color_propagation/
python compute_rmse.py --datatype VAL ../data/color_results/STAGE0_RESULT/
```

To evaluate VPN results:

```
cd $video_prop_networks/color_propagation/
python compute_rmse.py --datatype VAL ../data/color_results/STAGE1_RESULT/
```

This will give RMSE values of around 27.89 for BNN-Identity and 28.15 for VPN. The numbers may slightly vary due to the random sampling (of input pixels).

## Citations

Please consider citing the below paper if you make use of this work and/or the corresponding code:

```
@inproceedings{jampani:cvpr:2017,
	title = {Video Propagation Networks},
	author = {Jampani, Varun and Gadde, Raghudeep and Gehler, Peter V.},
	booktitle = { IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
	month = july,
	year = {2017}
}
```
