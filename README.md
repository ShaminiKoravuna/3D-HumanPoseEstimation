# 3D-Human Pose Estimation Using Different Sensor Modalities

3D human pose estimation from images or videos has several high-end applications in the field of robotics, computer vision and graphics. In this thesis, the task of 3D human pose estimation is divided into two segments:
1)	Estimating 2D poses from the top-performing 2D pose detectors
2)	Mapping the 2D predictions into 3D space

1.	Install cuda and cudnn
* Cuda 9.0 : https://developer.nvidia.com/cuda-90-download-archive
* Cudnn 7.0.5: https://developer.nvidia.com/rdp/cudnn-archive
* Cuda 9.0 is compatible with cudnn 7.0.5
2.	Creating conda envionment: conda create -n env_name python=version
* Conda create -n env_name python=3.5
3.	Activate env_name
* activate env_name
4.	Install tensorflow: pip install tensorflow-gpu (1.0 or above)
5.	Pip install h5py
6.	Update the required parameters in train.py file
7.	Training the model from scratch
* For training the model from scratch use the command:
* Python train.py --use_sh –camera_frame –dropout 0.5
8.	Evaluating the trained model
* To evaluate the trained model, use the command:
* Python train.py –use_sh –camera_frame –dropout 0.5 –load 2345678 –evaluate 
--Here, 2345678 is passed to load which is the checkpoint point for the global iteration number.
9.	Fine-tuning an existing model
* Command:
Python train.py –use_sh –camera_frame –dropout 0.5 –load 1798200
10.	Create 3D poses from a sequence of 2D poses
* Command:
Python output.py –use_sh –camera_frame

## Description of all the files
### The repository contains 7 python files
* cam.py: It loads the information of the cameras of Human3.6M dataset
* data.py: It contains function for dealing with human3.6M dataset
* model.py: It contains the model designed using RNNs with seq-to-seq network
* procustes.py: It computes similarity transformation
* skeleton.py: It contains functions to visualize human poses
* train.py: It trains the model.
* output.py: It generates a sequence of 3D poses from a sequence of 2D poses.

### The data is segregated in order to have an easy access for retrieving:

The 2D predictions are given in 8 sets for each subject

![Image_traffic](https://github.com/ShaminiKoravuna/3D-HumanPoseEstimation/blob/main/imgs/1.png)
 
The 3D predictions are given in 2 sets for each subject

![Image_traffic](https://github.com/ShaminiKoravuna/3D-HumanPoseEstimation/blob/main/imgs/2.png)
 
We need to make sure that the data is loaded properly Log will be generated to log for every training. The model will be stored in trained_model folder. For visualizing it in tensorboard:
* Tensorboard --logdir  ./log/train_log/


## Stacked hourglass model and dependencies
### Requirements:
* Torch7
* Hdf5
* cudnn
* programming language: lua

### Torch 7 docker image for stacked hourglass model:

* sudo nvidia-docker pull registry.cn-hangzhou.aliyuncs.com/docker_learning_aliyun/torch: v1
* Running the model for 2D predictions:
* sudo nvidia-docker run -it --rm -v /path/to/pose-hg-demo-master:/media registry.cn-hangzhou.aliyuncs.com/docker_learning_aliyun/torch: v1
* root@8f1548fc3b34: ~/torch 
* cd /media 
* th main.lua predict-test 

