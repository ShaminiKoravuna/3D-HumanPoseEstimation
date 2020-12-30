from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import sys
import time
import h5py
import copy
import glob

import matplotlib.pyplot as plt
import numpy as np
from six.moves import xrange
import tensorflow as tf
import socket

import skeleton as viz
import cam as cameras
import model as temporal_model
import data as data_utils
import cv2
from PIL import Image


tf.app.flags.DEFINE_string("model_dir","trained_model/32/checkpoint-1798202", "Training directory.")
tf.app.flags.DEFINE_float("learning_rate", 1e-5, "Learning rate")
tf.app.flags.DEFINE_boolean("sgd", False, "Whether to use SGD for optimization. Default is adam")
tf.app.flags.DEFINE_integer("seqlen", 5, "Length of sequence")




tf.app.flags.DEFINE_boolean("use_sh", False, "Use 2d pose predictions from StackedHourglass")
tf.app.flags.DEFINE_integer("linear_size", 1024, "Size of each model layer.")
tf.app.flags.DEFINE_boolean("camera_frame", False, "Convert 3d poses to camera coordinates")
tf.app.flags.DEFINE_string("data_dir", "./data/h36m","Data directory")
tf.app.flags.DEFINE_string("data_2d_path", "fed/check/all/test.h5","Location of the 2D_pose detection")

tf.app.flags.DEFINE_string("img_dir", "fed/check/all/","Image directory")
tf.app.flags.DEFINE_integer("sub_id", 9, "Subject_id for camera for reconstruction")
tf.app.flags.DEFINE_integer("cam_id", 2, "Camera_ID for reconstruction")

tf.app.flags.DEFINE_string("output_dir", "output_results/","Output directory for visualtization")

tf.app.flags.DEFINE_boolean("use_fp16", False, "Train using fp16 instead of fp32.")
tf.app.flags.DEFINE_boolean("use_cpu", False, "Whether to use the CPU")

FLAGS = tf.app.flags.FLAGS




def create_model( session, isTraining, dim_to_use_3d, batch_size, data_mean=0, data_std=0, dim_to_ignore_3d=0):


  model = temporal_model.TemporalModel(
      FLAGS.sgd,
      FLAGS.linear_size,
      batch_size,
      FLAGS.learning_rate,
      '',
      dim_to_use_3d,
      data_mean,
      data_std,
      dim_to_ignore_3d,
	  FLAGS.camera_frame,
      FLAGS.seqlen,
      dtype=tf.float16 if FLAGS.use_fp16 else tf.float32)

  if os.path.isfile(FLAGS.model_dir+'.index'):
    print("Loading model ")
    model.saver.restore( session, FLAGS.model_dir )
  else:
    raise ValueError("Asked to load checkpoint {0}, but it does not seem to exist".format(model_dir))

  return model


  #return model

def define_actions( action ):
  """
  Given an action string, returns a list of corresponding actions.
  Args
    action: String. either "all" or one of the h36m actions
  Returns
    actions: List of strings. Actions to use.
  Raises
    ValueError: if the action is not a valid action in Human 3.6M
  """
  actions = ["Directions","Discussion","Eating","Greeting",
           "Phoning","Photo","Posing","Purchases",
           "Sitting","SittingDown","Smoking","Waiting",
           "WalkDog","Walking","WalkTogether"]

  if action == "All" or action == "all":
    return actions

  if not action in actions:
    raise( ValueError, "Unrecognized action: %s" % action )

  return [action]



def create_3D():
  actions = define_actions("All")
  rcams, vcams = cameras.load_cameras('cameras.h5', [1,5,6,7,8,9,11])
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
  device_count = {"GPU": 0} if FLAGS.use_cpu else {"GPU": 1}
  os.system('mkdir -p ' + FLAGS.output_dir)
  train_set_3d, test_set_3d, data_mean_3d, data_std_3d, dim_to_ignore_3d, dim_to_use_3d, train_root_positions, test_root_positions = data_utils.read_3d_data(
    actions, FLAGS.data_dir, FLAGS.camera_frame, rcams, vcams )
  if(FLAGS.use_sh):
    train_set_2d, test_set_2d, data_mean_2d, data_std_2d, dim_to_ignore_2d, dim_to_use_2d = data_utils.read_2d_predictions(actions, FLAGS.data_dir )
  else:
    train_set_2d, test_set_2d, data_mean_2d, data_std_2d, dim_to_ignore_2d, dim_to_use_2d = data_utils.create_2d_data(actions, FLAGS.data_dir, rcams, vcams)

  with tf.Session(config=tf.ConfigProto( gpu_options=gpu_options, device_count = device_count )) as sess:
    # === Create the model ===
    isTraining = False
    batch_size   = 1
    nsamples     = batch_size
    isTraining = False

    model = create_model( sess, isTraining, dim_to_use_3d, 1, data_mean_3d, data_std_3d, dim_to_ignore_3d)
    print("Model created")

    with h5py.File(FLAGS.data_2d_path, 'r' ) as h5f:
      enc_in = h5f['all'][:]
      enc_in = enc_in[ :, dim_to_use_2d ]

    mu = data_mean_2d[dim_to_use_2d]
    stddev = data_std_2d[dim_to_use_2d]
    enc_in = np.divide( (enc_in - mu), stddev )
    n2d = enc_in.shape[0]
    n_extra = n2d%FLAGS.seqlen
    if n_extra>0:
      enc_in  = enc_in[:-n_extra, :]
    n2d = enc_in.shape[0]

    pose_2d_sliding = []
    encoder_inputs = []
    for i in range(n2d-FLAGS.seqlen+1):
      pose_2d_sliding.append(enc_in[i:i+FLAGS.seqlen,:])
    pose_2d_list = np.stack(pose_2d_sliding)
    encoder_inputs.append(pose_2d_list)
    encoder_inputs = np.vstack(encoder_inputs)
    n_splits = n2d-FLAGS.seqlen+1
    encoder_inputs   = np.array_split( encoder_inputs,  n_splits)
    all_poses_3d = []
    enc_inputs = []
    ### MAKE PREDICTIONS ######

    for bidx in range( len(encoder_inputs) ):
      dp = 1.0
      enc_in = encoder_inputs[bidx]
      dec_out  = np.zeros(shape=(1,FLAGS.seqlen,48))
      enc_gt = 0
      _, _, poses3d = model.step( sess, enc_in, dec_out, dp, isTraining=False )

      enc_in  = np.reshape(enc_in,[-1,16*2])
      poses3d = np.reshape(poses3d,[-1,16*3])
      if not(bidx==0):
        enc_in = np.expand_dims(enc_in[FLAGS.seqlen-1,:],axis=0)
        poses3d = np.expand_dims(poses3d[FLAGS.seqlen-1,:],axis=0)
      inp = data_utils.unNormalizeData(  enc_in, data_mean_2d, data_std_2d, dim_to_ignore_2d )
      poses3d = data_utils.unNormalizeData( poses3d, data_mean_3d, data_std_3d, dim_to_ignore_3d )
      enc_inputs.append(inp)
      all_poses_3d.append( poses3d )

    enc_in  = np.vstack( enc_inputs )
    poses3d = np.vstack( all_poses_3d )

    ## Choose camera_id for reconstruction into world coordinate

    the_cam = rcams[(FLAGS.sub_id,FLAGS.cam_id)]    #54138969# 55011271# 58860488 # 60457274
    R, _, _, _, _, _, name = the_cam
    print(name)
    # # Apply inverse rotation and translation
    poses3d = np.reshape(poses3d,[-1, 3])
    X_cam = R.T.dot( poses3d.T)
    poses3d = np.reshape(X_cam.T, [-1, 32*3])

    poses3d = poses3d - np.tile( poses3d[:,:3], [1,32] )


    ##### GENERATE THE 3D predictions


    fnames = sorted(glob.glob(FLAGS.img_dir+"*.jpg"))
    for i in range(n2d):
      #t0 = time()
      fig = plt.figure( figsize=(12.8, 7.2) )
      ax1 = fig.add_subplot( 1, 2, 1 )
      ax2 = fig.add_subplot( 1, 2, 1 + 1, projection='3d')
      n2d = enc_in.shape[0]
      ob1 = viz.Ax2DPose(ax1)
      ob2 = viz.Ax3DPose(ax2, lcolor="#9b59b6", rcolor="#2ecc71")
      print("Working on figure {0:04d} / {1:05d}... \n".format(i+1, n2d), end='')
      p2d = enc_in[i,:]
      im =  Image.open( fnames[i] )
      ob1.update(im,p2d)
      # Plot 3d gt
      p3d = poses3d[i,:]
      ob2.update(p3d)
      fig.canvas.draw()
      img_str = np.fromstring (fig.canvas.tostring_rgb(), np.uint8)
      ncols, nrows = fig.canvas.get_width_height()
      nparr = np.fromstring(img_str, dtype=np.uint8).reshape(nrows, ncols, 3)
      print(FLAGS.output_dir+'{0:05d}.jpg'.format(i+1))
      cv2.imwrite(FLAGS.output_dir+'{0:05d}.jpg'.format(i+1), nparr[:,:,::-1])
      print(p2d)
      print(p3d)
      plt.show()



def main(_):
  create_3D()

if __name__ == "__main__":
  tf.app.run()
