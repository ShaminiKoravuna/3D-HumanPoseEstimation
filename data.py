"""
Functions for data handling.
"""

from __future__ import division

import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cam as cameras
import skeleton as viz
import h5py
import glob
import copy

def load_data( bpath, subjects, actions, dim=3,verbose=True ):
  """
    bpath: path to load the data from,
    subjects. List of integers. Subjects whose data will be loaded.
    actions. List of strings. The actions to load.
	camera_frame. Boolean. Tells whether to retrieve data in camera coordinate system
  Returns:
    data. Dictionary with keys k=(subject, action, seqname)
          There will be 2 entries per subject/action if loading 3d data.
          There will be 8 entries per subject/action if loading 2d data.
  """

  if not dim in [2,3]:
    raise(ValueError, 'dim must be 2 or 3')

  data = {}

  for subj in subjects:
    for action in actions:
      if verbose:
        print('Reading subject {0}, action {1}'.format(subj, action))

      dpath = os.path.join( bpath, 'S{0}'.format(subj), 'MyPoses/{0}D_positions'.format(dim), '{0}*.h5'.format(action) )
      print( dpath )

      fnames = glob.glob( dpath )

      loaded_seqs = 0
      for fname in fnames:
        seqname = os.path.basename( fname )

        if seqname.startswith( action ):
          if verbose:
            print( fname )
          loaded_seqs = loaded_seqs + 1

          with h5py.File( fname, 'r' ) as h5f:
            poses = h5f['{0}D_positions'.format(dim)][:]

          poses = poses.T
          data[ (subj, action, seqname) ] = poses

      if dim == 2:
        assert loaded_seqs == 8, "Expecting 8 sequences, found {0} instead".format( loaded_seqs )
      else:
        assert loaded_seqs == 2, "Expecting 2 sequences, found {0} instead".format( loaded_seqs )

  return data

def normalization_stats(completeData, dim, predict_14=False ):

  if not dim in [2,3]:
    raise(ValueError, 'dim must be 2 or 3')

  data_mean = np.mean(completeData, axis=0)
  data_std  =  np.std(completeData, axis=0)

  dimensions_to_ignore = []

  if dim == 2:

    dimensions_to_use    = np.array( [0,1,2,3,6,7,8,12,13,15,17,18,19,25,26,27] )
    dimensions_to_use = np.sort( np.hstack( (dimensions_to_use*2, dimensions_to_use*2+1)))
    dimensions_to_ignore = np.delete( np.arange(32*2), dimensions_to_use )
  else: # dim == 3

    dimensions_to_use    = np.array( [0,1,2,3,6,7,8,12,13,14,15,17,18,19,25,26,27] )
    if predict_14:
      dimensions_to_use = np.delete( dimensions_to_use, [0,7,9] )
    else:
      dimensions_to_use = np.delete( dimensions_to_use, 0 )
    dimensions_to_use = np.sort( np.hstack( (dimensions_to_use*3, dimensions_to_use*3+1, dimensions_to_use*3+2)))
    dimensions_to_ignore = np.delete( np.arange(32*3), dimensions_to_use )

  return data_mean, data_std, dimensions_to_ignore, dimensions_to_use

def transform_world_to_camera(poses_set, cams, ncams=4 ):
    """
    Project 3d poses from world coordinate to camera coordinate system
    Args:
      poses_set: dictionary with 3d poses
      cams: dictionary with cameras
      ncams: number of cameras per subject
    Return:
      t3d_camera: dictionary with 3d poses in camera coordinate
    """
    t3d_camera = {}
    for t3dk in sorted( poses_set.keys() ):
      subj, a, seqname = t3dk
      t3d_world = poses_set[ t3dk ]

      for c in range( ncams ):
        R, T, f, c, k, p, name = cams[ (subj, c+1) ]
        camera_coord = cameras.world_to_camera_frame( np.reshape(t3d_world, [-1, 3]), R, T,f, c, k, p)
        camera_coord = np.reshape( camera_coord, [-1, 96] )
        sname = seqname[:-3]+"."+name+".h5" #Waiting 1.58860488.h5
        t3d_camera[ (subj, a, sname) ] = camera_coord

    return t3d_camera


def normalize_data( data, data_mean, data_std, dim_to_use, actions,dim=3):

  data_out = {}
  nactions = len(actions)
  for key in data.keys():
    data[ key ] = data[ key ][ :, dim_to_use ]
    mu = data_mean[dim_to_use]
    stddev = data_std[dim_to_use]
    data_out[ key ] = np.divide( (data[key] - mu), stddev )
  return data_out

def unNormalizeData(normalizedData, data_mean, data_std, dimensions_to_ignore ):

  T = normalizedData.shape[0] # Batch size
  D = data_mean.shape[0] # Dimensionality
  origData = np.zeros((T, D), dtype=np.float32)
  dimensions_to_use = []
  for i in range(D):
    if i in dimensions_to_ignore:
      continue
    dimensions_to_use.append(i)
  #print()
  dimensions_to_use = np.array(dimensions_to_use)

  origData[:, dimensions_to_use] = normalizedData

  stdMat = data_std.reshape((1, D))
  stdMat = np.repeat(stdMat, T, axis=0)
  meanMat = data_mean.reshape((1, D))
  meanMat = np.repeat(meanMat, T, axis=0)
  origData = np.multiply(origData, stdMat) + meanMat
  return origData


def define_actions( action ):

  actions = ["Directions","Discussion","Eating","Greeting",
           "Phoning","Photo","Posing","Purchases",
           "Sitting","SittingDown","Smoking","Waiting",
           "WalkDog","Walking","WalkTogether"]
  return [action]

def project_to_cameras( data_dir, poses_set, cams, ncams=4 ):
  """
  Project 3d poses to obtain 2d ones
  Args:
    poses_set: dictionary with 3d poses
    cams: dictionary with cameras
    ncams: number of cameras per subject
  Return:
    t2d: dictionary with 2d poses
  """
  t2d = {}

  for t3dk in sorted( poses_set.keys() ):
    subj, a, seqname = t3dk
    t3d = poses_set[ t3dk ]
    for cam in range( ncams ):
      R, T, f, c, k, p, name = cams[ (subj, cam+1) ]
      pts2d, _, _, _, _ = cameras.project_point_radial( np.reshape(t3d, [-1, 3]), R, T, f, c, k, p )
      pts2d = np.reshape( pts2d, [-1, 64] )
      sname = seqname[:-3]+"."+name+".h5"
      t2d[ (subj, a, sname) ] = pts2d


  return t2d

def merge_two_dicts(x, y):
  """
  Given two dicts, merge them into a new dict as a shallow copy.
  """
  z = x.copy()
  z.update(y)
  return z

def load_stacked_hourglass(data_dir,subjects,actions,verbose=True):
  """
  Load data from disk, and put it in an easy-to-acess dictionary.

  Args:
    bpath. String. Base path where to load the data from,
    subjects. List of integers. Subjects whose data will be loaded.
    actions. List of strings. The actions to load.
    camera_frame. Boolean. Tells whether to retrieve data in camera coordinate system
  Returns:
    data. Dictionary with keys k=(subject, action, seqname)
          There will be 2 entries per subject/action if loading 3d data.
          There will be 8 entries per subject/action if loading 2d data.
  """
  data = {}
  for subj in subjects:
    for action in actions:
      if verbose:
        print('Reading subject {0}, action {1}'.format(subj, action))
      dpath = os.path.join( data_dir, 'S{0}'.format(subj), 'StackedHourglass/{0}*.h5'.format(action))
      print( dpath )
      fnames = glob.glob( dpath )
      loaded_seqs = 0
      for fname in fnames:
        seqname = os.path.basename( fname )
        seqname = seqname.replace('_',' ')

        if seqname.startswith( action ):
          if verbose:
            print( fname )
          loaded_seqs = loaded_seqs + 1
          with h5py.File( fname, 'r' ) as h5f:
            poses = h5f['poses'][:]
            permutation_idx = np.array([6,2,1,0,3,4,5,7,8,9,13,14,15,12,11,10])
            ### PERMUTE TO MAKE IT COMPATIBLE with h36m
            poses = poses[:,permutation_idx,:]
            poses = np.reshape(poses,[poses.shape[0],-1])
            poses_final = np.zeros([poses.shape[0],32*2])
            dim_to_use_x    = np.array( [0,1,2,3,6,7,8,12,13,15,17,18,19,25,26,27],dtype=np.int32 )*2
            dim_to_use_y    = dim_to_use_x+1
            dim_to_use = np.zeros(16*2,dtype=np.int32)
            dim_to_use[0::2] = dim_to_use_x
            dim_to_use[1::2] = dim_to_use_y
            poses_final[:,dim_to_use] = poses
            seqname = seqname+'-sh'
            data[ (subj, action, seqname) ] = poses_final

      # Make sure we loaded 8 sequences
      if (subj == 11 and action == 'Directions'): # <-- this video is damaged
        assert loaded_seqs == 7, "Expecting 7 sequences, found {0} instead. S:{1} {2}".format(loaded_seqs, subj, action )
      else:
        assert loaded_seqs == 8, "Expecting 8 sequences, found {0} instead. S:{1} {2}".format(loaded_seqs, subj, action )

  return data

def read_2d_predictions(actions, data_dir):

  rcams, vcams = cameras.load_cameras('cameras.h5', [1,5,6,7,8,9,11], n_interpolations=0)
  train_set = load_stacked_hourglass(data_dir, [1, 5, 6, 7, 8], actions)
  test_set  = load_stacked_hourglass( data_dir, [9, 11], actions)

  complete_train = copy.deepcopy( np.vstack( train_set.values() ))
  data_mean, data_std,  dim_to_ignore, dim_to_use = normalization_stats( complete_train, dim=2 )
  train_set = normalize_data( train_set, data_mean, data_std, dim_to_use, actions,2 )
  test_set  = normalize_data( test_set,  data_mean, data_std, dim_to_use, actions,2 )

  return train_set, test_set, data_mean, data_std, dim_to_ignore,dim_to_use


def create_2d_data( actions, data_dir, rcams, vcams, n_interpolations=0 ):
  """
  Creates 2d data from 3d points and real or virtual cameras.
  """

  # Load 3d data
  train_set = load_data( data_dir, [1, 5, 6, 7, 8], actions, dim=3 )
  test_set  = load_data( data_dir, [9, 11], actions, dim=3 )

  train_set_r = project_to_cameras( data_dir, train_set, rcams, ncams=4)
  train_set_v = project_to_cameras( data_dir, train_set, vcams, ncams=4*(n_interpolations))
  train_set = merge_two_dicts( train_set_r, train_set_v )

  test_set  = project_to_cameras( data_dir, test_set, rcams, ncams=4)
  # Apply 2d post-processing  ### FIXME
  # Compute normalization statistics.
  complete_train = copy.deepcopy( np.vstack( train_set.values() ))
  data_mean, data_std, dim_to_ignore, dim_to_use = normalization_stats( complete_train, dim=2 )

  # Divide every dimension independently (good if predicting 3d points directly)
  train_set = normalize_data( train_set, data_mean, data_std, dim_to_use, actions )
  test_set  = normalize_data( test_set,  data_mean, data_std, dim_to_use, actions )

  return train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use

def postprocess_2d( poses_set,bbs ):

  return poses_set

def read_3d_data( actions, data_dir, camera_frame=False,rcams=0,vcams=0,n_interpolations=0,predict_14=False):
  """
  Loads 3d data and normalizes it.
  """

  # Load 3d data
  train_set = load_data( data_dir, [1, 5, 6, 7, 8], actions, dim=3 )
  test_set  = load_data( data_dir, [9, 11], actions, dim=3 )

  if camera_frame:
    train_set_r = transform_world_to_camera(train_set,rcams,ncams=4)
    train_set_v = transform_world_to_camera( train_set, vcams, ncams=4*(n_interpolations) )
    train_set   = merge_two_dicts( train_set_r, train_set_v )
    test_set    = transform_world_to_camera(test_set,rcams,ncams=4)

  # Apply 3d post-processing
  train_set, train_root_positions = postprocess_3d( train_set )
  test_set,  test_root_positions  = postprocess_3d( test_set )
  complete_train = copy.deepcopy( np.vstack( train_set.values() ))

  # Compute normalization statistics
  if predict_14:
    data_mean, data_std, dim_to_ignore, dim_to_use = normalization_stats( complete_train, dim=3, predict_14=True )
  else:
    data_mean, data_std, dim_to_ignore, dim_to_use = normalization_stats( complete_train, dim=3 )

  train_set = normalize_data( train_set, data_mean, data_std, dim_to_use, actions )
  test_set  = normalize_data( test_set,  data_mean, data_std, dim_to_use, actions )

  return train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use, train_root_positions, test_root_positions

def postprocess_3d( poses_set ):
  """
  Center 3d points around root
  """
  root_positions = {}
  for k in poses_set.keys():
    root_positions[k] = copy.deepcopy(poses_set[k][:,:3])

    # Remove the root from the 3d position
    poses = poses_set[k]
    poses = poses - np.tile( poses[:,:3], [1, 32] )
    poses_set[k] = poses

  return poses_set, root_positions
