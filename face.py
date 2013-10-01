import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math

"""
Original Columns Names
left_eye_center_x, left_eye_center_y
right_eye_center_x, right_eye_center_y
left_eye_inner_corner_x, left_eye_inner_corner_y
left_eye_outer_corner_x, left_eye_outer_corner_y
right_eye_inner_corner_x, right_eye_inner_corner_y
right_eye_outer_corner_x, right_eye_outer_corner_y
left_eyebrow_inner_end_x, left_eyebrow_inner_end_y
left_eyebrow_outer_end_x, left_eyebrow_outer_end_y
right_eyebrow_inner_end_x, right_eyebrow_inner_end_y
right_eyebrow_outer_end_x, right_eyebrow_outer_end_y
nose_tip_x, nose_tip_y
mouth_left_corner_x, mouth_left_corner_y
mouth_right_corner_x, mouth_right_corner_y
mouth_center_top_lip_x, mouth_center_top_lip_y
mouth_center_bottom_lip_x, mouth_center_bottom_lip_y
Image
"""

# keypoints is a list of the 15 facial keypoint features provided in the training data
keypoints = ['left_eye_center', 'right_eye_center', 
              'left_eye_inner_corner', 'left_eye_outer_corner',
              'right_eye_inner_corner', 'right_eye_outer_corner',
              'left_eyebrow_inner_end', 'left_eyebrow_outer_end',
              'right_eyebrow_inner_end', 'right_eyebrow_outer_end',
              'nose_tip',
              'mouth_left_corner', 'mouth_right_corner',
              'mouth_center_top_lip', 'mouth_center_bottom_lip']

              
              
def readData():
  """
  Reads .csv file and outputs DataFrame containing information
  For training.csv, joins x and y coordinates into tuple
  """
  
  train = pd.read_csv('training.csv')
  # converts x and y coordinates corresponding to same feature into tuple
  for item in keypoints:
    train[item] = train[[item + '_x', item + '_y']].apply(tuple, axis=1)
  train = train[keypoints + ['Image']]
  
  test = pd.read_csv('test.csv', index_col = 'ImageId')

  return train, test
<<<<<<< HEAD
=======
 
>>>>>>> d97c6c277ced5d818e3fa2f2337abe8f7810146a
  
  
def plot(data, num):
  """
  Prints image according to entry num and also facial keypoints if from training data
  """
  gray_vals = np.array(data.ix[num, 'Image'].split(' '))
  gray_vals = gray_vals.astype('int64')
  gray_vals = gray_vals.reshape((96, 96))
  plt.imshow(gray_vals, cmap=cm.Greys_r)
  
  for item in keypoints:
    if item in data.columns:
      if data[item][num][0] and data[item][num][1]:
        plt.scatter(data[item][num][0], data[item][num][1])
  
  plt.show()
