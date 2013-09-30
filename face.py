import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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

keypoints = ['left_eye_center', 'right_eye_center', 
              'left_eye_inner_corner', 'left_eye_outer_corner',
              'right_eye_inner_corner', 'right_eye_outer_corner',
              'left_eyebrow_inner_end', 'left_eyebrow_outer_end',
              'right_eyebrow_inner_end', 'right_eyebrow_outer_end',
              'nose_tip',
              'mouth_left_corner', 'mouth_right_corner',
              'mouth_center_top_lip', 'mouth_center_bottom_lip']

              
              
def read(filename):
  data = pd.read_csv(filename)
  
  for item in keypoints:
    data[item] = data[[item + '_x', item + '_y']].apply(tuple, axis=1)
  
  data = data[keypoints + ['Image']]
  
  return data
 
  
  
def plot(data, num):
  gray_vals = np.array(data.ix[num, 'Image'].split(' '))
  gray_vals = gray_vals.astype('int64')
  gray_vals = gray_vals.reshape((96, 96))
  plt.imshow(gray_vals, cmap=cm.Greys_r)
  
  for item in keypoints:
    plt.scatter(data[item][num][0], data[item][num][1])
  
  plt.show()