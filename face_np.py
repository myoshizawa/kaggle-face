import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import csv
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

keypoints = ['left_eye_center', 'right_eye_center', 
              'left_eye_inner_corner', 'left_eye_outer_corner',
              'right_eye_inner_corner', 'right_eye_outer_corner',
              'left_eyebrow_inner_end', 'left_eyebrow_outer_end',
              'right_eyebrow_inner_end', 'right_eyebrow_outer_end',
              'nose_tip',
              'mouth_left_corner', 'mouth_right_corner',
              'mouth_center_top_lip', 'mouth_center_bottom_lip']

def read(filename):
  
  image_list = []
  first_row = True
  
  with open(filename, 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    
    for row in reader:
      if first_row:
        first_row = False
        continue
      column_num = 0
      row_dict = {}
      for item in keypoints:
        if not row[column_num] and not row[column_num+1]:
          row_dict[item] = 'NaN'
        else:
          row_dict[item] = (float(row[column_num]), float(row[column_num+1]))
        column_num += 2
      
      row_dict['Image'] = [int(float(x)) for x in row[30].split(' ')]
      
      image_list.append(row_dict)
  return image_list
  
def plot(image_list, num):
  grid = np.array(image_list[num]['Image'])
  grid = grid.reshape((96,96))
  plt.imshow(grid, cmap=cm.Greys_r)
  
  for item in keypoints:
    plt.scatter(image_list[num][item][0], image_list[num][item][1])
  plt.show()