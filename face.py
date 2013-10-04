import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import csv
import math
import itertools

"""
  10.1.2013 9:20am - Decided not to combine x,y coordinate columns into a single column of tuples
                     Added more comments

  10.3.2013 1:00pm - Replaced readData() with readTrain() and readTest() that save each Image as a 96x96 numpy array during file reading
                     However, the training DataFrame takes up a decent chunk of memory so I recommend you don't load the whole thing until you need to
                     Modified (now named) plotImage() to accomodate above changes
"""
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

               
def readTrain(numRows = float('inf')):
  """
  Loads the first numRows rows of training data from the .csv file into a pandas DataFrame, converts Image into 96x96 numpy array
  """
  image_list = []
  first_row = True
  countRows = 0
  
  with open('training.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    
    for row in reader:
      if first_row:
        first_row = False
        continue
      if countRows == numRows:
        break
      column_num = 0
      row_dict = {}
      for item in keypoints:
        if not row[column_num] and not row[column_num+1]:
          row_dict[item + '_x'] = float('nan')
          row_dict[item + '_y'] = float('nan')
        else:
          row_dict[item + '_x'] = float(row[column_num])
          row_dict[item + '_y'] = float(row[column_num+1])
        column_num += 2
      
      row_dict['Image'] = np.array([int(float(x)) for x in row[30].split(' ')]).reshape(96,96)
      
      image_list.append(row_dict)
      countRows += 1
      
  return DataFrame(image_list)
  
def readTest():
  """
  Loads the test data from .csv into a pandas DataFrame, converts Image into 96x96 numpy array 
  """

  image_list = []
  first_row = True
  
  with open('test.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    
    for row in reader:
      if first_row:
        first_row = False
        continue
      row_dict['Image'] = np.array([int(float(x)) for x in row[1].split(' ')]).reshape(96,96)
      image_list.append(row_dict)
      
  return DataFrame(image_list)
  
  
def plotImage(data, num):
  """
  Prints image according to entry num and also facial keypoints if from training data
  """
  # matplotlib gray-scale pyplot 
  plt.imshow(data.ix[num,'Image'], cmap=cm.Greys_r)
  
  # if data contains keypoints, plot keypoints
  for item in keypoints:
    if item + '_x' in data.columns and item + '_y' in data.columns:
      if data[item + '_x'][num] and data[item + '_y'][num]:
        plt.scatter(data[item + '_x'][num], data[item + '_y'][num])
  
  plt.show()
  

  
def avgPatch(data, feature, size):
  """
  Returns square-sized patch that is the average of all patches about facial feature with side length (2*size + 1)
  """
  numImages = 0
  patch = np.zeros((size*2+1,size*2+1))

  for i in data.index:
    # ignore if feature is empty
    if np.isnan(data[feature + '_x'][i]):
      continue
  
    x_left = data[feature + '_x'][i] - size
    x_right = data[feature + '_x'][i] + size
    y_top = data[feature + '_y'][i] - size
    y_bottom = data[feature + '_y'][i] + size
    
    # ignore if patch goes off the edge
    if x_left < 0 or y_top < 0:
      continue
    elif x_right > 95 or y_bottom > 95:
      continue
    else:
      numImages += 1
      patch += data['Image'][i][y_top:y_bottom+1,x_left:x_right+1]
      
  patch = patch.astype('int64') / float(numImages)
  
  # Uncomment following lines if you want to view average patch
  # plt.imshow(patch, cmap=cm.Greys_r, interpolation='nearest')
  # plt.show()
      
  return patch
  


def bestMatch(data, image, patch, feature, searchSize):
  """
  Returns coordinates of best match for given patch in image
  """
  # determine average location for feature
  avg_x = data[feature + '_x'].mean()
  avg_y = data[feature + '_y'].mean()
  
  # calculate patch size from given patch
  patchSize = len(patch)
  patchRad = (patchSize - 1)/2
  
  patchSer = Series(np.squeeze(patch.reshape(1,patchSize*patchSize)))
  
  # calculate search grid around average location of feature of size searchSize
  x_left = int(avg_x - searchSize)
  x_right = int(avg_x + searchSize)
  y_top = int(avg_y - searchSize)
  y_bottom = int(avg_y + searchSize)
  
  # create Series to log correlations of images from search grid to given patch
  corrList = Series(index = itertools.product(range(x_left,x_right),range(y_top,y_bottom)))
  
  for x,y in corrList.index:
    # find patch centered at x,y
    image_xy = image[y-patchRad:y+patchRad+1,x-patchRad:x+patchRad+1]
    imageSer = Series(np.squeeze(image_xy.reshape(1,patchSize*patchSize)))
    # log correlation
    corrList[(x,y)] = imageSer.corr(patchSer)
    
  # select coordinates with best correlation to given patch
  pred = corrList.idxmax()
  
  # plot prediction against test image
  plt.imshow(image, cmap=cm.Greys_r)
  plt.scatter(pred[0],pred[1])
  plt.show()
  
  return pred