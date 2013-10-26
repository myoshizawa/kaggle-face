import random
import pandas as pd
from pandas import DataFrame, Series
import numpy as np
import face
from time import time

def demo(type, x1, y1, x2, y2, keypoint = 'left_eye_center'):
  """
  Input: type = 12,21,13,31, or 22
         x1,y1,x2,y2 = upper left and lower right corners of feature
         keypoint = facial keypoint
  Reads training data from .csv file, creates patchSet of training patches for desired keypoint, 
  finds integral images of each patch, and trains weak classifier for entered feature
  """
  train = face.readTrain()
  patchSet = trainingPatches(train, keypoint)
  calcIntImage(patchSet)
  calcFeatures(patchSet, type, x1, y1, x2, y2)
  patchSet['Weights'] = np.ones(len(patchSet))
  weakClassifier(patchSet, type, x1, y1, x2, y2)
  # writeH5(patchSet, 'store', 'patchSet')
  

def trainingPatches(data, keypoint, patchRadius = 12, numWith = 4, numWithout = 4):
  """
  Returns a 3-column DataFrame
    Patch: contains square patches (side length = 2 * patchRadius) as a numpy array
    1: a 1 in this column means facial keypoint present in patch (0 otherwise)
    0: a 1 in this column means facial keypoint not present (0 otherwise)
  Records numWith patches containing keypoint and numWithout patches that do not contain keypoint
  Does not record patches if keypoint is too close (within patchRadius/2) to the boundary of image or sample has no keypoint information
  Containing the keypoint means the patch center is within patchRadius/2 of the keypoint location (taxi cab metric)
  Not containing the keypoint means the patch center is more than patchRadius from the keypoint location (taxi cab metric)
  """

  # will become the columns of the returned DataFrame
  patchList = []
  keypointYes = []
  keypointNo = []
  
  for sample in data.index:
  
    # extract keypoint location
    keypointLocX = data[keypoint + '_x'][sample]
    keypointLocY = data[keypoint + '_y'][sample]
    
    # if keypoint location is not present, report
    if np.isnan(keypointLocX) or np.isnan(keypointLocY):
      print 'Sample %s does not have %s data' % (sample, keypoint)
      
      """
      for i in range(numWithout):
      
        imageDict = {}
      
        contained = True
      
        patchCenterX = random.randint(patchRadius,95-patchRadius)
        patchCenterY = random.randint(patchRadius,95-patchRadius)
        
        imageDict['Patch'] = data['Image'][sample][patchCenterY - patchRadius:patchCenterY + patchRadius, patchCenterX - patchRadius:patchCenterX + patchRadius]
      
        patchList.append(imageDict)
        containsFeature.append(0) 
      """
      
      continue
    
    # report if feature location is too close to boundary
    if keypointLocX < patchRadius/2:
      print 'Sample %s is too close to left boundary at (%d,%d)' % (sample, keypointLocX, keypointLocY)
      continue
    elif keypointLocX > 95-patchRadius/2:
      print 'Sample %s is too close to right boundary at (%d,%d)' % (sample, keypointLocX, keypointLocY)
      continue
    elif keypointLocY < patchRadius/2:
      print 'Sample %s is too close to top boundary at (%d,%d)' % (sample, keypointLocX, keypointLocY)
      continue
    elif keypointLocY > 95-patchRadius/2:
      print 'Sample %s is too close to bottom boundary at (%d,%d)' % (sample, keypointLocX, keypointLocY)
      continue
    
    # if feature location is present, record patches that do not contain feature
    for i in range(numWithout):
    
      imageDict = {}
      
      contained = True
      
      # ensure that patch does not contain feature
      while contained:
        patchCenterX = random.randint(patchRadius,95-patchRadius)
        patchCenterY = random.randint(patchRadius,95-patchRadius)
        
        if (patchCenterX < int(keypointLocX - patchRadius) or patchCenterX > int(keypointLocX + patchRadius)) and (patchCenterY < int(keypointLocY - patchRadius) or patchCenterY > int(keypointLocY + patchRadius)):
          contained = False
          
      imageDict['Patch'] = data['Image'][sample][patchCenterY - patchRadius:patchCenterY + patchRadius, patchCenterX - patchRadius:patchCenterX + patchRadius]
      
      patchList.append(imageDict)
      keypointNo.append(1)
      keypointYes.append(0)
  
    # if feature is not too close to boundary, record patches containing feature
    for i in range(numWith):
    
      imageDict = {}

      closeToBoundary = True
    
      # ensure patch is not too close to boundary
      while closeToBoundary:
        patchCenterX = random.randint(int(keypointLocX - patchRadius/2), int(keypointLocX + patchRadius/2)+1)
        patchCenterY = random.randint(int(keypointLocY - patchRadius/2), int(keypointLocY + patchRadius/2)+1)
      
        if patchCenterX >= patchRadius and patchCenterX <= 95-patchRadius and patchCenterY >= patchRadius and patchCenterY <= 95-patchRadius:
          closeToBoundary = False
      
      imageDict['Patch'] = data['Image'][sample][patchCenterY - patchRadius:patchCenterY + patchRadius, patchCenterX - patchRadius:patchCenterX + patchRadius]
      
      patchList.append(imageDict)
      keypointYes.append(1)
      keypointNo.append(0)

  # combine lists into a single DataFrame
  trainingSet = DataFrame(patchList)
  trainingSet[1] = keypointYes
  trainingSet[0] = keypointNo
  
  return trainingSet
  
  
def calcIntImage(trainingSet):
  """
  Reads in a DataFrame with a column 'Patch' and returns DataFrame with a new column containing integral images
  """
  ii = []
  
  for sample in trainingSet.index:
    ii.append(integralImage(trainingSet['Patch'][sample]))
    
  trainingSet['IntImage'] = ii

  
def integralImage(patch):
  """
  This function computes the integral image. 
  INPUT: An NxN numpy array representing an image
  OUTPUT: The corresponding integral image 
  where the (x,y) coordinate gives you the sum of the pixels in the rectangle between (0,0) and (x,y)
  note that if x or y is 0, the integral image is 0
  """
 
  N=len(patch)
  s = np.zeros((N+1,N+1), dtype=np.int32)
  int_im = np.zeros((N+1,N+1), dtype=np.int32)
    
  for x in xrange(1,N+1):
    for y in xrange(1,N+1):
      s[x][y] = s[x][y-1] + patch[x-1][y-1]
      int_im[x][y] = int_im[x-1][y] + s[x][y]
 	 
  return int_im


def feature21(x1,y1,x2,y2,ii):
  """
  2-features parted vertically
  """
  y12 = (y1 + y2) / 2
  
  return ii[y2,x2] - 2*ii[y12,x2] + ii[y1,x2] - ii[y2,x1] + 2*ii[y12,x1] - ii[y1,x1]
  

def feature12(x1,y1,x2,y2,ii):
  """
  2-features parted horizontally
  """
  
  x12 = (x1 + x2) / 2
  
  return ii[y2,x2] - 2*ii[y2,x12] + ii[y2,x1] - ii[y1,x2] + 2*ii[y1,x12] - ii[y1,x1]
  
  
def feature31(x1,y1,x2,y2,ii):
  """
  3-features sliced vertically
  """
  
  third = (y2 - y1)/3
  y13 = y1 + third
  y23 = y13 + third
  
  return ii[y2,x2] - 2*ii[y23,x2] + 2*ii[y13,x2] - ii[y1,x2] - ii[y2,x1] + 2*ii[y23,x1] - 2*ii[y13,x1] + ii[y1,x1]

def feature13(x1,y1,x2,y2,ii):
  """
  3-features sliced horizontally
  """
  
  third = (x2 - x1)/3
  x13 = x1 + third
  x23 = x13 + third
  
  return ii[y2,x2] - 2*ii[y2,x23] + 2*ii[y2,x13] - ii[y2,x1] - ii[y1,x2] + 2*ii[y1,x23] - 2*ii[y1,x13] + ii[y1,x1]
  
  
def feature22(x1,y1,x2,y2,ii):
  """
  4-features
  """
  
  x12 = (x1 + x2)/2
  y12 = (y1 + y2)/2
  
  return ii[y2,x2] - 2*ii[y2,x12] + ii[y2,x1] - 2*ii[y12,x2] + 4*ii[y12,x12] - 2*ii[y12,x1] + ii[y1,x2] - 2*ii[y1,x12] + ii[y1,x1]
  
  
def rectFeature(type,x1,y1,x2,y2,ii):
  """
  DESCRIPTION: Given an integral image this function is capable of computing any kind of feature.
  INPUT: The type of feature (one of 12,21,13,31,22)
  (x1,y1) is the coordinates of the top left corner of the feature, (x2,y2) the coordinates of the bottom right corner of the feature
    Note that we use the coordinate system where (0,0) indicates the top left corner of the patch
  ii is the integral image
  OUTPUT: The desired feature
  """
  featureTypes = {12: feature12, 21: feature21, 13: feature13, 31: feature31, 22: feature22}
  
  return featureTypes[type](x1,y1,x2,y2,ii)
  
  
def calcFeature(patchSet, type, x1, y1, x2, y2):
  """
  Takes a dataFrame with integral images (under column 'IntImage') and adds a column '(type,x1,y1,x2,y2)' with the feature values
  """
  columnName = '(%d,%d,%d,%d,%d)' % (type, x1, y1, x2, y2)
  patchSet[columnName] = patchSet['IntImage'].apply(lambda x: rectFeature(type, x1, y1, x2, y2, x))
  
  
def weakClassifier(patchSet, type, x1, y1, x2, y2):
  """
  Input: patchSet is a DataFrame with columns 1, 0, and 'Weights', and '(type,x1,y1,x2,y2)'
    type, x1, y1, x2, y2 uniquely determines a feature
  Outputs the minimum error, threshold, and parity
    parity = 1 => feature value > threshold are classified as containing the facial keypoint
    parity = -1 => feature value =< threshold are classified as containing the facial keypoint
  """
  featureName = '(%d,%d,%d,%d,%d)' % (type, x1, y1, x2, y2)
  
  # create DataFrame that has 0 and 1 as column headings, unique feature value as rows, and number of occurrences weighted by 'Weights' as the table entries
  featureVals = DataFrame(index = patchSet.index)
  featureVals[1] = patchSet[1] * patchSet['Weights']
  featureVals[0] = patchSet[0] * patchSet['Weights']
  featureVals = featureVals.groupby(patchSet[featureName]).sum()
  
  # diff calculates how much the error changes if we moved the threshold from the minimum feature value to the current feature value
  diff = (featureVals[1] - featureVals[0]).cumsum()
  
  # log the minimum error for both possible parities
  posError = featureVals[0].sum() + diff.min()
  negError = featureVals[1].sum() - diff.max()
  
  # return the least error
  if posError <= negError:
    return posError/len(patchSet), diff.idxmin(), 1
  else:
    return negError/len(patchSet), diff.idxmax(), -1