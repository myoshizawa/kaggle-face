import random
import pandas as pd
from pandas import DataFrame, Series
import numpy as np
import csv
import math
from time import time

def trainingPatches(data, feature, patchRadius = 12, numWith = 4, numWithout = 4):
  """
  Returns a two column DataFrame, one containis square patches (side length = 2 * patchRadius) as a numpy array, the other a 1 or 0 depending on whether it contains the feature
  Records numWith patches containing feature and numWithout patches that do not contain feature
  Does not record patches containing feature if feature is too close (within patchRadius/2) to the boundary of image or sample has no feature information
  Containing the feature means the patch center is within patchRadius/2 of the feature location (taxi cab metric)
  Not containing the feature means the patch center is more than patchRadius from the feature location (taxi cab metric)
  """

  # will become the columns of the returned DataFrame
  patchList = []
  containsFeature = []
  
  for sample in data.index:
  
    # extract feature location
    featureLocX = data[feature + '_x'][sample]
    featureLocY = data[feature + '_y'][sample]
    
    # if feature location is not present, report and just record patches that do not contain feature
    if np.isnan(featureLocX) or np.isnan(featureLocY):
      print 'Sample %s does not have %s data' % (sample, feature)
      
      for i in range(numWithout):
      
        imageDict = {}
      
        contained = True
      
        patchCenterX = random.randint(patchRadius,95-patchRadius)
        patchCenterY = random.randint(patchRadius,95-patchRadius)
        
        imageDict['Patch'] = data['Image'][sample][patchCenterY - patchRadius:patchCenterY + patchRadius, patchCenterX - patchRadius:patchCenterX + patchRadius]
      
        patchList.append(imageDict)
        containsFeature.append(0) 
      
      continue
    
    # if feature location is present, record patches that do not contain feature
    for i in range(numWithout):
    
      imageDict = {}
      
      contained = True
      
      # ensure that patch does not contain feature
      while contained:
        patchCenterX = random.randint(patchRadius,95-patchRadius)
        patchCenterY = random.randint(patchRadius,95-patchRadius)
        
        if (patchCenterX < int(featureLocX - patchRadius) or patchCenterX > int(featureLocX + patchRadius)) and (patchCenterY < int(featureLocY - patchRadius) or patchCenterY > int(featureLocY + patchRadius)):
          contained = False
          
      imageDict['Patch'] = data['Image'][sample][patchCenterY - patchRadius:patchCenterY + patchRadius, patchCenterX - patchRadius:patchCenterX + patchRadius]
      
      patchList.append(imageDict)
      containsFeature.append(0)   
    
    # report if feature location is too close to boundary
    if featureLocX < patchRadius/2:
      print 'Sample %s is too close to left boundary at (%d,%d)' % (sample, featureLocX, featureLocY)
      continue
    elif featureLocX > 95-patchRadius/2:
      print 'Sample %s is too close to right boundary at (%d,%d)' % (sample, featureLocX, featureLocY)
      continue
    elif featureLocY < patchRadius/2:
      print 'Sample %s is too close to top boundary at (%d,%d)' % (sample, featureLocX, featureLocY)
      continue
    elif featureLocY > 95-patchRadius/2:
      print 'Sample %s is too close to bottom boundary at (%d,%d)' % (sample, featureLocX, featureLocY)
      continue
  
    # if feature is not too close to boundary, record patches containing feature
    for i in range(numWith):
    
      imageDict = {}

      closeToBoundary = True
    
      # ensure patch is not too close to boundary
      while closeToBoundary:
        patchCenterX = random.randint(int(featureLocX - patchRadius/2), int(featureLocX + patchRadius/2)+1)
        patchCenterY = random.randint(int(featureLocY - patchRadius/2), int(featureLocY + patchRadius/2)+1)
      
        if patchCenterX >= patchRadius and patchCenterX <= 95-patchRadius and patchCenterY >= patchRadius and patchCenterY <= 95-patchRadius:
          closeToBoundary = False
      
      imageDict['Patch'] = data['Image'][sample][patchCenterY - patchRadius:patchCenterY + patchRadius, patchCenterX - patchRadius:patchCenterX + patchRadius]
      
      patchList.append(imageDict)
      containsFeature.append(1)     

  # combine lists into a single DataFrame
  trainingSet = DataFrame(patchList)
  trainingSet['Feature?'] = containsFeature
  
  return trainingSet
  
  
def writeH5(data, filename, title):
  """
  Writes data to filename.h5 with key title
  """
  store = pd.HDFStore(filename + '.h5')
  
  store[title] = data
  
  store.close()

  
def readH5(filename, title):
  """
  Reads data from filename.h5 with key title
  """
  store = pd.HDFStore(filename + '.h5')
  
  return store[title]
  
  
def calcIntImage(trainingSet):
  """
  Reads in a DataFrame with a column 'Patch' and returns DataFrame with a new column containing integral images
  """
  ii = []
  
  for sample in trainingSet.index:
    ii.append(integralImage(trainingSet['Patch'][sample]))
    
  trainingSet['IntImage'] = ii

  
def featureCounter():
  """
  Function to calculate the number of features in a 24 x 24 patch
  """

  numFeatures = 0
  
  for a in range(24):
    for b in range(24):
      for x in range(a+1,25):
        for y in range(b+1,25):
          # horizontal 2-rectangle features
          if (x - a) % 2 == 0:
            numFeatures += 1
          # vertical 2-rectangle features
          if (y - b) % 2 == 0:
            numFeatures += 1
          # horizontal 3-rectangle features
          if (x - a) % 3 == 0:
            numFeatures += 1
          # vertical 3-rectangle features
          if (y - b) % 3 == 0:
            numFeatures += 1
          # 4-rectangle features
          if (x - a) % 2 == 0 and (y - b) % 2 == 0:
            numFeatures += 1

  return numFeatures
  

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
  Takes a dataFrame with integral images (under column 'IntImage') and adds a column 'Value' with the feature values
  """
  patchSet['Value'] = patchSet['IntImage'].apply(lambda x: rectFeatures(type, x1, y1, x2, y2, x))
  
  patchSet = patchSet[['Value','Feature?']]
  
  
def weakClassifier(patchSet):
  """
  Reads in a DataFrame with column 'Feature?' that contains 0 or 1 whether it contains the feature
    ALERT - currently assumes DataFrame contains a column labeled 'Value' that contains the feature values for running time analysis
    
  Outputs the minimum error, threshold, and parity
    parity = 1 => feature value > threshold are classified as containing the desired feature
    parity = -1 => feature value =< threshold are classified as containing the desired feature
  """
    
  # create DataFrame that has 0 and 1 as column headings, feature value as rows, and number of occurrences as the table entries
  featureVals = patchSet['Feature?'].groupby(patchSet['Value']).value_counts().unstack()
  
  # fill empty values with 0
  featureVals.fillna(0, inplace=True)
  
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