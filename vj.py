import random
import pandas as pd
from pandas import DataFrame, Series
import numpy as np
import csv
import features
from time import time

def trainingPatches(data, feature, patchRadius = 12, numWith = 4, numWithout = 8):
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
  Reads in a DataFrame with a column 'Patch' and returns a column with integral images calculated
  """
  ii = []
  
  for sample in trainingSet.index:
    ii.append(features.integral_image(trainingSet['Patch'][sample]))
    
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

  
def weakClassifier(patchSet, feature):
  """
  Reads in a DataFrame with column 'Feature?' that contains 0 or 1 whether it contains the feature
                            column 'IntImage' that contains the integral image data for the training patch
  Outputs the minimum error, threshold, and parity
    parity = 1 => feature value > threshold are classified as containing the desired feature
    parity = -1 => feature value =< threshold are classified as containing the desired feature
  """
  
  # initialize dict
  featureDict = {}
  featureDict[0] = {}
  featureDict[1] = {}
  
  # calculate feature for each patch and update dict
  for sample in patchSet.index:
    # ii = features.integral_image(patchSet['Patch'][sample])
    value = features.rect_features(feature[0],feature[1],feature[2],feature[3],feature[4],patchSet['IntImage'][sample])
    
    if value in featureDict[patchSet['Feature?'][sample]]:
      featureDict[patchSet['Feature?'][sample]][value] += 1
    else:
      featureDict[patchSet['Feature?'][sample]][value] = 1

  # convert dict into DataFrame, fill empty values with zeros
  featureVals = DataFrame(featureDict)
  featureVals = featureVals.fillna(0)
  featureVals = featureVals.sort_index()
  
  # initialize totals
  total0 = featureVals[0].sum()
  total1 = featureVals[1].sum()
  total = total0 + total1

  # initialize starting error
  minError = total
  maxError = 0
  
  # initialize starting Type I errors
  currentMinType1 = 0
  currentMaxType1 = total1
  
  minType1 = 0
  maxType1 = total1
  
  # initialize starting Type I + II error
  currentError = total0
  
  for i in featureVals.index:
    # update current error and Type I errors
    currentError += featureVals[1][i] - featureVals[0][i]
    currentMinType1 += featureVals[1][i]
    currentMaxType1 -= featureVals[1][i]
    
    # if error is a new minimum, record it
    if currentError < minError:
      minError = currentError
      minThreshold = i
      minType1 = currentMinType1
    # if there is a tie but Type I error is less, record it
    elif currentError == minError and currentMinType1 < minType1:
      minError = currentError
      minThreshold = i
      minType1 = currentMinType1
    
    # if error is a new maximum, record it
    if currentError > maxError:
      maxError = currentError
      maxThreshold = i
      maxType1 = currentMaxType1
    # if there is a tie but Type I error is less, record it
    elif currentError == maxError and currentMaxType1 < maxType1:
      maxError = currentError
      maxThreshold = i
      maxType1 = currentMaxType1

  # if total - maxError < minError, then switch parity
  if total - maxError < minError:
    return total - maxError, maxThreshold, -1
  # if there is a tie, go with parity that produces lowest Type I error
  elif total - maxError == minError:
    if maxType1 > minType1:
      return total - maxError, maxThreshold, -1
    else:
      return minError, minThreshold, 1
  # otherwise, maintain parity
  else:
    return minError, minThreshold, 1