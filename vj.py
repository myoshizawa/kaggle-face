import random
from pandas import DataFrame
import numpy as np
import csv

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
  
  
def writeH5(data, title, filename):
  """
  Writes data to filename.h5 with key title
  """
  store = pd.HDFStore(filename + '.h5')
  
  store[title] = data
  
  store.close()

  
def readH5(title, filename):
  """
  Reads data from filename.h5 with key title
  """
  store = pd.HDFStore(filename + '.h5')
  
  return store[title]