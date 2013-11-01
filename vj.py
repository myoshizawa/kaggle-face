import random
import pandas as pd
from pandas import DataFrame, Series
import numpy as np
import face
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from time import time


def initialize(keypoint = 'left_eye_center'):
  """
  Input: keypoint = facial keypoint name (without _x or _y ending)
  Output: DataFrame with set of (default = 35228) 24x24 pixel training patches
    Patch records the pixel values in a numpy array
    Columns 0 and 1 record whether the facial keypoint is present in the patch
    IntImage has the integral image information in a numpy array
  """
  # read training data from .csv
  print 'Reading training data from train.csv'
  train = face.readTrain()
  # create training patch set (currently finds 1 patch containg keypoint (if possible), and 4 random patches that do not
  print 'Creating set of patches for %s' % keypoint
  patchSet = trainingPatches(train, keypoint)
  # calculate integral image for each patch
  print 'Calculating integral images'
  calcIntImage(patchSet)
  """
  # stores training data and training patch set in storage.h5
  storage = pd.HDFStore('storage.h5')
  storage['train'] = train
  storage['patchSet'] = patchSet
  storage.close()
  """
  return patchSet

  
def demoWeakClassifier(patchSet, type = 12, x1 = 0, y1 = 0, x2 = 2, y2 = 2):
  """
  Input: patchSet = DataFrame with training patches and columns IntImage, 0, 1
         type = 12, 21, 13, 31, or 22 (type of rectangular feature)
         x1, y1 = upper left coordinates
         x2, y2 = bottom right coordinates
  Output: Weak classifier information for the desired rectangular feature
  """
  # calculate values of desired feature and save in DataFrame values
  print 'Calculating values for feature (%d,%d,%d,%d,%d)' % (type,x1,y1,x2,y2)
  values = calcFeature(patchSet, type, x1, y1, x2, y2)
  
  # initialize weights
  weights = Series(np.ones(len(patchSet)))
  weights[patchSet[0]==1] = weights[patchSet[0]==1] / (2 * patchSet[0].sum())
  weights[patchSet[1]==1] = weights[patchSet[1]==1] / (2 * patchSet[1].sum())
  
  # initialize weightedVals
  weightedVals = DataFrame(index = patchSet.index)
  weightedVals[1] = patchSet[1] * weights
  weightedVals[0] = patchSet[0] * weights
  
  # create weak classifier for desired feature
  return weakClassifier(weightedVals, values)
  
  
def demoStrongClassifier(patchSet, featureList, threshold = 'default'):
  """
  Input: patchSet - DataFrame with columns 0, 1, and IntImage
         featureList - list of features desired in strong classifier
         threshold - how lenient the final strong classifier will be (lower value => higher detection rates but also higher false positive rates
                     default determined by error rates of weak classifiers
  Output: none
  Creates a strong classifier based on features in featureList and outputs performance details
  """
  # initialize weights
  weights = Series(np.ones(len(patchSet)))
  weights[patchSet[0]==1] = weights[patchSet[0]==1] / (2 * patchSet[0].sum())
  weights[patchSet[1]==1] = weights[patchSet[1]==1] / (2 * patchSet[1].sum())
  
  strong = DataFrame(columns = ['feature', 'error', 'threshold', 'parity', 'alpha'])
  
  for feature in featureList:
  
    # create DataFrame that has 0 and 1 as column headings, unique feature value as rows, and number of occurrences (weighted) as the table entries
    weightedVals = DataFrame(index = patchSet.index)
    weightedVals[1] = patchSet[1] * weights
    weightedVals[0] = patchSet[0] * weights
  
    # calculate feature values
    values = calcFeature(patchSet, feature[0], feature[1], feature[2], feature[3], feature[4])
    
    # calculate weak classifier
    weak = weakClassifier(weightedVals, values)
    
    if weak['parity'] == 1:
      classCorrect = values > weak['threshold']
    else:
      classCorrect = values < weak['threshold']
    # determine which samples were predicted correctly
    correct = classCorrect*patchSet[1] + (-classCorrect) * patchSet[0]
          
    beta = weak['error'] / (1 - weak['error'])
    alpha = np.log(1/beta)
    # update weights
    weights = updateWeights(weights, correct, beta)
    
    # add best weak classifier to strong classifier
    weak = weak.append(Series({'feature': feature, 'alpha': alpha}))
    strong = strong.append(weak, ignore_index = True)
    
  print 'Performance on Training Patches:'
  print runStrong(patchSet, strong, threshold)
  
  return strong
  
  
def visualizePred(train, strong, sample, threshold = 'default'):
  """
  Input: train - DataFrame containing image pixel values (output of face.readTrain())
         strong - DataFrame containing strong classifier (output of demoStrongClassifier())
         sample - Number of a sample in train
         threshold - Custom threshold level
  Outputs the image and plots all predictions of strong classifier
  """

  print 'Acquiring subwindows'
  sampleSet = testPatches(train['Image'][sample])
  
  print 'Obtaining predictions'
  sampleSet['pred'] = runStrong2(sampleSet, sample, strong, threshold)
  
  print 'Plotting results'
  plt.imshow(train['Image'][sample], cmap=cm.Greys_r)
  
  for i in sampleSet[sampleSet['pred'] == True].index:
    plt.scatter(sampleSet['x'][i],sampleSet['y'][i])
  
  
def visualizeFeature(patchSet, type, x1, y1, x2, y2):
  """
  Input: patchSet with column 'Patch'
         target feature in the form type, x1, y1, x2, y2
  Output: None
  Prints the average of all patches that contains the keypoint and the outlines of the identified feature
  """

  averagePatch = patchSet[patchSet[1] == 1]['Patch'].sum() / len(patchSet[patchSet[1]==1])
  
  plt.imshow(averagePatch, cmap=cm.Greys_r, interpolation='nearest')
  
  plt.plot([x1,x2],[y1,y1])
  plt.plot([x1,x2],[y2,y2])
  plt.plot([x1,x1],[y1,y2])
  plt.plot([x2,x2],[y1,y2])
  
  if type == 12:
    x12 = (x1 + x2)/2
    plt.plot([x12,x12],[y1,y2])
  elif type == 21:
    y12 = (y1 + y2)/2
    plt.plot([x1,x2],[y12,y12])
  elif type == 13:
    diff = (x2 - x1)/3
    x13 = x1 + diff
    x23 = x13 + diff
    plt.plot([x13,x13],[y1,y2])
    plt.plot([x23,x23],[y1,y2])
  elif type == 31:
    diff = (y2 - y1)/3
    y13 = y1 + diff
    y23 = y13 + diff
    print y13, y23
    plt.plot([x1,x2],[y13,y13])
    plt.plot([x1,x2],[y23,y23])
  elif type == 22:
    x12 = (x1 + x2)/2
    y12 = (y1 + y2)/2
    plt.plot([x12,x12],[y1,y2])
    plt.plot([x1,x2],[y12,y12])
    
  
def featureValues(patchSet):
  """
  Input: patchSet - DataFrame with column IntImage
  Output: none
  Calculates features for each integral image in patchSet and stores them in a local h5 storage file named storage.h5
  Features will be stored in 7 different data frames, named according to nameDict: '12-0', '12-1', '21-0', '21-1', '13', '31', '22'
  This process will likely take about 2 hours and will require a decent amount of memory
  """

  storage = pd.HDFStore('storage.h5')
  for i in xrange(7):
    # calculate feature values
    values = featureDict[i](patchSet)
    # store feature values in storage.h5
    storage[nameDict[i]] = values
  storage.close()

  
################################################################

def trainingPatches(data, keypoint, numWithout = 4):
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
      
      for i in range(numWithout):
      
        imageDict = {}
      
        contained = True
      
        patchCenterX = random.randint(patchRadius,95-patchRadius)
        patchCenterY = random.randint(patchRadius,95-patchRadius)
        
        imageDict['Patch'] = data['Image'][sample][patchCenterY - patchRadius:patchCenterY + patchRadius, patchCenterX - patchRadius:patchCenterX + patchRadius]
      
        patchList.append(imageDict)
        keypointYes.append(0)
        keypointNo.append(1)
      
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
    
    # report if feature location is too close to boundary
    if keypointLocX < patchRadius:
      print 'Sample %s is too close to left boundary at (%d,%d)' % (sample, keypointLocX, keypointLocY)
      continue
    elif keypointLocX > 95-patchRadius:
      print 'Sample %s is too close to right boundary at (%d,%d)' % (sample, keypointLocX, keypointLocY)
      continue
    elif keypointLocY < patchRadius:
      print 'Sample %s is too close to top boundary at (%d,%d)' % (sample, keypointLocX, keypointLocY)
      continue
    elif keypointLocY > 95-patchRadius:
      print 'Sample %s is too close to bottom boundary at (%d,%d)' % (sample, keypointLocX, keypointLocY)
      continue
  
    # if feature is not too close to boundary, record patch containing feature
        
    imageDict = {}
    imageDict['Patch'] = data['Image'][sample][keypointLocY - patchRadius:keypointLocY + patchRadius, keypointLocX - patchRadius:keypointLocX + patchRadius]
    
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
  f1 = ii[y1,x1]
  f2 = ii[y12,x1]
  f3 = ii[y2,x1]
  f4 = ii[y1,x2]
  f5 = ii[y12,x2]
  f6 = ii[y2,x2]
  
  return f6 - f5 - f5 + f4 - f3 + f2 + f2 - f1
  

def feature12(x1,y1,x2,y2,ii):
  """
  2-features parted horizontally
  """
  
  x12 = (x1 + x2) / 2
  f1 = ii[y1,x1]
  f2 = ii[y1,x12]
  f3 = ii[y1,x2]
  f4 = ii[y2,x1]
  f5 = ii[y2,x12]
  f6 = ii[y2,x2]
  
  return f6 - f5 - f5 + f4 - f3 + f2 + f2 - f1
  
  
def feature31(x1,y1,x2,y2,ii):
  """
  3-features sliced vertically
  """
  
  third = (y2 - y1)/3
  y13 = y1 + third
  y23 = y13 + third
  f1 = ii[y1,x1]
  f2 = ii[y13,x1]
  f3 = ii[y23,x1]
  f4 = ii[y2,x1]
  f5 = ii[y1,x2]
  f6 = ii[y13,x2]
  f7 = ii[y23,x2]
  f8 = ii[y2,x2]
  
  return f8 - f7 - f7 + f6 + f6 - f5 - f4 + f3 + f3 - f2 - f2 + f1

def feature13(x1,y1,x2,y2,ii):
  """
  3-features sliced horizontally
  """
  
  third = (x2 - x1)/3
  x13 = x1 + third
  x23 = x13 + third
  f1 = ii[y1,x1]
  f2 = ii[y1,x13]
  f3 = ii[y1,x23]
  f4 = ii[y1,x2]
  f5 = ii[y2,x1]
  f6 = ii[y2,x13]
  f7 = ii[y2,x23]
  f8 = ii[y2,x2]
  
  return f8 - f7 - f7 + f6 + f6 - f5 - f4 + f3 + f3 - f2 - f2 + f1
  
  
def feature22(x1,y1,x2,y2,ii):
  """
  4-features
  """
  
  x12 = (x1 + x2)/2
  y12 = (y1 + y2)/2
  f1 = ii[y1,x2]
  f2 = ii[y1,x12]
  f3 = ii[y1,x2]
  f4 = ii[y12,x1]
  f5 = ii[y12,x12]
  f6 = ii[y12,x2]
  f7 = ii[y2,x1]
  f8 = ii[y2,x12]
  f9 = ii[y2,x2]
  
  return f9 - f8 - f8 + f7 - f6 - f6 + f5 + f5 + f5 + f5 - f4 - f4 + f3 - f2 - f2 + f1
  
  

def calcFeature(patchSet, type, x1, y1, x2, y2):
  """
  Takes a dataFrame with integral images (under column 'IntImage') and adds a column '(type,x1,y1,x2,y2)' with the feature values
  """
  
  return patchSet['IntImage'].apply(lambda x: featureTypes[type](x1, y1, x2, y2, x)).astype(np.int32)
  
  
def featureValues12even(patchSet):
  """
  Calculates values for features using the integral images in patchSet
  2-rectangle horizontal features with even top-left x-coordinate
  """
  patchSize = len(patchSet['IntImage'][0])
  
  values = DataFrame(index = patchSet.index)

  for a in xrange(0,patchSize-1,2):
    for b in xrange(patchSize-1):
      for x in xrange(a+4,patchSize,4):
        for y in xrange(b+2,patchSize, 2):
          values[(12,a,b,x,y)] = calcFeature(patchSet,12,a,b,x,y)
  return values
  

def featureValues12odd(patchSet):
  """
  Calculates values for features using the integral images in patchSet
  2-rectangle horizontal features with odd top-left x-coordinate
  """
  patchSize = len(patchSet['IntImage'][0])
  
  values = DataFrame(index = patchSet.index)

  for a in xrange(1,patchSize-1,2):
    for b in xrange(patchSize-1):
      for x in xrange(a+4,patchSize,4):
        for y in xrange(b+2,patchSize,2):  
          values[(12,a,b,x,y)] = calcFeature(patchSet,12,a,b,x,y)
  return values
  
  
def featureValues21even(patchSet):
  """
  Calculates values for features using the integral images in patchSet
  2-rectangle vertical features with even top-left x-coordinate
  """
  patchSize = len(patchSet['IntImage'][0])
  
  values = DataFrame(index = patchSet.index)

  for a in xrange(patchSize-1):
    for b in xrange(0,patchSize-1,2):
      for x in xrange(a+2,patchSize,2):
        for y in xrange(b+4,patchSize,4):  
          values[(21,a,b,x,y)] = calcFeature(patchSet,21,a,b,x,y)
  return values


def featureValues21odd(patchSet):
  """
  Calculates values for features using the integral images in patchSet
  2-rectangle vertical features with odd top-left x-coordinate
  """
  patchSize = len(patchSet['IntImage'][0])
  
  values = DataFrame(index = patchSet.index)

  for a in xrange(patchSize-1):
    for b in xrange(1,patchSize-1,2):
      for x in xrange(a+2,patchSize,2):
        for y in xrange(b+4,patchSize,4):
          values[(21,a,b,x,y)] = calcFeature(patchSet,21,a,b,x,y)
  return values
  

def featureValues13(patchSet):
  """
  Calculates values for features using the integral images in patchSet
  3-rectangle horizontal features
  """
  patchSize = len(patchSet['IntImage'][0])
  
  values = DataFrame(index = patchSet.index)

  for a in xrange(patchSize-1):
    for b in xrange(patchSize-1):
      for x in xrange(a+6,patchSize,6):
        for y in xrange(b+2,patchSize,2):   
          values[(13,a,b,x,y)] = calcFeature(patchSet,13,a,b,x,y)
  return values
  
  
def featureValues31(patchSet):
  """
  Calculates values for features using the integral images in patchSet
  3-rectangle vertical features
  """
  patchSize = len(patchSet['IntImage'][0])
  
  values = DataFrame(index = patchSet.index)

  for a in xrange(patchSize-1):
    for b in xrange(patchSize-1):
      for x in xrange(a+2,patchSize,2):
        for y in xrange(b+6,patchSize,6):
          values[(31,a,b,x,y)] = calcFeature(patchSet,31,a,b,x,y)
  return values
  
  
def featureValues22(patchSet):
  """
  Calculates values for features using the integral images in patchSet
  4-rectangle features
  """
  patchSize = len(patchSet['IntImage'][0])
  
  values = DataFrame(index = patchSet.index)

  for a in xrange(patchSize-1):
    for b in xrange(patchSize-1):
      for x in xrange(a+4,patchSize,4):
        for y in xrange(b+4,patchSize,4):
          values[(22,a,b,x,y)] = calcFeature(patchSet,22,a,b,x,y)
  return values
            
  
def weakClassifier(weightedVals, values):
  """
  Input: patchSet is a DataFrame with columns 1, 0, and 'Weights', and '(type,x1,y1,x2,y2)'
    type, x1, y1, x2, y2 uniquely determines a feature
  Outputs the minimum error, threshold, and parity
    parity = 1 => feature value > threshold are classified as containing the facial keypoint
    parity = -1 => feature value =< threshold are classified as containing the facial keypoint
  """
  
  featureVals = weightedVals.groupby(values).sum()
  
  # diff calculates how much the error changes if we moved the threshold from the minimum feature value to the current feature value
  diff = (featureVals[1] - featureVals[0]).cumsum()
  
  # log the minimum error for both possible parities
  posError = featureVals[0].sum() + diff.min()
  negError = featureVals[1].sum() - diff.max()
  
  # return the least error
  if posError <= negError:
    return Series({'error': posError, 'threshold': diff.idxmin(), 'parity': 1})
  else:
    return Series({'error': negError, 'threshold': diff.idxmax(), 'parity': -1})


def updateWeights(weights, correct, beta):
  """
  Updates weights after an interation of adaBoost
  """
  weights = weights * (beta ** correct)
  weights = weights / weights.sum()
  return weights

  
def adaBoost(patchSet, numFeatures, strong = 0, weights = 0):
  """
  Input: patchSet with columns 0, 1
         numFeatures = number of features desired in strong classifier
         weights sets the initial weight values for each sample
  Output: Strong classifier information in a DataFrame
          last set of weights (can input into later runs to start adaBoost where previous run left off)
  """
  if type(weights) == int:
    # initialize weights
    weights = Series(np.ones(len(patchSet)))
    weights[patchSet[0]==1] = weights[patchSet[0]==1] / (2 * patchSet[0].sum())
    weights[patchSet[1]==1] = weights[patchSet[1]==1] / (2 * patchSet[1].sum())
  
  if type(strong) == int:
    strong = DataFrame(columns = ['feature', 'error', 'threshold', 'parity', 'alpha'])
    
  store = pd.HDFStore('storage.h5')
  
  for j in xrange(numFeatures):
  
    # create DataFrame that has 0 and 1 as column headings, unique feature value as rows, and number of occurrences (weighted) as the table entries
    weightedVals = DataFrame(index = patchSet.index)
    weightedVals[1] = patchSet[1] * weights
    weightedVals[0] = patchSet[0] * weights
  
    error = float('inf')
  
    # retrieve each set of features and determine minimum error
    for i in xrange(7):
  
      featureVals = store[nameDict[i]]
            
      errorVal = featureVals.apply(lambda x: weakClassifier(weightedVals, x))
      
      curFeature = errorVal.ix['error',:].idxmin()
      curError = errorVal[curFeature]['error']
      
      # keep track of feature that provides minimum error, its weak classifier, and which samples were classified correctly
      if curError < error:
        feature = curFeature
        error = curError
        weak = errorVal[feature]
        
        # determine which samples were classified as containing feature by weak classifier
        if weak['parity'] == 1:
          classCorrect = featureVals[feature] > weak['threshold']
        else:
          classCorrect = featureVals[feature] < weak['threshold']
        # determine which samples were correctly classified by weak classifier
        correct = classCorrect*patchSet[1] + (-classCorrect) * patchSet[0]
          
    beta = error / (1 - error)
    alpha = np.log(1/beta)
    # update weights
    weights = updateWeights(weights, correct, beta)
    
    # add best weak classifier to strong classifier
    print 'Added feature:'
    print weak
    
    weak = weak.append(Series({'feature': feature, 'alpha': alpha}))
    strong = strong.append(weak, ignore_index = True)
    
  store.close()
    
  return strong, weights
  
  
def runStrong(patchSet, strongClassifier, threshold = 'default'):
  """
  Input: patchSet DataFrame with 1, 0, and IntImage columns
         strongClassifier DataFrame with alpha, feature, parity, threshold columns
         threshold = custom threshold for strong classifier (defaults to threshold = 1/2 sum of alpha values)
  Output: error, detection rate, and false positive rate of strong classifier
  """
  
  # if no threshold inputted, 
  if threshold == 'default':
    threshold = strongClassifier['alpha'].sum() / 2
    
  print 'Testing with threshold %f' % threshold
  
  predictionValues = Series(np.zeros(len(patchSet)), index = patchSet.index)
  
  # for each weak classifier in strong classifier, add weighted value to those classified as containing keypoint
  for i in strongClassifier.index:
    type, x1, y1, x2, y2 = strongClassifier['feature'][i]
    # calculate feature for weak classifier
    featureValues = calcFeature(patchSet, type, x1, y1, x2, y2)
    # make prediction
    if strongClassifier['parity'][i] == 1:
      predictYes = featureValues > strongClassifier['threshold'][i]
    else:
      predictYes = featureValues < strongClassifier['threshold'][i]
    # add alpha value  
    predictionValues = predictionValues + strongClassifier['alpha'][i] * predictYes
    
  # determine predictions (score must be above threshold)
  pred = (predictionValues > threshold)
  # actual results
  actual = (patchSet[1] == 1)
  # error rate
  error = (pred != actual).sum() / float(len(patchSet))
  # detection rate (percent of patches containing keypoint correctly classified as so)
  detectRate = (patchSet[1] * pred).sum() / float(patchSet[1].sum())
  # false positive rate (percent of patches not containing keypoint classified as containing it)
  falsePosRate = (patchSet[0] * pred).sum() / float(patchSet[0].sum())
  
  return Series({'error': error, 'detection_rate': detectRate, 'false_positive_rate': falsePosRate})
  
  
def testPatches(image):
  """
  Returns all possible subwindows for entered image
  """
  patchList = []
  centerListX = []
  centerListY = []
  
  imageLength = len(image)
  
  for i in xrange(patchRadius, imageLength - patchRadius):
    for j in xrange(patchRadius, imageLength - patchRadius):
      imageDict = {}
      
      imageDict['Patch'] = image[j-patchRadius:j+patchRadius,i-patchRadius:i+patchRadius]
      
      centerListX.append(i)
      centerListY.append(j)
      patchList.append(imageDict)
        
  testPatchSet = DataFrame(patchList)
  testPatchSet['x'] = centerListX
  testPatchSet['y'] = centerListY
  
  return testPatchSet
  
  
def runStrong2(testSet, sample, strongClassifier, threshold = 'default'):
  """
  version of runStrong that calculates features directly, rather than with integral image
  """
  
  # if no threshold inputted, 
  if threshold == 'default':
    threshold = strongClassifier['alpha'].sum() / 2
    
  print 'Testing with threshold %f' % threshold
  
  predictionValues = Series(np.zeros(len(testSet)), index = testSet.index)
  
  # for each weak classifier in strong classifier, add weighted value to those classified as containing keypoint
  for i in strongClassifier.index:
    type, x1, y1, x2, y2 = strongClassifier['feature'][i]
    # calculate feature for weak classifier
    if type == 12:
      x12 = (x1 + x2)/2
      featureValues = testSet['Patch'].apply(lambda x: x[y1:y2,x12:x2].sum() - x[y1:y2,x1:x12].sum())
    elif type == 21:
      y12 = (y1 + y2)/2
      featureValues = testSet['Patch'].apply(lambda x: x[y12:y2,x1:x2].sum() - x[y1:y12,x1:x2].sum())
    elif type == 13:
      diff = (x2 - x1) / 3
      x13 = x1 + diff
      x23 = x13 + diff
      featureValues = testSet['Patch'].apply(lambda x: x[y1:y2,x1:x13].sum() + x[y1:y2,x23:x2].sum() - x[y1:y2,x13:x23].sum())
    elif type == 31:
      diff = (y2 - y1) / 3
      y13 = y1 + diff
      y23 = y13 + diff
      featureValues = testSet['Patch'].apply(lambda x: x[y1:y13,x1:x2].sum() + x[y23:y2,x1:x2].sum() - x[y13:y23,x1:x2].sum())
    elif type == 22:
      x12 = (x1 + x2)/2
      y12 = (y1 + y2)/2
      featureValues = testSet['Patch'].apply(lambda x: x[y1:y12,x1:x12].sum() + x[y12:y2,x12:x2].sum() - x[y1:y12,x12:x2].sum() - x[y12:y2,x1:x12])
    # make prediction
    if strongClassifier['parity'][i] == 1:
      predictYes = featureValues > strongClassifier['threshold'][i]
    else:
      predictYes = featureValues < strongClassifier['threshold'][i]
    # add alpha value  
    predictionValues = predictionValues + strongClassifier['alpha'][i] * predictYes
    
  # determine predictions (score must be above threshold)
  return predictionValues > threshold

  
    
featureTypes = {12: feature12, 21: feature21, 13: feature13, 31: feature31, 22: feature22}
nameDict = {0: '12-0', 1: '12-1', 2: '21-0', 3: '21-1', 4: '13', 5: '31', 6: '22'}
featureDict = {0: featureValues12even, 1: featureValues12odd, 2: featureValues21even, 3: featureValues21odd, 4: featureValues13, 5: featureValues31, 6: featureValues22}

patchRadius = 12

"""
To Do:
  [Y & S] Optimize adaBoost?
  [T] Function that takes a test image and produces all possible 24x24 subwindows
  [T] Function that runs strong classifier on all subwindows
  Substep: Function that graphs successes from strong classifier
  Function that, given results of strong classifier, predicts location of keypoint
"""