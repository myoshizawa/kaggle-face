import random
import pandas as pd
from pandas import DataFrame, Series
import numpy as np
import face
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
import UnionFind as UF
import cython_mod as cy


def initialize(keypoint = 'left_eye_center'):
  """
  Returns training data and an initial training set of patches as DataFrames
  Input: keypoint = facial keypoint name (without _x or _y ending)
  Output: 
    train:
      DataFrame with training images as numpy arrays and facial keypoint locations
    patchSet:
      DataFrame with training patches as numpy arrays of size patchRadius (global var)
      A 1 in columns 0 and 1 records whether the facial keypoint is not present or present respectively (0 otherwise)
      IntImage column contains integral image in a numpy array
  Running time: 54 s
  """
  # read training data from .csv
  print 'Reading training data from train.csv'
  train = face.readTrain()
  # create training patch set (currently finds 1 patch containg keypoint (if possible), and 4 random patches that do not
  print 'Creating initial set of patches for %s' % keypoint
  patchSet = trainingPatches(train, keypoint = keypoint)
  """
  # stores training data and training patch set in storage.h5
  storage = pd.HDFStore('storage.h5')
  storage['train'] = train
  storage['patchSet'] = patchSet
  storage.close()
  """
  return train, patchSet

def visualizePred(image, strong, threshold = None):
  """
  Plots all positive predictions of classifier over image
  Input: image = numpy array with pixel values
         strong = single or list of DataFrames containing strong classifier (output of demoStrongClassifier())
         threshold = singe or list of thresholds
  Outputs the image and plots all predictions of strong classifier (red is prediction with highest score)
  Running time: 2.47 s
  """
  # find all subwindows from image
  print 'Acquiring subwindows'
  sampleSet = cy.getSubwindows(image)
  
  calcIntImage(sampleSet)
  
  # convert strong to cascade, if necessary
  if not isinstance(strong, list):
    strong = [strong]
    if threshold is not None:
      threshold = [threshold]
    
  # obtain predictions
  pred = cascadePred(sampleSet, strong, threshold)
    
  numDetections = pred.sum()
  numSamples = len(sampleSet)
  
  print 'Number of detections: %d / %d' % (numDetections, numSamples)
  
  # print image and positive detections
  plt.imshow(image, cmap=cm.Greys_r)
  
  for i in sampleSet[pred == True].index:
    plt.scatter(sampleSet['x'][i], sampleSet['y'][i], color = 'blue')

  
def visualizeFeature(patchSet, type, x1, y1, x2, y2):
  """
  Input: patchSet = DataFrame with column 'Patch'
         type, x1, y1, x2, y2 = target feature
  Output: None
  Prints the average of all patches that contains the keypoint and the outlines of the identified feature
  Running time: 50.3 ms
  """
  # calculate average patch of all patches containing facial keypoint
  averagePatch = patchSet[patchSet[1] == 1]['Patch'].sum() / len(patchSet[patchSet[1]==1])
  
  plt.imshow(averagePatch, cmap=cm.Greys_r, interpolation='nearest')
  
  # plot feature rectangles
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
    
def calcFeatureValues(patchSet):
  """
  Input: patchSet - DataFrame with column IntImage
  Output: none
  Calculates features for each integral image in patchSet and stores them in a local h5 storage file named storage.h5
  Features will be stored in 5 different data frames, named according to nameDict: '12', '21', '13', '31', '22'
  This process will likely take about 2 hours
  """

  storage = pd.HDFStore('storage.h5')
  
  storage['12'] = featureValues(patchSet, 12)
  print 'Features 12 completed'
  storage['21'] = featureValues(patchSet, 21)
  print 'Features 21 completed'
  storage['13'] = featureValues(patchSet, 13)
  print 'Features 13 completed'
  storage['31'] = featureValues(patchSet, 31)
  print 'Features 31 completed'
  storage['22'] = featureValues(patchSet, 22)
  print 'Features 22 completed'
  
  storage.close()

  
################################################################

def trainingPatches(data, strong = False, threshold = False, keypoint = 'left_eye_center', numWithout = 1, maxTries = 100):
  """
  Input: data = training data with columns 'Image' and location of keypoint
         strong = single or cascade (list) of strong classifier DataFrames (new patches will only be selected if they test positive)
         threshold = threshold or list of thresholds for strong classifiers
         keypoint = facial keypoint
         numWithout = number of patches selected from each image that do not contain keypoint (default is 1)
  Output: 4-column DataFrame of patches
    Patch: contains square patches (side length = 2 * patchRadius) as a numpy array
    1: a 1 in this column means facial keypoint present in patch (0 otherwise)
    0: a 1 in this column means facial keypoint not present (0 otherwise)
    IntImage: integral image of patches
  Does not record patches containing keypoint if keypoint is too close (within patchRadius) to the boundary of image or sample has no keypoint information
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
        # if no strong classifier, select patch randomly
        if not strong:
          patchCenterX = random.randint(patchRadius,95-patchRadius)
          patchCenterY = random.randint(patchRadius,95-patchRadius)
          imageDict = {'Patch': data['Image'][sample][patchCenterY - patchRadius:patchCenterY + patchRadius, patchCenterX - patchRadius:patchCenterX + patchRadius]}
          patchList.append(imageDict)
          keypointYes.append(0)
          keypointNo.append(1) 
        # if strong classifier, select patch that is classified positive by the strong classifier
        else:
          flag = True
          numTries = 0
          
          while flag:
            patchCenterX = random.randint(patchRadius,95-patchRadius)
            patchCenterY = random.randint(patchRadius,95-patchRadius)
            patch = data['Image'][sample][patchCenterY - patchRadius:patchCenterY + patchRadius, patchCenterX - patchRadius:patchCenterX + patchRadius]
            if predPatch(patch, strong, threshold):
              imageDict = {'Patch': data['Image'][sample][patchCenterY - patchRadius:patchCenterY + patchRadius, patchCenterX - patchRadius:patchCenterX + patchRadius]}
              patchList.append(imageDict)
              keypointYes.append(0)
              keypointNo.append(1)   
              flag = False
            elif numTries > maxTries:
              flag = False
            numTries += 1
         
      continue
    
    # if keypoint location is present, record patches that do not contain keypoint
    for i in range(numWithout):
      # if no strong classifier, select patch so that it does not contain keypoint
      if not strong:
        contained = True
        
        # ensure that patch does not contain keypoint
        while contained:
          patchCenterX = random.randint(patchRadius,95-patchRadius)
          patchCenterY = random.randint(patchRadius,95-patchRadius)
          if (patchCenterX < int(keypointLocX - patchRadius) or patchCenterX > int(keypointLocX + patchRadius)) and (patchCenterY < int(keypointLocY - patchRadius) or patchCenterY > int(keypointLocY + patchRadius)):
            imageDict = {'Patch': data['Image'][sample][patchCenterY - patchRadius:patchCenterY + patchRadius, patchCenterX - patchRadius:patchCenterX + patchRadius]}
            patchList.append(imageDict)
            keypointNo.append(1)
            keypointYes.append(0)
            contained = False
      # if strong classifier, select patch so that it does not contain keypoint and is classified positive by strong classifier
      else:
        flag = True
        numTries = 0
        while flag:
          patchCenterX = random.randint(patchRadius,95-patchRadius)
          patchCenterY = random.randint(patchRadius,95-patchRadius)
          if (patchCenterX < int(keypointLocX - patchRadius) or patchCenterX > int(keypointLocX + patchRadius)) and (patchCenterY < int(keypointLocY - patchRadius) or patchCenterY > int(keypointLocY + patchRadius)):
            patch = data['Image'][sample][patchCenterY - patchRadius:patchCenterY + patchRadius, patchCenterX - patchRadius:patchCenterX + patchRadius]
            if predPatch(patch, strong, threshold):
              imageDict = {'Patch': data['Image'][sample][patchCenterY - patchRadius:patchCenterY + patchRadius, patchCenterX - patchRadius:patchCenterX + patchRadius]}
              patchList.append(imageDict)
              keypointNo.append(1)
              keypointYes.append(0)
              flag = False
            elif numTries > maxTries:
              flag = False
            numTries += 1
    
    # report if keypoint location is too close to boundary
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
  
    # if keypoint is not too close to boundary, record patch containing keypoint
    imageDict = {'Patch': data['Image'][sample][keypointLocY - patchRadius:keypointLocY + patchRadius, keypointLocX - patchRadius:keypointLocX + patchRadius]}
    patchList.append(imageDict)
    keypointYes.append(1)
    keypointNo.append(0)

  # combine lists into a single DataFrame
  trainingSet = DataFrame(patchList)
  trainingSet[1] = keypointYes
  trainingSet[0] = keypointNo
  
  # calculate integral images
  calcIntImage(trainingSet)
  
  return trainingSet

  
def calcIntImage(trainingSet):
  """
  Reads in a DataFrame with a column 'Patch' and returns DataFrame with a new column containing integral images
  """
    
  trainingSet['IntImage'] = trainingSet['Patch'].apply(integralImage)

def integralImage(patch, normalize = True):
  """
  This function computes the integral image after normalizing the original patch 
  INPUT: An NxN numpy array representing an image
  OUTPUT: The corresponding normalized integral image 
  where the (x,y) coordinate gives you the sum of the pixels in the rectangle between (0,0) and (x,y)
  note that if x or y is 0, the integral image is 0
  """
 
  N=len(patch)
  int_im = np.zeros((N+1,N+1))
  
  int_im[1:,1:] = np.cumsum(np.cumsum(patch, axis=0), axis=1)
  
  if normalize:
    var_sqrt = math.sqrt(math.fabs((int_im[N][N]/patch.size)**2 - ((patch*patch).sum() / patch.size)))
    return (int_im / var_sqrt).astype(np.int)
  else:
    return int_im.astype(np.int)

def featureValues(patchSet, type):
  """
  Input: patchSet = DataFrame of patches as numpy arrays
         type = type of feature (12, 21, 13, 31, or 22)
  Returns 'all' feature values of given type for patches in patchSet
  """
  
  values = DataFrame(cy.featureValues(patchSet['IntImage'],type), index = patchSet.index)
  values.columns = pd.MultiIndex.from_tuples(values.columns, names = ['type', 'x1', 'y1', 'x2', 'y2'])
          
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

  
def adaBoost(patchSet, numFeatures, strong = None):
  """
  Input: patchSet with columns 0, 1
         numFeatures = number of features desired in strong classifier
         weights sets the initial weight values for each sample
  Output: Strong classifier information in a DataFrame
          last set of weights (can input into later runs to start adaBoost where previous run left off)
  """
  if strong is None:
    # initialize weights
    weights = Series(np.ones(len(patchSet)))
    weights[patchSet[0]==1] = weights[patchSet[0]==1] / (2 * patchSet[0].sum())
    weights[patchSet[1]==1] = weights[patchSet[1]==1] / (2 * patchSet[1].sum())
    
    # initialize strong classifier DataFrame
    strong = DataFrame(columns = ['feature', 'error', 'threshold', 'parity', 'alpha'])
  else:
    # determine weights from previous weak classifiers in strong
    weights = getWeights(patchSet, strong)
    
  store = pd.HDFStore('storage.h5')
  
  for j in xrange(numFeatures):
  
    # create DataFrame that has 0 and 1 as column headings, unique feature value as rows, and number of occurrences (weighted) as the table entries
    weightedVals = DataFrame(index = patchSet.index)
    weightedVals[1] = patchSet[1] * weights
    weightedVals[0] = patchSet[0] * weights
  
    error = float('inf')
  
    # retrieve each set of features and determine minimum error
    for i in xrange(5):
  
      featureVals = store[nameDict[i]]

      # train weak classifiers on every feature in featureVals
      weaks = featureVals.apply(lambda x: weakClassifier(weightedVals, x))
      
      curFeature = weaks.ix['error',:].idxmin()
      curError = weaks[curFeature]['error']
      
      # keep track of feature that provides minimum error, its weak classifier, and which samples were classified correctly
      if curError < error:
        feature = curFeature
        error = curError
        weak = weaks[feature]
        
        # determine which samples were classified as containing feature by weak classifier
        if weak['parity'] > 0:
          classCorrect = featureVals[feature] > weak['threshold']
        else:
          classCorrect = featureVals[feature] < weak['threshold']
        
        # determine which samples were correctly classified by weak classifier
        correct = classCorrect*patchSet[1] + (-classCorrect) * patchSet[0]
       
    beta = error / (1 - error)
    alpha = math.log(1/beta)
    # update weights (reduces weight for correctly classified samples and normalizes)
    weights = updateWeights(weights, correct, beta)
    
    # add best weak classifier to strong classifier
    print 'Added feature %d:' % (len(strong)+1)
    print weak
    
    weak = weak.append(Series({'feature': feature, 'alpha': alpha}))
    strong = strong.append(weak, ignore_index = True)
    
  store.close()
    
  return strong
  
def getSubwindows(image, spacing = 1):
  """
  Input: pixel values of image as numpy array
  Output: DataFrame containing all patches of size patchRadius (global var), center of patch, and their integral images
  """
  patchList = []
  centerListX = []
  centerListY = []
  
  imageLength = len(image)
  
  # add patches to list
  for i in xrange(patchRadius, imageLength - patchRadius, spacing):
    for j in xrange(patchRadius, imageLength - patchRadius, spacing):
      imageDict = {}
      
      imageDict['Patch'] = image[j-patchRadius:j+patchRadius,i-patchRadius:i+patchRadius]
      
      centerListX.append(i)
      centerListY.append(j)
      patchList.append(imageDict)

  # convert output to a DataFrame
  testPatchSet = DataFrame(patchList)
  testPatchSet['x'] = centerListX
  testPatchSet['y'] = centerListY
  
  calcIntImage(testPatchSet)
  
  return testPatchSet
  
  
def findThreshold(patchSet, strong, minDetectionRate = .999):
  """
  Determines (approx.) maximum threshold that produces a detection rate > user inputted rate
  Input: patchSet = DataFrame with patches
         strong = strong classifier DataFrame
         minDetectionRate = the minimum detection rate desired (in decimal form)
  """
  
  # set initial threshold
  threshold = strong['alpha'].sum()
  
  numSuccesses = 0
  spacingChange = {0: 5, 1: 1, 2: 0.5, 3: 0.1, 4: 0.05, 5: 0.01}
  
  flag = True
  
  while flag:
    report = strongReport(patchSet[patchSet[1] == 1], strong, threshold)
    
    if report['detect_rate'] < minDetectionRate:
      threshold -= spacingChange[numSuccesses]
    else:
      numSuccesses += 1
      if numSuccesses == len(spacingChange):
        print strongReport(patchSet, strong, threshold)
        return threshold
      threshold += spacingChange[numSuccesses-1] - spacingChange[numSuccesses]

  
def getWeights(patchSet, strong):
  """
  Returns adaBoost weights on samples in patchSet after using strong classifier
  Input: patchSet = DataFrame with columns IntImage, 0, 1
         strong = strong classifier DataFrame
  Output: Series containing weights
  """
  # initialize weights
  weights = Series(np.ones(len(patchSet)))
  weights[patchSet[0]==1] = weights[patchSet[0]==1] / (2 * patchSet[0].sum())
  weights[patchSet[1]==1] = weights[patchSet[1]==1] / (2 * patchSet[1].sum())
  
  for i in strong.index:
  
    # calculate feature values
    type, x1, y1, x2, y2 = strong['feature'][i]
    values = cy.calcFeature(patchSet['IntImage'], type, x1, y1, x2, y2)
    
    # determine predictions of weak classifier
    classCorrect = values * strong['parity'][i] > strong['threshold'][i] * strong['parity'][i]
    
    # determine which samples were predicted correctly
    correct = classCorrect*patchSet[1] + (-classCorrect) * patchSet[0]
    
    # determine beta from alpha value
    beta = np.exp(-strong['alpha'][i])

    # update weights
    weights = updateWeights(weights, correct, beta)
  
  return weights
  
  
def runStrong(patchSet, strong):
  """
  Runs strong classifier on set of patches and returns predictions
  Input: patchSet DataFrame with 1, 0, and IntImage columns
         strong DataFrame with alpha, feature, parity, threshold columns
         threshold = custom threshold for strong classifier (defaults to threshold = 1/2 sum of alpha values)
  Output: error, detection rate, and false positive rate of strong classifier
  """
  
  predVals = Series(np.zeros(len(patchSet)), index = patchSet.index)
  
  # for each weak classifier in strong classifier, add weighted value to those classified as containing keypoint
  for i in strong.index:
    type, x1, y1, x2, y2 = strong['feature'][i]
    # calculate feature for weak classifier
    featureVals = cy.calcFeature(patchSet['IntImage'], type, x1, y1, x2, y2)
    # make prediction
    if strong['parity'][i] == 1:
      predictYes = featureVals > strong['threshold'][i]
    else:
      predictYes = featureVals < strong['threshold'][i]
    # add alpha value  
    predVals += strong['alpha'][i] * predictYes
    
  return predVals


def strongReport(patchSet, strong, threshold = None):
  """
  Runs strong classifier on set of patches and returns error rate, detection rate, and false positive rate
  Input: patchSet = DataFrame of patches
         strong = strong classifier DataFrame
         threshold = custom threshold for strong classifier (default will be 1/2 sum of alpha values)
  Output: Series containing error, detect_rate, and false_positive_rate
  """
  # if threshold is None, set threshold to default (1/2 sum of alpha values)
  if threshold is None:
    threshold = strong['alpha'].sum() / 2
    print 'Testing with default threshold %f' % threshold
    
  # obtain scores from strong classifier
  predVals = runStrong(patchSet, strong)
  
  # determine predictions (score must be above threshold)
  pred = (predVals > threshold)
  
  # actual results
  actual = (patchSet[1] == 1)
  # error rate
  error = (pred != actual).sum() / float(len(patchSet))
  # detection rate (percent of patches containing keypoint correctly classified as so)
  detect = (patchSet[1] * pred).sum() / float(patchSet[1].sum())
  # false positive rate (percent of patches not containing keypoint classified as containing it)
  falsePos = (patchSet[0] * pred).sum() / float(patchSet[0].sum())
  
  return Series({'error': error, 'detect_rate': detect, 'false_positive_rate': falsePos})
  
  
def strongPred(patchSet, strong, threshold = None):
  """
  Runs strong classifier on set of patches and returns predictions and highest scoring sample
  Input: patchSet = DataFrame of patches
         strong = strong classifier DataFrame
         threshold = custom threshold for strong classifier (default will be 1/2 sum of alpha values)
  Output: Series of bools containing predictions, values given by strong classifier
  """
  
  # if threshold is None, set threshold to default (1/2 sum of alpha values)
  if threshold is None:
    threshold = strong['alpha'].sum() / 2
    print 'Testing with default threshold %f' % threshold
  
  # obtain scores from strong classifier
  predVals = runStrong(patchSet, strong)
  
  # return bool of predictions and value of highest scoring prediction
  return predVals >= threshold


  
def cascadePred(patchSet, cascade, thresholds = None):
  """
  Runs cascade of strong classifiers and returns predictions on patchSet
  Input: patchSet = DataFrame of patches
         cascade = list of strong classfiiers
         thresholds = list of threshold values
  Output: predictions by cascade and highest scoring sample
  """
  # if thresholds is None, set to default
  if thresholds is None:
    thresholds = []
    for strong in cascade:
      thresholds.append(strong['alpha'].sum() / 2)

  # patches to be tested by subsequent classifier
  remainingPatches = patchSet
  
  # each strong classifier makes predictions, update scores, repeat only on positive predictions
  for i in xrange(len(cascade)):
    classifiedYes = strongPred(remainingPatches, cascade[i], thresholds[i])
    remainingPatches = remainingPatches.ix[classifiedYes]
  
  # positive predictions are the remaining patches
  return Series(patchSet.index.isin(remainingPatches.index))

  
def cascadeReport(patchSet, cascade, thresholds = None):
  """
  Runs cascade of strong classifiers and returns error rate, detection rate, and false positive rate
  Input: patchSet = DataFrame of patches
         cascade = list of strong classfiiers
         thresholds = list of threshold values
  Output: Series containing error, detection_rate, and false_positive_rate
  """
  # if thresholds is None, set to default
  if thresholds is None:
    thresholds = []
    for strong in cascade:
      thresholds.append(strong['alpha'].sum() / 2)

  # patches to be tested by subsequent classifier
  remainingPatches = patchSet
  
  # each strong classifier makes predictions, repeat only on positive predictions
  for i in xrange(len(cascade)):
    classifiedYes = strongPred(remainingPatches, cascade[i], thresholds[i])
    remainingPatches = remainingPatches.ix[classifiedYes]
  
  # positive predictions are the remaining patches  
  pred = patchSet.index.isin(remainingPatches.index)

  actual = (patchSet[1] == 1)
  error = (pred != actual).sum() / float(len(patchSet))
  detectRate = (patchSet[1] * pred).sum() / float(patchSet[1].sum())
  falsePosRate = (patchSet[0] * pred).sum() / float(patchSet[0].sum())
  
  return Series({'error': error, 'detection_rate': detectRate, 'false_positive_rate': falsePosRate})
  
  
def predPatch(patch, cascade, thresholds = None):
  """
  Input: patch = numpy array containing pixel values
         cascade = strong classifier DataFrame with columns feature, parity, threshold, and alpha
         threshold = custom threshold for strong classifier
  Returns True/False whether cascade classifies patch as containing facial keypoint
  """
  # if cascade and thresholds are not lists, make them lists
  if not isinstance(cascade, list):
    cascade = [cascade]
    
    if thresholds is not None:
      thresholds = [thresholds]
  
  # if thresholds is None, set to defaults
  if thresholds is None:
    thresholds = []
    for strong in cascade:
      thresholds.append(strong['alpha'].sum() / 2)

  # determine integral image for patch
  ii = integralImage(patch)
  
  for i in xrange(len(cascade)):
    weakSum = 0
    for j in xrange(len(cascade[i])):
      type, x1, y1, x2, y2 = cascade[i]['feature'][j]
      value = featureTypes[type](x1, y1, x2, y2, ii)
      if cascade[i]['parity'][j] > 0:
        if value > cascade[i]['threshold'][j]:
          weakSum += cascade[i]['alpha'][j]
      else:
        if value < cascade[i]['threshold'][j]:
          weakSum += cascade[i]['alpha'][j]
    if weakSum < thresholds[i]:
      return False
      
  return True


  
def trainingPatches2(data, keypoint1 = 'left_eye_center', keypoint2 = 'right_eye_center', fudge = (0,0)):
  """
  Variant of trainingPatches() that creates a patch set consisting of patches centered at keypoint1 and patches centered at keypoint2
  Input: data = training data with columns 'Image' and location of keypoint
         keypoint1 = facial keypoint you are trying to classify
         keypoint2 = second facial keypoint that you are not trying to classify
  Output: 4-column DataFrame of patches
    Patch: contains square patches (side length = 2 * patchRadius) as a numpy array
    1: a 1 in this column means facial keypoint present in patch (0 otherwise)
    0: a 1 in this column means facial keypoint not present (0 otherwise)
    IntImage: integral image of patches
  Does not record patches containing a keypoint if the keypoint is too close (within patchRadius) to the boundary of image or sample has no keypoint information
  """

  # will become the columns of the returned DataFrame
  patchList = []
  keypointYes = []
  keypointNo = []
  
  for sample in data.index:
  
    # extract keypoint locations
    keypoint1LocX = data[keypoint1 + '_x'][sample]
    keypoint1LocY = data[keypoint1 + '_y'][sample]
    
    keypoint2LocX = data[keypoint2 + '_x'][sample] + fudge[0]
    keypoint2LocY = data[keypoint2 + '_y'][sample] + fudge[1]
    
    # if keypoint1 location is not present, report
    if np.isnan(keypoint1LocX) or np.isnan(keypoint1LocY):
      print 'Sample %s does not have %s data' % (sample, keypoint1) 
    # report if keypoint1 location is too close to boundary
    elif keypoint1LocX < patchRadius:
      print 'Keypoint %s of sample %d is too close to left boundary at (%d,%d)' % (keypoint1, sample, keypoint1LocX, keypoint1LocY)
    elif keypoint1LocX > 95-patchRadius:
      print 'Keypoint %s of sample %d is too close to right boundary at (%d,%d)' % (keypoint1, sample, keypoint1LocX, keypoint1LocY)
    elif keypoint1LocY < patchRadius:
      print 'Keypoint %s of sample %d is too close to top boundary at (%d,%d)' % (keypoint1, sample, keypoint1LocX, keypoint1LocY)
    elif keypoint1LocY > 95-patchRadius:
      print 'Keypoint %s of sample %d is too close to bottom boundary at (%d,%d)' % (keypoint1, sample, keypoint1LocX, keypoint1LocY)
    else:
      # if keypoint1 is not too close to boundary, record patch containing keypoint1
      imageDict = {'Patch': data['Image'][sample][keypoint1LocY - patchRadius:keypoint1LocY + patchRadius, keypoint1LocX - patchRadius:keypoint1LocX + patchRadius]}
      patchList.append(imageDict)
      keypointYes.append(1)
      keypointNo.append(0)
      
    # if keypoint2 location is not present, report
    if np.isnan(keypoint2LocX) or np.isnan(keypoint2LocY):
      print 'Sample %s does not have %s data' % (sample, keypoint2) 
    # report if keypoint2 location is too close to boundary
    elif keypoint2LocX < patchRadius:
      print 'Keypoint %s of sample %d is too close to left boundary at (%d,%d)' % (keypoint2, sample, keypoint2LocX, keypoint2LocY)
    elif keypoint2LocX > 95-patchRadius:
      print 'Keypoint %s of sample %d is too close to right boundary at (%d,%d)' % (keypoint2, sample, keypoint2LocX, keypoint2LocY)
    elif keypoint2LocY < patchRadius:
      print 'Keypoint %s of sample %d is too close to top boundary at (%d,%d)' % (keypoint2, sample, keypoint2LocX, keypoint2LocY)
    elif keypoint2LocY > 95-patchRadius:
      print 'Keypoint %s of sample %d is too close to bottom boundary at (%d,%d)' % (keypoint2, sample, keypoint2LocX, keypoint2LocY)
    else:
      # if keypoint2 is not too close to boundary, record patch containing keypoint2
      imageDict = {'Patch': data['Image'][sample][keypoint2LocY - patchRadius:keypoint2LocY + patchRadius, keypoint2LocX - patchRadius:keypoint2LocX + patchRadius]}
      patchList.append(imageDict)
      keypointYes.append(0)
      keypointNo.append(1)

  # combine lists into a single DataFrame
  trainingSet = DataFrame(patchList)
  trainingSet[1] = keypointYes
  trainingSet[0] = keypointNo
  
  # calculate integral images
  calcIntImage(trainingSet)
  
  return trainingSet

    
def clusterPred(image, cascade, thresholds, plot = False, maxSpace = 8, minClusterFrac = 10, keypoint = 'left_eye_center'):
  """
  Predicts location of keypoint using center of top-right cluster (that contains at least 1/10 of positives from cascade)
  Inputs: image: pixel values in numpy array
          cascade: list of strong classifiers
          thresholds: list of threshold values
          distance: max spacing of clustering (default = 5)
  Output: Graphs predicted location of left-eye keypoint  
  """
  # obtain subwindows from image
  sampleSet = cy.getSubwindows(image)
  calcIntImage(sampleSet)

  # convert single strong classifier to a cascade, if necessary
  if not isinstance(cascade, list):
    cascade = [cascade]
    if thresholds is not None:
      thresholds = [thresholds]
    
  # obtain predictions from cascade
  pred = cascadePred(sampleSet, cascade, thresholds)
  
  # obtain index values of locations that have tested positive
  predIndex = sampleSet[pred == True].index
  
  # edgeList consist of tuples of the form (Euclidean distance between point i and point j, i, j)
  prevIndex = []
  edgeList = []
  
  for i in predIndex:
    prevIndex.append(i)
    for j in predIndex.diff(prevIndex):
      if j <= i + len(image) * maxSpace:
        distance = np.linalg.norm(sampleSet.ix[i,['x','y']] - sampleSet.ix[j,['x','y']])
        if distance <= maxSpace:
          edgeList.append((distance, i, j))

  # create union find class that contains all positive locations
  uf = UF.UnionFind(predIndex)
  
  # union points together until edge length exceeds maxSpace
  for edge in edgeList:
    uf.union(edge[1],edge[2])

  # obtain list of clusters that contain at least 1/minClusterFrac of the positive locations
  parentTable = Series(uf.parent)
  parentVals = parentTable.value_counts()
  parents = parentVals[parentVals > (len(predIndex)/minClusterFrac)].index
  
  # obtain the center of each cluster and measure its distance to the top-left corner
  centers = DataFrame(index = parents, columns = ['x','y'])
  
  for parent in parents: 
    centers.ix[parent] = sampleSet.ix[parentTable[parentTable == parent].index, ['x','y']].mean()
    
  centers['topLeft'] = (96 - centers['x'])**2 + centers['y']**2
  centers = centers.astype(float)
  
  # predict location of left eye is the center of teh top-leftmost cluster, plot in red
  leftEyePred = centers['topLeft'].idxmin()
  
  if plot:
    # plot image and positive locations in blue
    plt.imshow(image, cmap=cm.Greys_r)
    
    for i in sampleSet[pred == True].index:
      plt.scatter(sampleSet['x'][i], sampleSet['y'][i], color = 'blue')
    
    plt.scatter(centers['x'][leftEyePred], centers['y'][leftEyePred], color = 'red') 
  
  return Series({keypoint + '_x': centers['x'][leftEyePred], keypoint + '_y': centers['y'][leftEyePred]})
  
  
def loadCascade(numStrong):
  """
  Loads cascade and thresholds from storage.h5
  """
  store = pd.HDFStore('storage.h5')
  thresholds = store['thresholds']
  
  cascade = []
  
  for i in xrange(numStrong):
    cascade.append(store['strong' + str(i)])
  
  return thresholds, cascade
  
  
  
def evalPred(train, cascade, thresholds, keypoint = 'left_eye_center'):
  """
  Predicts the location of keypoint based on clusterPred() and computes RMSE
  Inputs: train: training data DataFrame
          cascade: list of strong classifiers
          thresholds: list of threshold values
  """

  pred = train['Image'].apply(lambda x: clusterPred(x, cascade, thresholds))

  actual = train[[keypoint + '_x', keypoint + '_y']]
  
  return math.sqrt(((actual - pred)**2).sum().sum() / (2 * len(actual)))

  
def createStrong(patchSet, maxFalseRate, minDetectRate = .999):

  strong = adaBoost(patchSet, 1)
  
  threshold = findThreshold(patchSet, strong, minDetectRate)
  report = strongReport(patchSet, strong, threshold)

  if report['false_positive_rate'] < maxFalseRate:
    return strong, threshold
    
  else:
    flag = True

    while flag:
      strong = adaBoost(patchSet, 1, strong)
      threshold = findThreshold(patchSet, strong, minDetectRate)
      report = strongReport(patchSet, strong, threshold)
      if report['false_positive_rate'] < maxFalseRate:
        return strong, threshold
        


featureTypes = {12: cy.feature12, 21: cy.feature21, 13: cy.feature13, 31: cy.feature31, 22: cy.feature22}
nameDict = {0: '12', 1: '21', 2: '13', 3: '31', 4: '22'}
patchRadius = 12