import random
import pandas as pd
from pandas import DataFrame, Series
import numpy as np
import face
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math


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

def demoWeakClassifier(patchSet, type, x1, y1, x2, y2):
  """
  Trains and returns specified weak classifier 
  Input: patchSet = DataFrame with training patches and columns IntImage, 0, 1
         type = 12, 21, 13, 31, or 22 (type of rectangular feature)
         x1, y1 = upper left coordinates
         x2, y2 = bottom right coordinates
  Output: Weak classifier for feature (type, x1, y1, x2, y2) trained on patchSet
  Running time: 65 ms
  """
  # calculate values of desired feature and save in DataFrame values
  values = calcFeature(patchSet, type, x1, y1, x2, y2)
  
  # initialize weightedVals
  weightedVals = DataFrame(index = patchSet.index)
  weightedVals[1] = patchSet[1] * 1/float(2 * patchSet[1].sum())
  weightedVals[0] = patchSet[0] * 1/float(2 * patchSet[0].sum())
  
  # create weak classifier for desired feature
  return weakClassifier(weightedVals, values)

def demoStrongClassifier(patchSet, featureList, threshold = None):
  """
  Trains and returns specified strong classifier
  Input: patchSet = DataFrame with columns IntImage, 0, 1
         featureList = list of features in tuple form (order listed matters - weak classifiers trained on samples weighted by previous weak classifiers!)
         threshold = how lenient the final strong classifier will be (lower value => higher detection rates but also higher false positive rates)
                     if False, sets to half the sum of the alpha values of weak classifiers
  Output: strong classifier made up of weak classifiers trained on patchSet
          also prints performance on patchSet
  Running time (20 feature classifier): 2.46 s
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
    
    # train weak classifier
    weak = weakClassifier(weightedVals, values)
    
    # determine predictions of weak classifier
    classCorrect = values * weak['parity'] > weak['threshold'] * weak['parity']
    
    # determine which samples were predicted correctly
    correct = classCorrect*patchSet[1] + (-classCorrect) * patchSet[0]
          
    beta = weak['error'] / (1 - weak['error'])
    alpha = np.log(1/beta)
    # update weights
    weights = updateWeights(weights, correct, beta)
    
    # add best weak classifier to strong classifier
    weak = weak.append(Series({'feature': feature, 'alpha': alpha}))
    strong = strong.append(weak, ignore_index = True)
    
  # evaluate strong classifier
  print strongReport(patchSet, strong, threshold)
  
  return strong
  
  
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
  sampleSet = getSubwindows(image)
  
  # convert strong to cascade, if necessary
  if not isinstance(strong, list):
    strong = [strong]
    if threshold is not None:
      threshold = [threshold]
    
  # obtain predictions
  pred, predVals = cascadePred(sampleSet, strong, threshold)
    
  maxId = predVals.idxmax()
  numDetections = pred.sum()
  numSamples = len(sampleSet)
  
  print 'Number of detections: %d / %d' % (numDetections, numSamples)
  
  # print image and positive detections
  plt.imshow(image, cmap=cm.Greys_r)
  
  for i in sampleSet[pred == True].index:
    plt.scatter(sampleSet['x'][i], sampleSet['y'][i], color = 'blue')
  
  plt.scatter(sampleSet['x'][maxId],sampleSet['y'][maxId], color = 'red')

  
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

def trainingPatches(data, strong = False, threshold = False, keypoint = 'left_eye_center', numWithout = 1):
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
        # if strong classifier, select patch that is classified positive by the strong classifier
        else:
          flag = True
          
          while flag:
            patchCenterX = random.randint(patchRadius,95-patchRadius)
            patchCenterY = random.randint(patchRadius,95-patchRadius)
            patch = data['Image'][sample][patchCenterY - patchRadius:patchCenterY + patchRadius, patchCenterX - patchRadius:patchCenterX + patchRadius]
            if predPatch(patch, strong, threshold):
              flag = False
        imageDict = {'Patch': data['Image'][sample][patchCenterY - patchRadius:patchCenterY + patchRadius, patchCenterX - patchRadius:patchCenterX + patchRadius]}
        patchList.append(imageDict)
        keypointYes.append(0)
        keypointNo.append(1)    
      continue
    
    # if feature location is present, record patches that do not contain feature
    for i in range(numWithout):
      # if no strong classifier, select patch so that it does not contain keypoint
      if not strong:
        contained = True
        
        # ensure that patch does not contain feature
        while contained:
          patchCenterX = random.randint(patchRadius,95-patchRadius)
          patchCenterY = random.randint(patchRadius,95-patchRadius)
          if (patchCenterX < int(keypointLocX - patchRadius) or patchCenterX > int(keypointLocX + patchRadius)) and (patchCenterY < int(keypointLocY - patchRadius) or patchCenterY > int(keypointLocY + patchRadius)):
            contained = False
      # if strong classifier, select patch so that it does not contain keypoint and is classified positive by strong classifier
      else:
        flag = True
        while flag:
          patchCenterX = random.randint(patchRadius,95-patchRadius)
          patchCenterY = random.randint(patchRadius,95-patchRadius)
          if (patchCenterX < int(keypointLocX - patchRadius) or patchCenterX > int(keypointLocX + patchRadius)) and (patchCenterY < int(keypointLocY - patchRadius) or patchCenterY > int(keypointLocY + patchRadius)):
            patch = data['Image'][sample][patchCenterY - patchRadius:patchCenterY + patchRadius, patchCenterX - patchRadius:patchCenterX + patchRadius]
            if predPatch(patch, strong, threshold):
              flag = False
      
      imageDict = {'Patch': data['Image'][sample][patchCenterY - patchRadius:patchCenterY + patchRadius, patchCenterX - patchRadius:patchCenterX + patchRadius]}
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
  Running time: 80 ms
  """
 
  N=len(patch)
  int_im = np.zeros((N+1,N+1))
  
  int_im[1:,1:] = np.cumsum(np.cumsum(patch, axis=0), axis=1)
  
  if normalize:
    var_sqrt = math.sqrt(math.fabs(math.pow(int_im[N][N]/patch.size,2) - ((patch*patch).sum() / (patch.size))))
    return int_im / var_sqrt
  else:
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
  
  return patchSet['IntImage'].apply(lambda x: featureTypes[type](x1, y1, x2, y2, x))
  
def featureValues(patchSet, type):
  """
  Input: patchSet = DataFrame of patches as numpy arrays
         type = type of feature (12, 21, 13, 31, or 22)
  Returns 'all' feature values of given type for patches in patchSet
  """
  patchSize = len(patchSet['IntImage'][0])
  
  values = DataFrame(index = patchSet.index)
  
  spacingX = (type % 10) * 2
  spacingY = int(type / 10) * 2
  
  for a in xrange(0,patchSize-1,2):
    for b in xrange(0,patchSize-1,2):
      for x in xrange(a+spacingX,patchSize,spacingX):
        for y in xrange(b+spacingY,patchSize, spacingY):
          values[(type,a,b,x,y)] = calcFeature(patchSet,type,a,b,x,y)
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
        classCorrect = featureVals[feature] * weak['parity'] > weak['threshold'] * weak['parity']
        
        # determine which samples were correctly classified by weak classifier
        correct = classCorrect*patchSet[1] + (-classCorrect) * patchSet[0]
       
    beta = error / (1 - error)
    alpha = np.log(1/beta)
    # update weights (reduces weight for correctly classified samples and normalizes)
    weights = updateWeights(weights, correct, beta)
    
    # add best weak classifier to strong classifier
    print 'Added feature:'
    print weak
    
    weak = weak.append(Series({'feature': feature, 'alpha': alpha}))
    strong = strong.append(weak, ignore_index = True)
    
  store.close()
    
  return strong
  
def getSubwindows(image):
  """
  Input: pixel values of image as numpy array
  Output: DataFrame containing all patches of size patchRadius (global var), center of patch, and their integral images
  """
  patchList = []
  centerListX = []
  centerListY = []
  
  imageLength = len(image)
  
  # add patches to list
  for i in xrange(patchRadius, imageLength - patchRadius):
    for j in xrange(patchRadius, imageLength - patchRadius):
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
  
  
def findThreshold(patchSet, strong, minDetectionRate):
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
    report = strongReport(patchSet, strong, threshold)
    
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
    values = calcFeature(patchSet, type, x1, y1, x2, y2)
    
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
  Running time: 1.1 s
  """
  
  predVals = Series(np.zeros(len(patchSet)), index = patchSet.index)
  
  # for each weak classifier in strong classifier, add weighted value to those classified as containing keypoint
  for i in strong.index:
    type, x1, y1, x2, y2 = strong['feature'][i]
    # calculate feature for weak classifier
    featureVals = calcFeature(patchSet, type, x1, y1, x2, y2)
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
  Output: Series of bools containing predictions, id value of highest scoring sample
  """
  
  # if threshold is None, set threshold to default (1/2 sum of alpha values)
  if threshold is None:
    threshold = strong['alpha'].sum() / 2
    print 'Testing with default threshold %f' % threshold
  
  # obtain scores from strong classifier
  predVals = runStrong(patchSet, strong)
  
  # return bool of predictions and value of highest scoring prediction
  return predVals > threshold, predVals


  
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
  
  # records the score given by strong classifier
  predVals = Series(np.zeros(len(patchSet)), index = patchSet.index)
  
  # each strong classifier makes predictions, update scores, repeat only on positive predictions
  for i in xrange(len(cascade)):
    classifiedYes, vals = strongPred(remainingPatches, cascade[i], thresholds[i])
    predVals = predVals.add(vals, fill_value = 0)
    remainingPatches = remainingPatches.ix[classifiedYes]
  
  # positive predictions are the remaining patches
  pred = patchSet.index.isin(remainingPatches.index)
    
  return pred, predVals

  
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
    classifiedYes, vals = strongPred(remainingPatches, cascade[i], thresholds[i])
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
    # determine feature values corresponding to weak classifiers
    values = cascade[i]['feature'].apply(lambda x: featureTypes[x[0]](x[1], x[2], x[3], x[4], ii))
    # make predictions for each weak classifier
    predictions = values * cascade[i]['parity'] > cascade[i]['threshold'] * cascade[i]['parity']
    # add up relevant alpha values
    score = (cascade[i]['alpha'] * predictions).sum()
    # return false if below threshold, else continue to next strong classifier
    if score <= thresholds[i]:
      return False
  
  return True
  

featureTypes = {12: feature12, 21: feature21, 13: feature13, 31: feature31, 22: feature22}
nameDict = {0: '12', 1: '21', 2: '13', 3: '31', 4: '22'}
patchRadius = 12