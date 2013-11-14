import numpy as np
cimport numpy as np

import pandas as pd
from pandas import DataFrame, Series

DTYPE = np.int
ctypedef np.int_t DTYPE_t

def feature21(int x1, int y1, int x2, int y2, np.ndarray[DTYPE_t, ndim=2] ii):
  """
  2-features parted vertically
  """
  
  cdef int y12 = (y1 + y2) / 2
  cdef int f1 = ii[y1,x1]
  cdef int f2 = ii[y12,x1]
  cdef int f3 = ii[y2,x1]
  cdef int f4 = ii[y1,x2]
  cdef int f5 = ii[y12,x2]
  cdef int f6 = ii[y2,x2]
  
  return f6 - f5 - f5 + f4 - f3 + f2 + f2 - f1
  
def feature12(int x1, int y1, int x2, int y2, np.ndarray[DTYPE_t, ndim=2] ii):
  """
  2-features parted horizontally
  """
  
  cdef int x12 = (x1 + x2) / 2
  cdef int f1 = ii[y1,x1]
  cdef int f2 = ii[y1,x12]
  cdef int f3 = ii[y1,x2]
  cdef int f4 = ii[y2,x1]
  cdef int f5 = ii[y2,x12]
  cdef int f6 = ii[y2,x2]
  
  return f6 - f5 - f5 + f4 - f3 + f2 + f2 - f1
  
def feature31(int x1, int y1, int x2, int y2, np.ndarray[DTYPE_t, ndim=2] ii):
  """
  3-features sliced vertically
  """
  
  cdef int third = (y2 - y1)/3
  cdef int y13 = y1 + third
  cdef int y23 = y13 + third
  cdef int f1 = ii[y1,x1]
  cdef int f2 = ii[y13,x1]
  cdef int f3 = ii[y23,x1]
  cdef int f4 = ii[y2,x1]
  cdef int f5 = ii[y1,x2]
  cdef int f6 = ii[y13,x2]
  cdef int f7 = ii[y23,x2]
  cdef int f8 = ii[y2,x2]
  
  return f8 - f7 - f7 + f6 + f6 - f5 - f4 + f3 + f3 - f2 - f2 + f1
  
def feature13(int x1, int y1, int x2, int y2, np.ndarray[DTYPE_t, ndim=2] ii):
  """
  3-features sliced horizontally
  """
  
  cdef int third = (x2 - x1)/3
  cdef int x13 = x1 + third
  cdef int x23 = x13 + third
  cdef int f1 = ii[y1,x1]
  cdef int f2 = ii[y1,x13]
  cdef int f3 = ii[y1,x23]
  cdef int f4 = ii[y1,x2]
  cdef int f5 = ii[y2,x1]
  cdef int f6 = ii[y2,x13]
  cdef int f7 = ii[y2,x23]
  cdef int f8 = ii[y2,x2]
  
  return f8 - f7 - f7 + f6 + f6 - f5 - f4 + f3 + f3 - f2 - f2 + f1
  
def feature22(int x1, int y1, int x2, int y2, np.ndarray[DTYPE_t, ndim=2] ii):
  """
  4-features
  """
  
  cdef int x12 = (x1 + x2)/2
  cdef int y12 = (y1 + y2)/2
  cdef int f1 = ii[y1,x2]
  cdef int f2 = ii[y1,x12]
  cdef int f3 = ii[y1,x2]
  cdef int f4 = ii[y12,x1]
  cdef int f5 = ii[y12,x12]
  cdef int f6 = ii[y12,x2]
  cdef int f7 = ii[y2,x1]
  cdef int f8 = ii[y2,x12]
  cdef int f9 = ii[y2,x2]
  
  return f9 - f8 - f8 + f7 - f6 - f6 + f5 + f5 + f5 + f5 - f4 - f4 + f3 - f2 - f2 + f1
  
cimport cython
@cython.boundscheck(False)
@cython.wraparound(False)  
def calcFeature(np.ndarray[np.int] col_ii, int type, int x1, int y1, int x2, int y2):
  """
  Takes a dataFrame with integral images (under column 'IntImage') and adds a column '(type,x1,y1,x2,y2)' with the feature values
  """
  cdef Py_ssize_t i, n = len(col_ii)
  
  cdef np.ndarray[double] featureVals = np.empty(n)
  
  for i in range(n):
    featureVals[i] = featureTypes[type](x1,y1,x2,y2,col_ii[i])
  
  return featureVals.astype(np.int)
  
  
def featureValues(np.ndarray[np.int] col_ii, int type):
  """
  Input: patchSet = DataFrame of patches as numpy arrays
         type = type of feature (12, 21, 13, 31, or 22)
  Returns 'all' feature values of given type for patches in patchSet
  """
  cdef Py_ssize_t patchSize = len(col_ii[0]), n = len(col_ii)
  
  cdef int spacingX = (type % 10) * 2
  cdef int spacingY = (type / 10) * 2
  cdef Py_ssize_t a,b,x,y,i
  
  valFunc = featureTypes[type]
  
  cdef np.ndarray[double] featureVals = np.empty(n)
  
  values = {}
  
  for a in range(0,patchSize-1,2):
    for b in range(0,patchSize-1,2):
      for x in range(a+spacingX,patchSize,spacingX):
        for y in range(b+spacingY,patchSize, spacingY):
          for i in range(n):
            featureVals[i] = valFunc(a,b,x,y,col_ii[i])
        
          values[(type,a,b,x,y)] = featureVals.astype(np.int)
          
  return values

  
def getSubwindows(np.ndarray[DTYPE_t, ndim=2] image, int spacing = 1):
  """
  Input: pixel values of image as numpy array
  Output: DataFrame containing all patches of size patchRadius (global var), center of patch, and their integral images
  """
  cdef Py_ssize_t imageLength = len(image)
  
  patchList = []
  centerX = []
  centerY = []
  
  cdef Py_ssize_t i,j
  
  # add patches to list
  for i in range(patchRadius, imageLength - patchRadius, spacing):
    for j in range(patchRadius, imageLength - patchRadius, spacing):
      patchList.append(image[j-patchRadius:j+patchRadius,i-patchRadius:i+patchRadius])
      centerX.append(i)
      centerY.append(j)

  # convert output to a Frame
  sampleSet = DataFrame({'Patch': patchList, 'x': centerX, 'y': centerY})
  
  return sampleSet

  
featureTypes = {12: feature12, 21: feature21, 13: feature13, 31: feature31, 22: feature22}
cdef int patchRadius = 12