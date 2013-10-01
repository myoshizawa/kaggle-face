'''
Created on Sep 25, 2013

@author: mark
'''

import math

def featuresMatrix(line):
    #the line is comma delimited with the following data
    #the first 30 entries are (x,y) pairs of specific points
    #   some points may be missing, in which case there will be nothing between the commas
    #the final entry will be a space delimited list of 96x96 gray-scale numbers
    
    features = line.split(",")
    if (len(features) is not 31):
        print "The input is not correctly formatted"
        return
    
    coordinates = [entry for entry in features[:30] if entry is not '']
    points = []
    for i in xrange(0,len(coordinates),2):
        points.append((coordinates[i],coordinates[i+1]))
        
    grayList =features[30].split(" ") 
    grayMatrix = [[] for _ in xrange(96)]
    for i in xrange(96):
        grayMatrix[i] = grayList[i*96:(i+1)*96]
    
    bold = '\033[1m'
    normal = '\033[0m'
    for p in points:
        xUpper = int(math.ceil(float(p[0])))
        xLower = int(math.floor(float(p[0])))
        yUpper = int(math.ceil(float(p[1])))
        yLower = int(math.floor(float(p[1])))
        grayMatrix[yUpper][xUpper] = bold + grayMatrix[yUpper][xUpper] + normal
        grayMatrix[yUpper][xLower] = bold + grayMatrix[yUpper][xLower] + normal
        grayMatrix[yLower][xUpper] = bold + grayMatrix[yLower][xUpper] + normal
        grayMatrix[yLower][xLower] = bold + grayMatrix[yLower][xLower] + normal  
    
    for row in grayMatrix:
        print " ".join(row)
    
    return