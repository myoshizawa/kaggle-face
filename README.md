kaggle-face
===========

Facial Keypoints Detection Kaggle competition

face.py

readTrain(numRows) - reads in numRows of training.csv and returns as a pandas DataFrame

readTest() - reads in test.csv and returns as a pandas DataFrame

plotImage(dataframe, num) - uses matplotlib to plot image of entry num with facial keypoints

avgPatch(dataframe, feature, size) - averages the square of pixels around the desired feature over all samples in the given data frame

bestMatch(dataframe, image, patch, feature, searchSize) - compares patches of the given image to the given patch and determines the location of the best match
