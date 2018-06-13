# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 14:32:03 2018

@author: Fulvio Bertolini
"""
import gmmTools as tools
import os
import numpy as np
import cv2
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import scipy.signal as signal
import matplotlib.pyplot as plt


sessionID = "bankpark"
frameNbr = 135
if(sessionID == ""):
    sessionID = input("Enter session ID: ")

sessionPath = "../Sessions/" + sessionID + "/"
if os.path.exists(sessionPath):
    if(frameNbr == 0):
        frameNbr = input("Enter number frame: ")
    folder = sessionPath + "singleFrames/" + str(frameNbr)
    
    
    colorRGB, hsv, luv, ycrcb = tools.loadColorImages(folder)
    normalsRGB =  tools.loadNormalsImages(folder)
    depth = tools.loadDepthImages(folder)





    rawFeatures = tools.concatenateRawFeatures(tools.makeArray3(colorRGB),
                                               tools.makeArray3(normalsRGB), 
                                               tools.makeArray(depth),
                                               tools.makeArray3(hsv),
                                               tools.makeArray3(luv),
                                               tools.makeArray3(ycrcb))
    
    rawFeatures_filtered, saturatedIndices = tools.removeSaturated(rawFeatures)
    processedFeatures = tools.processFeatures(rawFeatures_filtered)
    
    pca = PCA(n_components=10)
    pcaFeatures = pca.fit_transform(processedFeatures)
    gmm = GaussianMixture(3).fit(pcaFeatures)

    prob = gmm.predict_proba(pcaFeatures)
    
    indexClasses = np.zeros(prob.shape[0])
    probClasses = np.zeros(prob.shape[0])
    for i in range(prob.shape[0]):
        probClasses[i] = np.amax(prob[i,:])
        if(probClasses[i] < 0.9):
            indexClasses[i] = -1
        else:
            indexClasses[i] = np.argmax(prob[i,:])
        
    plt.figure()
    plt.hist(probClasses,200,[0,1])
    plt.title("probabilities")
    
    
    
    wholeImageClasses = np.zeros((colorRGB.shape[0], colorRGB.shape[1]), np.float)
    
    wholeImageClasses = wholeImageClasses.ravel()
    nonSaturatedIndex = 0
    saturatedIndex = 0
    for i in range(wholeImageClasses.shape[0]):
        if(i == saturatedIndices[saturatedIndex]):
            wholeImageClasses[i] = -1
            if(len(saturatedIndices)-1 > saturatedIndex):
                saturatedIndex += 1
        else:
            wholeImageClasses[i] = indexClasses[nonSaturatedIndex]
            nonSaturatedIndex += 1

    wholeImageClasses = np.reshape(wholeImageClasses, (colorRGB.shape[0], colorRGB.shape[1]))
    
    plt.figure
    plt.imshow(wholeImageClasses)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    