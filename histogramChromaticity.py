# -*- coding: utf-8 -*-
"""
Created on Tue May  8 16:47:12 2018

@author: Fulvio Bertolini
"""


def bgr_to_chR_chG(bgrImg):
    rows, cols, _ = bgrImg.shape
    
    chR = np.zeros((rows, cols), np.float)
    chG = np.zeros((rows, cols), np.float)
    chB = np.zeros((rows, cols), np.float)
    
    for x in range(0,rows):
        for y in range(0,cols):
            if  (bgrImg[x,y,1] != 0):
               
                r = float(bgrImg[x,y,2]) / float(255)
                g = float(bgrImg[x,y,1]) / float(255)
                b = float(bgrImg[x,y,0]) / float(255)
                
                # r chromaticity componen
                chR[x,y] = np.uint8(r*255 / (r+g+b))
                
                #g chromaticity component
                chG[x,y] = np.uint8(g*255 / (r+g+b))
                
                chB[x,y] = np.uint8(b*255 / (r+g+b))
                
    return chR, chB, chG


import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('RGB_29.png')
#hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
chR, chB, chG = bgr_to_chR_chG(img)


chromaticity = np.zeros((chR.shape[0], chR.shape[1], 3), np.uint8)
chromaticity[:,:,0] = chR
chromaticity[:,:,1] = chG
chromaticity[:,:,2] = chB

plt.figure()
plt.subplot(111) 
plt.imshow(chromaticity)
plt.title("RGB")
rows, cols = chG.shape



plt.figure()
plt.subplot(111) 
plt.imshow( chG, cmap='gray')
plt.title("chG")


plt.figure()
plt.subplot(111) 
plt.imshow( chR, cmap='gray')
plt.title("chR")
 

plt.figure()
plt.subplot(111) 
plt.imshow( chB, cmap='gray')
plt.title("chB")

plt.figure()
plt.hist(chR.ravel(),256,[0,256])
plt.title("rosso")
plt.figure()
plt.hist(chG.ravel(),256,[0,256])
plt.title("verde")
plt.figure()
plt.hist(chB.ravel(),256,[0,256])
plt.title("blu")
#h = hsv[:,:,0]
#l = hsv[:,:,1]
#s = hsv[:,:,2]
#
#
#R = img[:,:,2]
#g = img[:,:,1]
#b = img[:,:,0]
# 
#
#
#plt.figure()
#plt.subplot(221) 
#plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#plt.title("RGB")
#plt.subplot(222) 
#plt.imshow(R, cmap='gray')
#plt.title("r")
#plt.subplot(223) 
#plt.imshow(g, cmap='gray')
#plt.title("g")
#plt.subplot(224) 
#plt.imshow(b, cmap='gray')
#plt.title("b")
#

# 
#plt.figure()
#plt.subplot(221) 
#plt.imshow(hsv)
#plt.title("HSV")
#plt.subplot(222) 
#plt.imshow(h, cmap='gray')
#plt.title("h")
#plt.subplot(223) 
#plt.imshow(l, cmap='gray')
#plt.title("l")
#plt.subplot(224) 
#plt.imshow(s, cmap='gray')
#plt.title("s")
#
#
#
#
##hist = cv2.calcHist(h.ravel(), 0, None, [180, 256], [0, 180, 0, 256])
#
r = cv2.selectROI(chromaticity)
#hCrop = h[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
#lCrop = l[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
#sCrop = s[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
#
#
#rCrop = R[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
#gCrop = g[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
#bCrop = b[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]

chRCrop = chR[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
chBCrop = chB[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
chGCrop = chG[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]

#plt.figure()
#plt.subplot(221) 
#plt.hist(rCrop.ravel(),256,[0,256])
#plt.hist(gCrop.ravel(),256,[0,256])
#plt.hist(bCrop.ravel(),256,[0,256])
#plt.title("HSV")
#plt.subplot(222) 
#plt.hist(rCrop.ravel(),256,[0,256])
#plt.title("h")
#plt.subplot(223) 
#plt.hist(gCrop.ravel(),256,[0,256])
#plt.title("s")
#plt.subplot(224) 
#plt.hist(bCrop.ravel(),256,[0,256])
#plt.title("v")



plt.figure()
plt.subplot(131) 
plt.hist(chRCrop.ravel(),256,[0,256])
plt.title("chR")
plt.subplot(132) 
plt.hist(chBCrop.ravel(),256,[0,256])
plt.title("chB")
plt.subplot(133) 
plt.hist(chGCrop.ravel(),256,[0,256])
plt.title("chG")




#
#plt.figure()
#plt.subplot(221) 
#plt.hist(hCrop.ravel(),256,[0,256])
#plt.hist(lCrop.ravel(),256,[0,256])
#plt.hist(sCrop.ravel(),256,[0,256])
#plt.title("HLS")
#plt.subplot(222) 
#plt.hist(hCrop.ravel(),256,[0,256])
#plt.title("h")
#plt.subplot(223) 
#plt.hist(lCrop.ravel(),256,[0,256])
#plt.title("l")
#plt.subplot(224) 
#plt.hist(sCrop.ravel(),256,[0,256])
#plt.title("s")
#
#
#plt.show()
#
#plt.show()
#
#plt.show()
#
#
#     