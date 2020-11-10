from __future__ import print_function
import cv2 as cv
import sys
from pylab import *
from matplotlib import *
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from pylab import figure, text, scatter, show
import matplotlib.image as mpimg
import argparse
import scipy.stats as stats
import math
import matplotlib.mlab as mlab
from scipy.stats import norm

bad_1 = "/home/adamanov/PycharmProjects/imgProcessing/006_1_2_75_left.png"
bad_2 = "/home/adamanov/PycharmProjects/imgProcessing/009_1_4_75_right.png"
bad_3 = "/home/adamanov/PycharmProjects/imgProcessing/005_3_2_75_right.png"
bad_4 = "/home/adamanov/PycharmProjects/imgProcessing/015_2_6_125_left.png"
img_bad_1 = cv.imread(bad_1)
img_bad_2 = cv.imread(bad_2)
img_bad_3 = cv.imread(bad_3)
img_bad_4 = cv.imread(bad_4)

images_bad = [img_bad_1,img_bad_2,img_bad_3,img_bad_4]

means_b   = np.zeros((size(images_bad),1),dtype= float)
stddevs_b = np.zeros((size(images_bad),1),dtype= float)


good_1 = "/home/adamanov/PycharmProjects/imgProcessing/639_2_6_175_left.png"
good_2 = '/home/adamanov/PycharmProjects/imgProcessing/604_1_6_175_right.png'
good_3 = '/home/adamanov/PycharmProjects/imgProcessing/602_2_5_125_left.png'
good_4 = '/home/adamanov/PycharmProjects/imgProcessing/603_4_6_175_right.png'
img_good_1 = cv.imread(good_1)
img_good_2 = cv.imread(good_2)
img_good_3 = cv.imread(good_3)
img_good_4 = cv.imread(good_4)
images_good = [img_good_1,img_good_2,img_good_3,img_good_4]

print(shape(img_bad_1[:,:,1]))

means_g   = np.zeros((size(images_good),1),dtype= float)
stddevs_g = np.zeros((size(images_good),1),dtype= float)

channels = [0]
mask = None
bins = [256]
ranges = [0, 256]


for i in arange(0,size(images_good)):
    # For good images
    imgGray = cv.cvtColor(images_good[i], cv.COLOR_BGR2GRAY)
    hist = cv.calcHist(imgGray, channels, mask, bins, ranges)
    #cv.normalize(hist, hist, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
    #plt.plot(hist,color = 'b',label = 'Good Images' )
    mean = np.mean(imgGray)
    stddev = np.std(imgGray)
    means_g[i]= mean
    stddevs_g[i] = stddev

    # For bad images
    imgGray = cv.cvtColor(images_bad[i], cv.COLOR_BGR2GRAY)
    hist = cv.calcHist(imgGray, channels, mask, bins, ranges)
    mean = np.mean(imgGray)
    stddev = np.std(imgGray)
    means_b[i]= mean
    stddevs_b[i] = stddev
    #plt.plot(hist,color = 'r' ,label = 'Bad images')
    #plt.legend(loc="upper left")
    #plt.show(block=False)
    #plt.pause(500)
    
avMean_g = np.average(means_g)
avStd_g  = np.average(stddevs_g)
print("good images: ",str(avMean_g)[0:5], str(avStd_g)[0:5])

avMean_b = np.average(means_b)
avStd_b = np.average(stddevs_b)
print("bad images:  " , str(avMean_b)[0:5], str(avStd_b)[0:5])

#x = np.linspace(-50,50, 100)
x = np.arange(-50,300)
plt.plot(x, stats.norm.pdf(x, avMean_g, avStd_g), color='b',label = 'Good Images')
plt.plot(x, stats.norm.pdf(x, avMean_b, avStd_b), color='r',label = 'Bad images')
plt.legend(loc="upper right")

plt.xlabel('Smarts')
plt.ylabel('Probability')
#plt.title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=%.3f,\ \sigma=%.3f$' %(mu, sigma))
plt.grid(True)

plt.show()