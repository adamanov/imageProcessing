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



####################################################################################################################
#    Applied Methods and Hyper-parameters to play:                                                                #
#                  Global Histogram Equalizer             ->  No Parameters                                       #
#                  Adaptive (local) Histogram Equalizer   ->  clipLimit and GridSize                              #
#                  Law Power Transformation               ->  gammaThreshold  (can be estimated through search)** #
#                  basicLinearTransformation              ->  contrastGain and brightnessBias                     #
####################################################################################################################


# ______________________Calculate histogram for current image______________________#
def hist_plot(img, adjImag, img_name, normalize=False, gammaPlot=False, grayHist=False):
    # Initial Parameters
    channels = [0]
    mask = None
    bins = [256]
    ranges = [0, 256]

    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    hist = cv.calcHist(imgGray, channels, mask, bins, ranges)
    if normalize:
        cv.normalize(hist, hist, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)

    if grayHist:
        ax21.set_title("Gray Channel")
        ax21.plot(hist, color='k')

    if not gammaPlot:
        fig, axes = plt.subplots(nrows=2, ncols=2)
        bx11, bx12, bx21, bx22 = axes.flatten()
        # fig.suptitle(img_name)
        fig.canvas.set_window_title("Compare Original with Adjusted")
        figure = matplotlib.pyplot.gcf()
        figure.set_size_inches(6, 7, forward=True)
        mgr = plt.get_current_fig_manager()
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        bx11.imshow(imgRGB)
        bx11.set_title("Original Image")

        adjRGB = cv.cvtColor(adjImag, cv.COLOR_BGR2RGB)
        bx12.imshow(adjRGB)
        bx12.set_title(img_name)

    if grayHist:
        bx21.set_title("Gray Channel")
        bx21.plot(hist, color='k')

    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        histr_org = cv.calcHist([img], [i], mask, bins, ranges)
        histr_adj = cv.calcHist([adjImag], [i], mask, bins, ranges)

        if normalize:
            cv.normalize(histr_org, histr_org, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
            cv.normalize(histr_adj, histr_adj, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)

        if gammaPlot:
            ax21.plot(histr_org, color=color[i])
            text(0.8,0.9, "Mean: " + str(mean(hist))[0:5], ha='center', va='center', transform=ax21.transAxes)
            text(0.8,0.8, "Std: " + str(std(hist))[0:5], ha='center', va='center', transform=ax21.transAxes)

            ax21.set_title("OrgImgHistogram")

            ax22.plot(histr_adj, color=color[i])
            ax22.set_title("AdjImgHistogram")
            text(0.8,0.9, "Mean: " + str(mean(histr_adj))[0:5], ha='center', va='center', transform=ax22.transAxes)
            text(0.8,0.8, "Std: " + str(std(hist))[0:5], ha='center', va='center', transform=ax22.transAxes)


        if not gammaPlot:
            bx21.plot(histr_org, color=color[i])
            bx21.set_title("OrgImgHistogram")

            bx22.plot(histr_adj, color=color[i])
            bx22.set_title("AdjImgHistogram")
    if not gammaPlot:
        plt.show(block=False)
        plt.pause(5)
        plt.close(fig)

    return


# ______________________Hisgoram Equaliztion for colored images____________________#
def GlobalHistEqualized(image):
    channels = cv.split(image)
    eq_channels = []
    for ch, color in zip(channels, ['B', 'G', 'R']):
        eq_channels.append(cv.equalizeHist(ch))

    eq_image = cv.merge(eq_channels)
    eq_image = cv.cvtColor(eq_image, cv.COLOR_BGR2RGB)

    return eq_image


# _____________________Adaptive Histogram Equalization (CLAHE)_____________________#
def AdptHistEqualized(image, clipContrastLimit=4.0, GridSize=5):
    ## Default Parameters
    # clipLimit=4.0,  // Sets threshold for contrast limiting.
    # GridSize=5      // for histogram equalization. Input image will be divided into equally sized rectangular tiles.

    channels = cv.split(image)
    eq_channels = []
    for ch, color in zip(channels, ['B', 'G', 'R']):
        clahe = cv.createCLAHE(clipLimit=clipContrastLimit, tileGridSize=(GridSize, GridSize))
        res = clahe.apply(ch)
        eq_channels.append(res)

    eq_image = cv.merge(eq_channels)
    eq_image = cv.cvtColor(eq_image, cv.COLOR_BGR2RGB)

    return eq_image


# _____________________Adjust Gamma (Power Law Transformation)_____________________#
def adjust_gamma(image, gammaThreshold):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values

    # Gamma correction can be used to correct the brightness of an image
    # by using a non linear transformation between the input values and the mapped output values:

    invGamma = np.divide(1.0, gammaThreshold)
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv.LUT(image,
                  table)  # LUT is used to improve the performance of the computation as only 256 values needs to be calculated once.


def gamma_search(img, gMax,gInterval = 0.5):
    global ax11, ax12, ax21, ax22
    # loop over various values of gamma in a given range
    for gamma in np.arange(0.0, gMax, gInterval):
        # ignore when gamma is 1 (there will be no change to the image)
        if gamma == 1:
            continue
        # apply gamma correction and show the images
        gamma = gamma if gamma > 0 else 0.1
        adjusted = adjust_gamma(img, gamma)

        # cv.putText(adjusted, "g={}".format(gamma), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
        # cv.namedWindow("Images", cv.WINDOW_NORMAL)
        # cv.imshow("Images", np.hstack([img, adjusted]))
        # cv.waitKey(2500)

        fig, axes = plt.subplots(nrows=2, ncols=2)
        ax11, ax12, ax21, ax22 = axes.flatten()
        fig.canvas.set_window_title("Adjusted with Gamma")
        figure = matplotlib.pyplot.gcf()
        figure.set_size_inches(6, 7, forward=True)

        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        ax11.imshow(imgRGB)
        ax11.set_title("Orginal")

        adjustedRGB = cv.cvtColor(adjusted, cv.COLOR_BGR2RGB)
        ax12.imshow(adjustedRGB)
        title_gamma = str(gamma)
        ax12.set_title("Gamma: " + title_gamma[0:3])

        hist_plot(img=img, adjImag=adjusted, img_name=gamma, normalize=True, gammaPlot=True)

        plt.show(block=False)
        plt.pause(5)
        plt.close(fig)


# _________Linear Transformation (correction of Brightness or Contrast)_____________#
def basicLinearTransform(img, contrastGain=2.2, brightnessBias=95):
    # Increasing (/ decreasing) the β value will add (/ subtract) a constant value to every pixel.
    # alpha = gain  (control contrast)    if α<1 , the color levels will be compressed -> less contrast.
    # beta  = bias  (control brightness)

    # Function: g(x) = alpha * f(x) + beta

    alpha = contrastGain
    beta = brightnessBias - 100

    res = cv.convertScaleAbs(img, alpha=alpha, beta=beta)
    img_corrected = cv.hconcat([img, res])
    return res


# ______________________________________________________________________________#
#                      _______Main__________                                    #
# ______________________________________________________________________________#

if __name__ == '__main__':
    location = "006_1_2_75_left.png"
    # location  = 'faceDetection.jpg'

    img = cv.imread(location)
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Global Histogram Equalization
    Eq_image = GlobalHistEqualized(img)
    hist_plot(img, Eq_image, img_name="Eq_imag", normalize=True, gammaPlot=False)

    # Local Histogram Equalization
    AdpEq_img = AdptHistEqualized(img, 5, 8)
    hist_plot(img, AdpEq_img, img_name="AdpEq_img", normalize=True, gammaPlot=False)

    # Linear Transformation
    lt_img = basicLinearTransform(img)
    hist_plot(img, lt_img, img_name="LT Image", normalize=True, gammaPlot=False)

#   for i in np.arange(0.5,5,0.5):
#       print(i)
#       lt_img= basicLinearTransform(img,i)
#       hist_plot(img, lt_img, img_name="contrastGain: " + str(i), normalize=True, gammaPlot=False)

   # Search in range for suitable gamma (Power Law Transform)
    gMax = 3  # Search range Hyper-parameter
    gamma_search(img,gMax = gMax,gInterval=0.5)

    # gamma = 2  # Hyper-parameter
    # gammaImg = adjust_gamma(img, gamma)

