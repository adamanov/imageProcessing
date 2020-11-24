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
import os,sys, getpass
import os.path
from os import path

# pylint: disable=E0602
# pylint: disable=E0611

####################################################################################################################
#    Applied Methods and Hyper-parameters to play:                                                                #
#                  -----Global Histogram Equalizer             ->  No Parameters                                  #
#                  Adaptive (local) Histogram Equalizer   ->  clipLimit and GridSize     -->
#                                                   (push to much blue channel, and subp-ress red channel         #
#                  Law Power Transformation               ->  gammaThreshold  (can be estimated through search)** #
#                  basicLinearTransformation              ->  contrastGain and brightnessBias                     #
####################################################################################################################

def normalizedHistogramHSV(image):
    
    hsv_base = cv.cvtColor(image,cv.COLOR_BGR2HSV)
    # Using th 0-th and 1-st 
    channels = [0,1]
    mask = None
    
    # Using 50 bins for hue and 60 for saturation ( because we need to process both H and S plane.)
    h_bins = [50]
    s_bins = [60]
    hist_Size = [h_bins,s_bins]

    # Hue value lies between 0 and 180 & Saturation lies between 0 and 256.
    h_ranges = [0, 180]
    s_ranges = [0, 256]
    ranges= [h_ranges,s_ranges]

    hist_base= cv.calcHist(hsv_base,[0,1],mask,[50,60],[0,180,0,256])
    cv.normalize(hist_base,hist_base,alpha = 0, beta = 1,norm_type= cv.NORM_MINMAX,mask=None)

    return hist_base

def hsvCompare(img,adjImag):
    # # HSV Histogram Comparasion
    fig, axes = plt.subplots(nrows=1, ncols=3)
    bx11, bx12, bx13= axes.flatten()
    img_good_hist = normalizedHistogramHSV(img)
    img_bad_hist = normalizedHistogramHSV(adjImag)

    bx11.imshow(img_good_hist)
    bx11.set_title("HSV_Hist")
    bx12.imshow(img_bad_hist)
    bx12.set_title("HSV_Hist")

    comp =cv.compareHist(img_good_hist,img_bad_hist,cv.HISTCMP_CORREL)
    text(0.71,0.9, "CORREL [-1 to 1] " + str(comp)[0:5], ha='center', va='center', transform=bx13.transAxes,color ="red")
    comp =cv.compareHist(img_good_hist,img_bad_hist,cv.HISTCMP_CHISQR)
    text(0.71,0.8, "CHISQR [inf to 0] " + str(comp)[0:5], ha='center', va='center', transform=bx13.transAxes,color ="red")
    comp =cv.compareHist(img_good_hist,img_bad_hist,cv.HISTCMP_INTERSECT)
    text(0.68,0.7, "INTERSECT [0 to 1]  " + str(comp)[0:5], ha='center', va='center', transform=bx13.transAxes,color ="red")
    comp =cv.compareHist(img_good_hist,img_bad_hist,cv.HISTCMP_BHATTACHARYYA)
    text(0.61,0.6, "BHATTACHARYYA [1 to 0] " + str(comp)[0:5], ha='center', va='center', transform=bx13.transAxes,color ="red")


# ______________________Calculate histogram for current image______________________#
def hist_plot(img, adjImag, img_name, normalize=True, grayHist=True):
    # Initial Parameters
    channels = [0]
    mask = None
    bins = [256]
    ranges = [0, 256]

    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    hist_gray = cv.calcHist(imgGray, channels, mask, bins, ranges)
    adjImgGray = cv.cvtColor(adjImag, cv.COLOR_BGR2GRAY)
    hist_grayAdj = cv.calcHist(adjImgGray, channels, mask, bins, ranges)
    if normalize:
        cv.normalize(hist_gray, hist_gray, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
        cv.normalize(hist_grayAdj, hist_grayAdj, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)


    fig, axes = plt.subplots(nrows=4, ncols=2)
    bx11, bx12, bx21, bx22, bx31, bx32,bx41,bx42 = axes.flatten()
    # fig.suptitle(img_name)
    fig.canvas.set_window_title("Compare Original with Adjusted")
    figure = matplotlib.pyplot.gcf()
    figure.set_size_inches(8, 10, forward=True)
    mgr = plt.get_current_fig_manager()
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    bx11.imshow(imgRGB)
    bx11.set_title("Original Image")

    adjRGB = cv.cvtColor(adjImag, cv.COLOR_BGR2RGB)
    bx12.imshow(adjRGB)
    bx12.set_title(img_name)


    if grayHist:
        ## Gray Channel for Img
        bx21.set_title("Gray Channel")
        bx21.plot(hist_gray, color='k')
        mean_gr, std_gr = cv.meanStdDev(imgGray)
        text(0.7, 0.9, r'mu=%.3f, sigma=%.3f' % (mean_gr,std_gr), ha='center', va='center', transform=bx21.transAxes, color="black")

        ## Gray Channel for AdjImg
        bx22.set_title("Gray Channel")
        bx22.plot(hist_grayAdj, color='k')
        mean_grAdj, std_grAdj = cv.meanStdDev(adjImgGray)
        text(0.7, 0.9, r'mu=%.3f, sigma=%.3f' % (mean_grAdj,std_grAdj), ha='center', va='center', transform=bx22.transAxes, color="black")


    color = ('b', 'g', 'r')

    mu_1_all = np.zeros((3,1),dtype= float)
    mu_2_all = np.zeros((3,1),dtype= float)
    sigma_1_all= np.zeros((3,1),dtype= float)
    sigma_2_all = np.zeros((3,1),dtype= float)

    for i, col in enumerate(color):
        histr_org = cv.calcHist([img], [i], mask, bins, ranges)
        histr_adj = cv.calcHist([adjImag], [i], mask, bins, ranges)

        mu_1_all[i] , sigma_1_all[i] = cv.meanStdDev(img[:,:,i])
        x = np.arange(ranges[0],ranges[1])
        bx41.plot(x, stats.norm.pdf(x, mu_1_all[i], sigma_1_all[i] ), color=col, label= col)
        text(0.5, 0.9 - i*0.1, r'mu=%.3f, sigma=%.3f' % (mu_1_all[i],sigma_1_all[i]), ha='center', va='center', transform=bx41.transAxes, color=col)

        mu_2_all[i] , sigma_2_all[i] = cv.meanStdDev(adjImag[:,:,i])
        bx42.plot(x, stats.norm.pdf(x, mu_2_all[i] , sigma_2_all[i] ), color=col, label= col)
        text(0.5, 0.9 - i*0.1, r'mu=%.3f, sigma=%.3f' % (mu_2_all[i],sigma_2_all[i]), ha='center', va='center', transform=bx42.transAxes, color=col)


        if normalize:
            cv.normalize(histr_org, histr_org, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
            cv.normalize(histr_adj, histr_adj, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)

        bx31.plot(histr_org, color=color[i])
        bx31.set_title("BGR Hist")
        bx32.plot(histr_adj, color=color[i])
        bx32.set_title("BGR Hist")


    bx41.legend(loc="upper right"); bx41.grid(True); bx31.grid(True);bx21.grid(True)
    text(0.2, 0.22, r'mu=%.3f' %(mu_1_all[1]), ha='center', va='center', transform=bx41.transAxes,color="black")
    text(0.2, 0.1, r'sigma=%.3f' %(sigma_1_all[1]), ha='center', va='center', transform=bx41.transAxes,color="black")

    bx42.legend(loc="upper right"); bx42.grid(True); bx32.grid(True);bx22.grid(True)
    text(0.2, 0.22, r'mu=%.3f' %(mu_2_all[1]), ha='center', va='center', transform=bx42.transAxes,color="black")
    text(0.2, 0.1, r'sigma=%.3f' %(sigma_2_all[1]), ha='center', va='center', transform=bx42.transAxes,color="black")


    keyboardClick = False
    while keyboardClick != True:
        keyboardClick = plt.waitforbuttonpress()
        plt.close()
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

        hist_plot(img=img, adjImag=adjusted, img_name=gamma, normalize=True)

        keyboardClick = False
        while keyboardClick != True:
            keyboardClick = plt.waitforbuttonpress()
            plt.close()


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

    ## Additional to calculate alpha beta from image
    alow = img.min()
    ahigh = img.max()
    amax = 255
    amin = 0
    # calculate alpha, beta
    alpha = ((amax - amin) / (ahigh - alow))
    beta = amin - alow * alpha

    return res
#_____________________________________________________________________________________
PERCENTILE_STEP = 1
GAMMA_STEP = 0.01
def percentile_to_bias_and_gain(gray, clip_hist_percent):
    # Calculate grayscale histogram
    # Truncate is used to keep the value in the valid range i.e. 0 to +255. If the value goes below 0 it will truncate it
    # to zero and if the value goes above 255 it will truncate it to 255.
    # For example : (-10) will be truncated to 0 and 270 will be truncated to 255.

    clip_hist_percent = clip_hist_percent
    hist = cv.calcHist([gray],[0],None,[256],[0,256])
    hist_size = len(hist)

    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index -1] + float(hist[index]))

    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100.0)
    clip_hist_percent /= 2.0

    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size -1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    return alpha, beta

def adjust_brightness_alpha_beta_gamma(gray_img, minimum_brightness, percentile_step = PERCENTILE_STEP, gamma_step = GAMMA_STEP):
    """Adjusts brightness with histogram clipping by trial and error.
    Algorithms for Adjusting Brightness and Contrast of an Image
    """

    if 3 <= len(gray_img.shape):
        raise ValueError("Expected a grayscale image, color channels found")

    new_img = gray_img
    percentile = percentile_step/2
    gamma = 1
    brightness_changed = False

    while True:
        cols, rows = new_img.shape
        brightness = np.sum(new_img) / (255 * cols * rows)
        mean, stddev = cv.meanStdDev(new_img)

        if not brightness_changed:
            old_brightness = brightness

        if brightness >= minimum_brightness:
            if avgMean < mean :  # this is redundant check
                break

        # adjust alpha and beta
        percentile += percentile_step
        alpha, beta = percentile_to_bias_and_gain(new_img, percentile)
        # print("alpha: %3.3f , beta: %3.3f" %(alpha,beta))
        new_img = cv.convertScaleAbs(gray_img, alpha = alpha, beta = beta)
        brightness_changed = True

        # adjust gamma
        gamma += gamma_step
        new_img = adjust_gamma(new_img, gamma = gamma)

    if brightness_changed:
        print("Old brightness: %3.3f, new brightness: %3.3f, current gamma: %3.3f " %(old_brightness, brightness,gamma))
        print("Current mean: %3.3f, std: %3.3f" %(mean,stddev))
    else:
        print("Maintaining brightness at %3.3f" % old_brightness)
        print("Current mean: %3.3f, std: %3.3f" %(mean,stddev))

    return new_img,gamma

def saturate(img, percentile):
    """Changes the scale of the image so that half of percentile at the low range
    becomes 0, half of percentile at the top range becomes 255.
    """

    if 2 != len(img.shape):
        raise ValueError("Expected an image with only one channel")

    # copy values
    channel = img[:, :].copy()
    flat = channel.ravel()

    # copy values and sort them
    sorted_values = np.sort(flat)

    # find points to clip
    max_index = len(sorted_values) - 1
    half_percent = percentile / 200
    low_value = sorted_values[math.floor(max_index * half_percent)]
    high_value = sorted_values[math.ceil(max_index * (1 - half_percent))]

    # saturate
    channel[channel < low_value] = low_value
    channel[channel > high_value] = high_value

    # scale the channel
    channel_norm = channel.copy()
    cv.normalize(channel, channel_norm, 0, 255, cv.NORM_MINMAX)

    return channel_norm

# _____________________Adjust Gamma (Power Law Transformation)_____________________#
def adjust_gamma(img, gamma):
    """Build a lookup table mapping the pixel values [0, 255] to
    their adjusted gamma values.
    """
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values

    # Gamma correction can be used to correct the brightness of an image
    # by using a non linear transformation between the input values and the mapped output values:

    # code from
    # https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv.LUT(img, table)  # LUT is used to improve the performance of the computation as only 256 values needs to be calculated once.


def adjust_brightness_with_gamma(gray_img, minimum_brightness, gamma_step = GAMMA_STEP):

    """Adjusts the brightness of an image by saturating the bottom and top
    percentiles, and changing the gamma until reaching the required brightness.
    """
    if 3 <= len(gray_img.shape):
        raise ValueError("Expected a grayscale image, color channels found")

    cols, rows = gray_img.shape
    changed = False
    old_brightness = np.sum(gray_img) / (255 * cols * rows)
    new_img = gray_img
    gamma = 1

    while True:
        brightness = np.sum(new_img) / (255 * cols * rows)
        mean, stddev = cv.meanStdDev(new_img)

        if brightness >= minimum_brightness:
            if avgMean < mean:  # this is redundant check
                break

        gamma += gamma_step
        new_img = adjust_gamma(gray_img, gamma = gamma)

        changed = True

    if changed:
        print("Old brightness: %3.3f, new brightness: %3.3f, current gamma: %3.3f  " %(old_brightness, brightness,gamma))
        print("Current mean: %3.3f, std: %3.3f" %(mean,stddev))
    else:
        print("Maintaining brightness at %3.3f" % old_brightness)
        print("Current mean: %3.3f, std: %3.3f" %(mean,stddev))

    return new_img,gamma


# ______________________________________________________________________________#
#                      _______Main__________                                    #
# ______________________________________________________________________________#
avgMean = 140.0
avgStd  = 40.32

if __name__ == '__main__':
    print(" If showHistPlot = True,  please press a random key to skip a plot window ")

    ImgFolderName  = "imagess"
    saveFolderName = str("ImagesNewFolder")

    showHistPlots = True
    writeOut = False
    minBrightness = 0.60

    method_1 = False
    method_2 = True

    Extension_for_files ="png"
    currentDir = os.getcwd()
    newFolderPath = currentDir + "/" + saveFolderName
    ImageSearchFolderPath = os.getcwd() + "/" + ImgFolderName + "/"

    for subdir, dirs, files in os.walk(ImageSearchFolderPath):
        for file in files:
            filepath = subdir + file
            if filepath.endswith("." + Extension_for_files + ""):
                img = cv.imread(filepath)
                new_file_name = filepath[0:-4]

                gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                saturated = saturate(gray, 1)
                print('-------------------------- New Image ----------------------------------')

                if method_1:
                    print("   Method_1: Gamma Threshold   ")
                    bright, gamma = adjust_brightness_with_gamma(saturated, minimum_brightness=minBrightness)
                    adjusted_RGB_gamma = adjust_gamma(img, gamma)
                    if showHistPlots:
                        nameWindow_1 = "Method_1 -> Gamma: " + str(gamma)[0:5]
                        hist_plot(img, adjusted_RGB_gamma, nameWindow_1)

                    if writeOut:
                        if path.exists(newFolderPath):
                            print(str(file), " saved into " + str(saveFolderName) + " folder")
                            new_file_name = str(newFolderPath + "/" + file)[0:-4]
                            cv.imwrite(new_file_name + "_1." + Extension_for_files, adjusted_RGB_gamma)
                        else:
                            print(str(file), " saved into same folder")
                            cv.imwrite(new_file_name + "_1." + Extension_for_files,adjusted_RGB_gamma)

                    print("")
                    #print('-----------------------------------------------------------------------')

                if method_2:
                    print("   Method_2: Alpha_Beta_Gamma   ")
                    brigth_g_a_b, gamma_alpha_beta = adjust_brightness_alpha_beta_gamma(saturated, minimum_brightness=minBrightness)
                    adjusted_RGB_gamma_alpha_beta = adjust_gamma(img,gamma_alpha_beta)

                    if showHistPlots:
                        nameWindow_2 = "Method_2 -> Gamma:" + str(gamma_alpha_beta)[0:5]
                        hist_plot(img, adjusted_RGB_gamma_alpha_beta, nameWindow_2)

                    if writeOut:
                        if path.exists(newFolderPath):
                            print(str(file), " saved into " + str(saveFolderName) + " folder")
                            new_file_name = str(newFolderPath + "/" + file)[0:-4]
                            cv.imwrite(new_file_name + "_2." + Extension_for_files, adjusted_RGB_gamma_alpha_beta)
                        else:
                            print(str(file), " saved into same folder")
                            cv.imwrite(new_file_name + "_2." + Extension_for_files, adjusted_RGB_gamma_alpha_beta)

                    print('-----------------------------------------------------------------------')