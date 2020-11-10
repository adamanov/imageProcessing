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

def plt_avg_gauss(norm = True):
    # Channel B
    if not norm:
        avMean_b   = 1315.43                #mu
        avMedian_b = 468.5
        avStd_b    = 2181.725                #Sigma
        avVar_b    = np.square(avStd_b)  
      
    if norm:
        avMean_b   = 0.1074                #mu
        avMedian_b = 0.0394
        avStd_b    = 0.1664                #Sigma
        avVar_b    = np.square(avStd_b)  
    x = np.linspace(avMean_b - 3*avStd_b, avMean_b + 3*avStd_b, 100)
    plt.plot(x, stats.norm.pdf(x, avMean_b, avStd_b),color='b')
    plt.plot(x, stats.norm.pdf(x, avMean_b, avStd_b),color='b')    
    # Channel G
    if not norm:
        avMean_g   = 1315.43 
        avMedian_g = 508.375
        avStd_g    = 1544.725
        avVar_g    = np.square(avStd_g)

    if norm:
        avMean_g   = 0.1445
        avMedian_g = 0.0557
        avStd_g    = 0.20445
        avVar_g    = np.square(avStd_g)
    
    x = np.linspace(avMean_g - 3*avStd_g, avMean_g + 3*avStd_g, 100)
    plt.plot(x, stats.norm.pdf(x, avMean_g, avStd_g),color='g')
    plt.plot(x, stats.norm.pdf(x, avMean_g, avStd_g),color='g')
    # Channel R

    if not norm:
        avMean_r   = 1315.4225
        avMedian_r = 450
        avStd_r    = 1982
        avVar_r    = np.square(avStd_r) 
    if norm:
        avMean_r   = 0.1315
        avMedian_r = 0.0426
        avStd_r    = 0.1966
        avVar_r    = np.square(avStd_r) 
    x = np.linspace(avMean_r - 3*avStd_r, avMean_r + 3*avStd_r, 100)
    plt.plot(x, stats.norm.pdf(x, avMean_r, avStd_r),color ='r')
    plt.plot(x, stats.norm.pdf(x, avMean_r, avStd_r),color ='r')


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


# ______________________Calculate histogram for current image______________________#
def hist_plot(img, adjImag, img_name, normalize=False, gammaPlot=False, grayHist=False):
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


    if not gammaPlot:
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
            mean_n, stddev_n	=	cv.meanStdDev(img[:,:,i])
            mean_n, stddev_n	=	cv.meanStdDev(adjImag[:,:,i])

        if not gammaPlot:
            bx31.plot(histr_org, color=color[i])
            bx31.set_title("BGR Hist")
            ## Add a box into the bx21 and calcualte chanel based mean std median? Do we
            # textstr = '\n'.join((
            #     r'$\mu=%.2f$' % (mu, ),
            #     r'$\mathrm{median}=%.2f$' % (median, ),
            #     r'$\sigma=%.2f$' % (sigma, )))
            # # these are matplotlib.patch.Patch properties
            # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            # bx21.text(0.05, 0.95, textstr, transform=bx21.transAxes, fontsize=14,
            # verticalalignment='top', bbox=props)
            bx32.plot(histr_adj, color=color[i])
            bx32.set_title("BGR Hist")

        if gammaPlot:
            ax21.plot(histr_org, color=color[i],normed=True)
            text(0.8,0.9, "Mean: " + str(mean(hist))[0:5], ha='center', va='center', transform=ax21.transAxes)
            text(0.8,0.8, "Std: " + str(std(hist))[0:5], ha='center', va='center', transform=ax21.transAxes)

            ax21.set_title("BGR Hist")

            ax22.plot(histr_adj, color=color[i])
            ax22.set_title("BGR Hist")
            text(0.8,0.9, "Mean: " + str(mean(histr_adj))[0:5], ha='center', va='center', transform=ax22.transAxes)
            text(0.8,0.8, "Std: " + str(std(hist))[0:5], ha='center', va='center', transform=ax22.transAxes)

    bx41.legend(loc="upper right"); bx41.grid(True)
    text(0.2, 0.22, r'mu=%.3f' %(mu_1_all[1]), ha='center', va='center', transform=bx41.transAxes,color="black")
    text(0.2, 0.1, r'sigma=%.3f' %(sigma_1_all[1]), ha='center', va='center', transform=bx41.transAxes,color="black")

    bx42.legend(loc="upper right"); bx42.grid(True)
    text(0.2, 0.22, r'mu=%.3f' %(mu_2_all[1]), ha='center', va='center', transform=bx42.transAxes,color="black")
    text(0.2, 0.1, r'sigma=%.3f' %(sigma_2_all[1]), ha='center', va='center', transform=bx42.transAxes,color="black")

    # # HSV Histogram Comparasion
    # img_good_hist = normalizedHistogramHSV(img)
    # img_bad_hist = normalizedHistogramHSV(adjImag)
    # bx31.imshow(img_good_hist)
    #
    # bx31.set_title("HSV_Hist")
    # bx32.imshow(img_bad_hist)
    # bx32.set_title("HSV_Hist")
    #
    # comp =cv.compareHist(img_good_hist,img_bad_hist,cv.HISTCMP_CORREL)
    # text(0.71,0.9, "CORREL [-1 to 1] " + str(comp)[0:5], ha='center', va='center', transform=bx31.transAxes,color ="red")
    # comp =cv.compareHist(img_good_hist,img_bad_hist,cv.HISTCMP_CHISQR)
    # text(0.71,0.8, "CHISQR [inf to 0] " + str(comp)[0:5], ha='center', va='center', transform=bx31.transAxes,color ="red")
    # comp =cv.compareHist(img_good_hist,img_bad_hist,cv.HISTCMP_INTERSECT)
    # text(0.68,0.7, "INTERSECT [0 to 1]  " + str(comp)[0:5], ha='center', va='center', transform=bx31.transAxes,color ="red")
    # comp =cv.compareHist(img_good_hist,img_bad_hist,cv.HISTCMP_BHATTACHARYYA)
    # text(0.61,0.6, "BHATTACHARYYA [1 to 0] " + str(comp)[0:5], ha='center', va='center', transform=bx31.transAxes,color ="red")

    if not gammaPlot:
        plt.show(block=False)
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
def adjust_gamma1(image, gammaThreshold):
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
    clip_hist_percent = clip_hist_percent/2
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
    """

    if 3 <= len(gray_img.shape):
        raise ValueError("Expected a grayscale image, color channels found")

    new_img = gray_img
    percentile = percentile_step
    gamma = 1
    brightness_changed = False

    while True:
        cols, rows = new_img.shape
        brightness = np.sum(new_img) / (255 * cols * rows)

        if not brightness_changed:
            old_brightness = brightness

        if brightness >= minimum_brightness:
            break

        # adjust alpha and beta
        percentile += percentile_step
        alpha, beta = percentile_to_bias_and_gain(new_img, percentile)
        new_img = cv.convertScaleAbs(gray_img, alpha = alpha, beta = beta)
        brightness_changed = True

        # adjust gamma
        gamma += gamma_step
        new_img = adjust_gamma(new_img, gamma = gamma)

    if brightness_changed:
        print("Old brightness: %3.3f, new brightness: %3.3f, current gamma: %3.3f " %(old_brightness, brightness,gamma))
    else:
        print("Maintaining brightness at %3.3f" % old_brightness)

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

def adjust_gamma(img, gamma):
    """Build a lookup table mapping the pixel values [0, 255] to
    their adjusted gamma values.
    """
    # code from
    # https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv.LUT(img, table)


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
        print("mean and std:  ", str(mean)[2:7], str(stddev)[2:7])
        #if avgMean - 15 < mean < avgMean + 15 and avgStd - 10 < stddev < avgStd + 10:
        if brightness >= minimum_brightness:   # additional if statement to check average mean = 2.255 and std 8.014
            break

        gamma += gamma_step
        new_img = adjust_gamma(gray_img, gamma = gamma)

        changed = True

    if changed:
        print("Old brightness: %3.3f, new brightness: %3.3f, current gamma: %3.3f  " %(old_brightness, brightness,gamma))
    else:
        print("Maintaining brightness at %3.3f" % old_brightness)

    return new_img,gamma


# ______________________________________________________________________________#
#                      _______Main__________                                    #
# ______________________________________________________________________________#
avgMean = 140.0
avgStd  = 40.32
if __name__ == '__main__':
    bad =   "/home/adamanov/PycharmProjects/imgProcessing/006_1_2_75_left.png"
    bad_2 = "/home/adamanov/PycharmProjects/imgProcessing/009_1_4_75_right.png"
    bad_3 = "/home/adamanov/PycharmProjects/imgProcessing/005_3_2_75_right.png"
    bad_4 = "/home/adamanov/PycharmProjects/imgProcessing/015_2_6_125_left.png"

    good =  "/home/adamanov/PycharmProjects/imgProcessing/639_2_6_175_left.png"
    good_2 ='/home/adamanov/PycharmProjects/imgProcessing/604_1_6_175_right.png'
    good_3 ='/home/adamanov/PycharmProjects/imgProcessing/602_2_5_125_left.png'
    good_4 ='/home/adamanov/PycharmProjects/imgProcessing/603_4_6_175_right.png'

    img_bad = cv.imread(bad)
    img_bad_2 = cv.imread(bad_2)
    img_bad_3 = cv.imread(bad_3)
    img_bad_4 = cv.imread(bad_4)

    img_good   = cv.imread(good)
    img_good_2 = cv.imread(good_2)
    img_good_3 = cv.imread(good_3)
    img_good_4 = cv.imread(good_4)

    filepath = bad_3
    img = cv.imread(filepath)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    saturated = saturate(gray, 1)
    print("gamma method")
    bright, gamma= adjust_brightness_with_gamma(saturated, minimum_brightness = 0.60)
    print("alpha/beta method")
    brigth_g_a_b,gamma_alpha_beta =adjust_brightness_alpha_beta_gamma(saturated, minimum_brightness = 0.60)

    adjusted_RGB_gamma = adjust_gamma(img,gamma)
    adjusted_RGB_gamma_alpha_beta = adjust_gamma(img,gamma_alpha_beta)

    # nameWindow_0 = "gamma vs alpha/beta"
    # cv.namedWindow(nameWindow_0,cv.WINDOW_FREERATIO)
    # cv.imshow(nameWindow_0, np.hstack([bright, brigth_g_a_b]))
    # cv.resizeWindow(nameWindow_0,400,200)
    nameWindow_1 = "with_gamma " + str(gamma)[0:5]
    #cv.namedWindow(nameWindow_1, cv.WINDOW_NORMAL)
    #cv.imshow(nameWindow_1, np.hstack([cv.cvtColor(bright, cv.COLOR_GRAY2BGR), cv.cvtColor(adjusted_RGB_gamma, cv.COLOR_BGR2RGB)]))

    nameWindow_2 = "with_g_a_b " + str(gamma_alpha_beta)[0:5]
    #cv.namedWindow(nameWindow_2, cv.WINDOW_NORMAL)
    #cv.imshow(nameWindow_2, np.hstack([cv.cvtColor(brith_gamma, cv.COLOR_GRAY2BGR), cv.cvtColor(adjusted_RGB_gamma_alpha_beta, cv.COLOR_BGR2RGB)]))


    cv.waitKey(1000)
    print(nameWindow_1)
    hist_plot(img, adjusted_RGB_gamma, nameWindow_1, normalize=True, gammaPlot=False, grayHist=True)
    #print(nameWindow_2)
    #hist_plot(img, adjusted_RGB_gamma_alpha_beta, nameWindow_2, normalize=True, gammaPlot=False, grayHist=True)

    # Local Histogram Equalization

    plt.show()

