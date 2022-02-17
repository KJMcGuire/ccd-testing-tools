#!/usr/bin/env python3

######################################################
##Author:  Kellie McGuire     kellie@kelliejensen.com
##Compute median and MAD of pixel values per column
##Generate mask and apply to pixel distribution
######################################################


import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy.ma as ma
from scipy import stats
import argparse
import ROOT
from collections import namedtuple

NamedArray = namedtuple("NamedArray", "name data median std")


###Stack imgs by row
def stack_imgs(fitsfiles, n_amps):

    shape = None
    stacked_imgs = None
    stacked_sum = None
    for i in range(len(fitsfiles)):
        with fits.open(fitsfiles[i]) as hdulist:
            #hdulist.info()
            header = hdulist[0].header
            data = hdulist[0].data
        rows = int(header["NROW"])
        skips = int(header["NSAMP"])
        cols = int(header["NCOL"]) // skips
        # rows = int(header["NAXIS2"])
        # skips = int(header["NDCMS"])
        # cols = int(header["NAXIS1"]) // skips
        if n_amps == 2:
            data = np.split(data, 2, axis=1)
        else:
            data = np.array(data)
            data = np.flip(data.T, axis=1)

            ###sum over rows to get total e per col
            sum = np.sum(data, axis=1)

        #print(data.shape)
        if i==0:
            shape = data.shape
            stacked_imgs = data
            stacked_sum = sum
        else:
            if data.shape != shape:
                raise AssertionError("Fits files must be the same shape")
            stacked_imgs = np.hstack((stacked_imgs, data))
            stacked_sum = np.vstack((stacked_sum, sum))

    return stacked_imgs, stacked_sum  #2D arrays of stacked images and stacked sum over cols


###Calculate 1-D projections (median, med_over_sum, MAD)
def projection_x(stacked, sums, gain):

    calibrated = stacked / gain

    ###Uncomment to flip ADU polarity
    #calibrated = np.amax(calibrated) - calibrated

    #e_tot = NamedArray("e_tot", np.sum(calibrated, axis = 1))
    pix_vals = NamedArray("pix_vals", calibrated.flatten(),
                       np.median(calibrated), np.std(calibrated))
    ###med1 is a simle median over all pixels in col over all imgs
    med1 = np.median(calibrated, axis = 1)
    median = NamedArray("median", med1, np.median(med1), np.std(med1))
    mad = stats.median_abs_deviation(calibrated, axis = 1)
    MAD = NamedArray("MAD", mad, np.median(mad), np.std(mad))
    ###med2 takes the sum of electrons in col per image and then takes median
    med2 = np.median(sums, axis = 0)
    med_over_sum = NamedArray("median", med2, np.median(med2), np.std(med2))


    return (pix_vals, median, MAD, med_over_sum)


def plot(data):
    fig, ax = plt.subplots(figsize=(10, 5.4))
    ax.plot(data.data, "k.")
    ax.set_xlabel("Column number".format(data.name))
    ax.set_ylabel("{} (e)".format(data.name))
    plt.show()
    return


###Plot histogram and fit using ROOT
def ROOThist(data, bin_sz, lam, norm, noise, mean):

    ROOT.gStyle.SetOptFit(10101)
    c = ROOT.TCanvas("canvas")

    n_bins = (np.amax(data.data)-np.amin(data.data))/bin_sz

    title = data.name
    hist = ROOT.TH1F("hist",title, int(n_bins), np.amin(data.data), np.amax(data.data))
    #s = np.random.poisson(6,3100)
    #s = s*bin_sz

    for i in data.data:
        hist.Fill(i)

    """
    Fit parameters:
    [0]  Norm (number of counts)
    [1]  lambda of poisson
    [2]  scaling factor
    [3]  pix noise (sigma of gauss)
    [4]  gauss mean offset


    """

    #func = ROOT.TF1("func", "[0]*TMath::Poisson(x*[2],[1])", -10, 100)
    func = []

    for i in range(6):
        func.append("[0]*TMath::Poisson({0}*[2],[1])*TMath::Gaus(x,{0}+[4],[3],1)".format(i))

    f = ROOT.TF1("poiss_gauss", " + ".join(func))

    f.SetParNames("norm","#lambda","scale","#sigma","#mu")
    f.SetParameters(norm, lam, 1., noise, mean)

    hist.GetXaxis().SetTitle("{} (e)".format(data.name))
    hist.GetYaxis().SetTitle("Count")

    hist.Fit(f)
    hist.Draw()

    #hist.GetXaxis().SetRangeUser(-5,7)
    #hist.SetTitle("[0]*TMath::Poisson(x*[2],[1])")
    #hist.SetCanExtend(ROOT.TH1.kAllAxes)
    #hist.ExtendAxis(10, hist.GetXaxis())
    std = hist.GetStdDev()
    mean = hist.GetMean()

    c.SetLogy()
    c.Update()


    input("Press Enter ........ ")

    return (std, mean)


###Generate mask
def make_mask(mad, med1, med2, strength):

    med_mask = np.ma.masked_where(np.logical_or(med1.data > med1.median + med1.std*strength,
                            med1.data < med1.median - med1.std*strength), med1.data, copy=False)
    MAD_mask = np.ma.masked_where(np.logical_or(mad.data > mad.median + mad.std*strength,
                            mad.data < mad.median - mad.std*strength), mad.data, copy=False)
    med_mask = np.ma.masked_where(np.logical_or(med2.data > med2.median + med2.std*strength,
                            med2.data < med2.median - med2.std*strength), med2.data, copy=False)

    mask = np.any((med_mask.mask, MAD_mask.mask), axis=0)


    masked = mask.sum()
    print("{} columns masked".format(masked))

    ###Mask column if adjacent to two masked columns
    for i in range(len(mask)-1):
        if (mask[i-1] and mask[i+1]):
            mask[i] = True

    for j in range(len(mask)-1):
        if (mask[j-2] and mask[j+2]):
            mask[j] = True


    masked = mask.sum()
    print("{} columns masked".format(masked))


    fig, ax1 = plt.subplots(figsize=(10, 5.4))
    ax1.set_xlabel("Column number")
    ax1.plot(med2.data, "r+", label=med2.name, markersize=1)
    ylab = med2.name + " (e)"
    ax1.set_ylabel(ylab)

    ax2 = ax1.twinx()
    ax3 = ax1.twinx()

    ax2.imshow(np.expand_dims(mask, axis=0), aspect="auto", cmap='binary', alpha=0.4 )
    ax2.get_yaxis().set_visible(False)
    ax3.plot(mad.data, "b+", label=mad.name, markersize=1 )
    ylab = mad.name + " (e)"
    ax3.set_ylabel(ylab)
    plt.title("{} columns masked  --  {} sigma cut".format(masked,strength))
    plt.legend()
    plt.show()
    return mask


def main():
    parser = argparse.ArgumentParser(description='Generate CCD Mask')
    parser.add_argument('-f','--files', metavar='FILES', type=str, help='FITS files',required=True, nargs='*')
    parser.add_argument('-s','--sigma', metavar='Noise',type=float,default=0.2, help='Noise parameter for single-e peaks fit')
    parser.add_argument('-l','--lam', metavar='Dark current',type=float,default=0.01, help='Dark current parameter for single-e peaks fit')
    parser.add_argument('-g','--gain',metavar='CCD gain', type=float,default=1., help='e to electron conversion for single-e peaks fit')
    args = parser.parse_args()

    stacked, sums = stack_imgs(args.files, n_amps=1)
    pix_vals, median, MAD, med_over_sum = projection_x(stacked, sums, args.gain)
    #plot(e_tot)
    #plot(median)
    # plot(MAD)
    #plot(med_over_sum)
    ROOThist(pix_vals, bin_sz=0.01,lam=0.001,norm=pix_vals.data.size,noise=0.16,mean=0)
    #MAD_std, MAD_med = ROOThist(MAD, 0.001, 0.1, norm=MAD.data.size,noise=0.16,mean=0)
    #med_std, med_med = ROOThist(median, 0.001, 0, norm=median.data.size,noise=0.16,mean=0)
    #med_sum_std, med_sum_med = ROOThist(med_over_sum, 0.1, 0, norm=median.data.size,noise=0.16,mean=0)

    sig_mask = 2
    #mask = make_mask(median, MAD, MAD_std, MAD_mean, med_std, med_mean)
    mask = make_mask(MAD, median, med_over_sum, sig_mask)



    expand_mask = np.tile(mask,stacked.shape[1]).reshape(stacked.T.shape)
    #print(stacked.shape)
    #print(expand_mask.shape)
    pix_vals_masked = np.ma.masked_array(stacked.T, mask=expand_mask)

    masked = NamedArray("masked spectrum ({}#sigma)".format(sig_mask), pix_vals_masked.flatten(),
                            np.median(pix_vals_masked), np.std(pix_vals_masked))
    ROOThist(masked, bin_sz=0.01,lam=0.001,norm=masked.data.size,noise=0.16,mean=0)


if __name__ == '__main__':
    main()
