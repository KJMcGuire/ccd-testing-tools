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
import scipy
from scipy import stats

print(scipy.__version__)
import argparse
import ROOT
from collections import namedtuple

NamedArray = namedtuple("NamedArray", "name data")


###Compute mean of multi-skip images; stack imgs by row
def stack_imgs(fitsfiles, n_amps, skp_img):

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
        if n_amps == 2:
            data = np.split(data, 2, axis=1)

        if skp_img:
            data = np.reshape(data, (cols, rows, skips))
            data = np.mean(data, axis=2)
        else:
            data = np.array(data)
            data = np.flip(data.T, axis=1)

            ##sum over rows to get total e per col
            sum = np.sum(data, axis=1)


        #print(data.shape)
        if i==0:
            shape = data.shape
            # if skp_img:
            #     data = np.mean(data, axis=2)
            stacked_imgs = data
            stacked_sum = sum
        else:
            if data.shape != shape:
                raise AssertionError("Fits files must be the same shape")
            #stacked_imgs = np.hstack((stacked_imgs, np.mean(data, axis=2)))
            stacked_imgs = np.hstack((stacked_imgs, data))
            stacked_sum = np.vstack((stacked_sum, sum))


    return stacked_imgs, stacked_sum  #2D array of stacked images and stacked sums


###Calculate 1-D projections (sum, median, MAD)
def projection_x(stacked, sums, gain):

    calibrated = stacked / gain
    pix_vals = NamedArray("pix_vals", calibrated.flatten())
    #calibrated = np.amax(calibrated) - calibrated
    e_tot = NamedArray("e_tot", np.sum(calibrated, axis = 1))

    median = NamedArray("median", np.median(calibrated, axis = 1))
    med_over_sum = NamedArray("med_over_sum", np.median(sums, axis = 0))

    print("med min = {}".format(np.amin(median.data)))
    print("med max = {}".format(np.amax(median.data)))
    MAD = NamedArray("MAD", stats.median_abs_deviation(calibrated, axis = 1))
    return (e_tot, median, MAD, med_over_sum)


def plot(data):
    fig, ax = plt.subplots(figsize=(10, 5.4))
    ax.plot(data.data, "k.")
    ax.set_xlabel("Column number".format(data.name))
    ax.set_ylabel("{} (e)".format(data.name))
    plt.show()
    return



###Plot histogram and fit using ROOT
def ROOThist(data, bin_sz, lam, norm):

    ROOT.gStyle.SetOptFit(111)
    c = ROOT.TCanvas("canvas")

    n_bins = (np.amax(data.data)-np.amin(data.data))/bin_sz
    #n_bins=100/bin_sz
    hist = ROOT.TH1F("hist","hist", int(n_bins), np.amin(data.data), np.amax(data.data))
    #s = np.random.poisson(6,3100)
    #s = s*bin_sz

    for i in data.data:
        hist.Fill(i)

    """
    Fit parameters:
    [0]  Normalization (number of counts)
    [1]  lambda of poisson
    [2]  scaling factor (1/bin_sz)

    """

    func = ROOT.TF1("func", "[0]*TMath::Poisson(x*[2],[1])", 0, 20)
    #func = ROOT.TF1("poiss_gauss","TMath::Poisson(x,[2])*TMath::Gaus(x,[1],[2])")

    #func.SetParNames("norm","#lambda_{poisson}","scale")
    func.SetParameters(norm, lam, 1/bin_sz)

    hist.GetXaxis().SetTitle("{} (e)".format(data.name))
    hist.GetYaxis().SetTitle("No. Cols")


    hist.Fit(func)
    hist.Draw()

    #hist.GetXaxis().SetRangeUser(0,100)
    hist.SetTitle("[0]*TMath::Poisson(x*[2],[1])")
    #hist.SetCanExtend(ROOT.TH1.kAllAxes)
    #hist.ExtendAxis(10, hist.GetXaxis())
    std = hist.GetStdDev()
    mean = hist.GetMean()
    #print(std)
    #print(mean)
    c.SetLogy()
    c.Update()


    input("Press Enter ........ ")

    return (std, mean)


###Generate mask
def make_mask(meds, MADs, MAD_std, MAD_mean, med_std, med_mean):

    med_mask = np.ma.masked_where(np.logical_or(meds.data>med_mean+med_std*3,
                            meds.data<med_mean-med_std*3),meds.data,copy=False)
    MAD_mask = np.ma.masked_where(MADs.data>MAD_mean+MAD_std*3,MADs.data,copy=False)
    mask = np.any((med_mask.mask, MAD_mask.mask), axis=0)


    masked = mask.sum()
    print("{} columns masked".format(masked))



    ##Mask column if adjacent to two masked columns
    for i in range(len(mask)):
        if (mask[i-1] and mask[i+1]):
            mask[i] = True


    masked = mask.sum()
    print("{} columns masked".format(masked))


    plt.imshow(np.expand_dims(mask, axis=0), aspect="auto", cmap='binary')
    plt.title("{} columns masked".format(masked))
    plt.xlabel("Columns")
    plt.show()
    return mask


def main():
    parser = argparse.ArgumentParser(description='Generate CCD Mask')
    parser.add_argument('-f','--files', metavar='FILES', type=str, help='FITS files',required=True, nargs='*')
    parser.add_argument('-s','--sigma', metavar='Noise',type=float,default=0.2, help='Noise parameter for single-e peaks fit')
    parser.add_argument('-l','--lamb', metavar='Dark current',type=float,default=0.01, help='Dark current parameter for single-e peaks fit')
    parser.add_argument('-g','--gain',metavar='CCD gain', type=float,default=1., help='e to electron conversion for single-e peaks fit')
    args = parser.parse_args()

    stacked, sums = stack_imgs(args.files, n_amps=1, skp_img=False)
    e_tot, median, MAD, med_over_sum = projection_x(stacked, sums, args.gain)
    #plot(e_tot)
    #plot(median)
    #plot(MAD)
    #plot(med_over_sum)
    #ROOThist(stacked, 0.1,0.1,3100)
    MAD_std, MAD_mean = ROOThist(MAD, 0.001, 0.1, 3100)
    med_std, med_mean = ROOThist(median, 0.001, 0, 3100)
    med_sum_std, med_sum_mean = ROOThist(med_over_sum, 0.01, 0, 3100)


    mask = make_mask(median, MAD, MAD_std, MAD_mean, med_std, med_mean)
    #med_masked = np.ma.masked_array(median, mask=mask)
    #MAD_masked = np.ma.masked_array(MAD, mask=mask)
    #ROOThist(med_masked, 0.1)


if __name__ == '__main__':
    main()
