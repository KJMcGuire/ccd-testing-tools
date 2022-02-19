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
ImageInfo = namedtuple("ImageInfo", "name data")


###Stack imgs by row
def stack_imgs(fitsfiles, n_amps):

    shape = None
    stacked_imgs = None
    stacked_sum = None
    cal = ImageInfo("cal", [])
    dc = ImageInfo("dc", [])
    noiseADU = ImageInfo("noiseADU", [])
    noise = ImageInfo("noise", [])
    id = ImageInfo("runid", [])

    for i in range(len(fitsfiles)):
        with fits.open(fitsfiles[i]) as hdulist:
            #hdulist.info()
            header = hdulist[0].header
            data = hdulist[0].data
        rows = int(header["NROW"])
        skips = int(header["NSAMP"])
        # cols = int(header["NCOL"]) // skips
        # dc = float(header["LAMBDA"])
        # noise = float(header["NOISE"])
        # cc = float(header["CAL"])
        # id = int(header["RUNID"])
        cal.data.append(float(header["CAL"]))
        noiseADU.data.append(float(header["NOISE"]))
        dc.data.append(float(header["LAMBDA"]))
        id.data.append(int(header["RUNID"]))
        noise.data.append(float(header["NOISE"])/float(header["CAL"]))
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

    return stacked_imgs, stacked_sum, dc, noiseADU, cal, id, noise  #2D arrays of stacked images and stacked sum over cols


###Calculate 1-D projections (median, med_over_sum, MAD)
def projection_x(stacked, sums, gain, min_col, max_col):

    calibrated = stacked / gain

    ###Uncomment to flip ADU polarity
    #calibrated = np.amax(calibrated) - calibrated

    #e_tot = NamedArray("e_tot", np.sum(calibrated, axis = 1))
    pix_vals = NamedArray("pix_vals", calibrated.flatten(),
                       np.median(calibrated), np.std(calibrated))
    ###med1 is a simle median over all pixels in col over all imgs
    med1 = np.median(calibrated, axis = 1)
    median = NamedArray("median", med1, np.median(med1[min_col:max_col]), np.std(med1[min_col:max_col]))
    mad = stats.median_abs_deviation(calibrated, axis = 1)
    MAD = NamedArray("MAD", mad, np.median(mad[min_col:max_col]), np.std(mad[min_col:max_col]))
    ###med2 takes the sum of electrons in col per image and then takes median
    med2 = np.median(sums, axis = 0)
    med_over_sum = NamedArray("median", med2, np.median(med2[min_col:max_col]), np.std(med2[min_col:max_col]))


    return (pix_vals, median, MAD, med_over_sum)


def plot(data):
    fig, ax = plt.subplots(figsize=(10, 5.4))
    ax.plot(data.data, "k.")
    ax.set_xlabel("")
    ax.set_ylabel("{}".format(data.name))
    plt.show()
    return


###Plot histogram and fit using ROOT
def ROOThist(data, bin_sz, lam, norm, noise, mean):

    ROOT.gStyle.SetOptFit(1111)
    ROOT.gStyle.SetOptStat(0)
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
    [0]  norm (number of counts)
    [1]  lambda of poisson
    [2]  gauss mean offset
    [3]  pix noise (sigma of gauss)


    """

    #func = ROOT.TF1("func", "[0]*TMath::Poisson(x*[2],[1])", -10, 100)
    func = []

    for i in range(3):
        func.append("[0]*TMath::Poisson({0},[1])*TMath::Gaus(x,{0}+[2],[3],1)".format(i))

    f = ROOT.TF1("poiss_gauss", " + ".join(func))

    f.SetParNames("norm","#lambda","#mu","#sigma")
    f.SetParameters(norm, lam, mean, noise)

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


    MAD_mask = np.ma.masked_where(mad.data > mad.median + mad.std*strength, mad.data, copy=False)
    #med_mask = np.ma.masked_where(med2.data > med2.median + med2.std*strength, med2.data, copy=False)
    med_mask = np.ma.masked_where(med1.data > med1.median + med1.std*strength, med1.data, copy=False)


    # med_mask = np.ma.masked_where(np.logical_or(med1.data > med1.median + med1.std*strength,
    #                          med1.data < med1.median - med1.std*strength), med1.data, copy=False)
    # MAD_mask = np.ma.masked_where(np.logical_or(mad.data > mad.median + mad.std*strength,
    #                         mad.data < mad.median - mad.std*strength), mad.data, copy=False)
    # med_mask = np.ma.masked_where(np.logical_or(med2.data > med2.median + med2.std*strength,
    #                         med2.data < med2.median - med2.std*strength), med2.data, copy=False)


    mask = np.any((med_mask.mask, MAD_mask.mask), axis=0)
    #mask = np.all((med_mask.mask, MAD_mask.mask), axis=0)


    masked = mask.sum()
    print("{} columns masked".format(masked))

    ###Mask if within three columns of masked column on both sides
    while True:
        hit = False
        for i in range(len(mask)-3):
            if not mask[i]:
                if (mask[i-1] or mask[i-2] or mask[i-3]) and (mask[i+1] or mask[i+2] or mask[i+3]):
                    mask[i] = True
                    hit = True
        if not hit:
            break



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

###Mask clusters -- masks all pixels above threshold and all leading pixels
###within radius and all trailing pixels within radius + cti
def mask_clusters(data, threshold, radius, cti):
    cols = data.shape[0]
    rows = data.shape[1]
    mask = np.zeros((cols, rows),dtype = bool)
    for i in range(cols):
        for j in range(rows):
            if(data[i,j] >= threshold):
                if(i < radius):  #pixel column is within radius of the left edge
                    if(j < radius):  #pixel row is within radius of the bottom
                        mask[:(i+radius+cti),:(j+radius+cti)] = True  #mask radius + 1 above and to the right of pixel

                    elif((rows - j) <= radius):  #pixel column is within radius from the right edge
                        mask[:(i+radius+cti),(j-radius):] = True  #mask radius + 1 above pixel row and radius to the left of edge
                    else:
                        mask[:(i+radius+cti),(j-radius):(j+radius+cti)] = True #mask radius + 1 to right of left edge and between radius leading  and radius+1 trailing of row
                elif((cols - i) <= radius):  #pix col is within radius of right edge
                    if(j < radius):  #pix row within radius of bottom
                        mask[(i-radius):,:(j+radius+cti)] = True #mask radius away from right edge and radius + 1 from bottom
                    elif((rows - j) <= radius): #pix within radius of top
                        mask[(i-radius):,(j-radius):] = True  #mask radius to left of pix and radius below
                    else:
                        mask[(i-radius):,(j-radius):(j+radius+cti)] = True #mask radius to left and radius below and radius +1 above
                else:
                    if(j < radius): #pixel is within radius of bottom
                        mask[(i-radius):(i+radius+cti),:(j+radius+cti)] = True  #mask radius to left and radius +1 to right and radius + 1 above
                    elif((rows - j) <= radius): #pix within radius of top
                        mask[(i-radius):(i+radius+cti),(j-radius):] = True
                    else:
                        mask[(i-radius):(i+radius+cti),(j-radius):(j+radius+cti)] = True
    return mask


def main():
    parser = argparse.ArgumentParser(description='Generate CCD Mask')
    parser.add_argument('-f','--files', metavar='FILES', type=str, help='FITS files',required=True, nargs='*')
    parser.add_argument('-s','--sigma', metavar='Noise',type=float,default=0.2, help='Noise parameter for single-e peaks fit')
    parser.add_argument('-l','--lam', metavar='Dark current',type=float,default=0.01, help='Dark current parameter for single-e peaks fit')
    parser.add_argument('-g','--gain',metavar='CCD gain', type=float,default=1., help='e to electron conversion for single-e peaks fit')
    args = parser.parse_args()

    stacked, sums, dc, noiseADU, cal, id, noise = stack_imgs(args.files, n_amps=1)
    pix_vals, median, MAD, med_over_sum = projection_x(stacked, sums, args.gain, min_col=2000, max_col=-1)
    print(med_over_sum.median)
    print(med_over_sum.std)

    plot(dc)
    plot(noise)
    plot(cal)
    #plot(med_over_sum)
    #ROOThist(pix_vals, bin_sz=0.01,lam=0.001,norm=pix_vals.data.size,noise=0.16,mean=0)
    cmask = mask_clusters(stacked.T, threshold=5, radius=5, cti=5)


    #MAD_std, MAD_med = ROOThist(MAD, 0.001, 0.1, norm=MAD.data.size,noise=0.16,mean=0)
    #med_std, med_med = ROOThist(med_over_sum, 0.001, 0, norm=med_over_sum.data.size,noise=0.16,mean=0)
    #med_sum_std, med_sum_med = ROOThist(med_over_sum, 0.1, 0, norm=median.data.size,noise=0.16,mean=0)

    sig_mask = 3
    #mask = make_mask(MAD, median, med_over_sum, sig_mask, MAD_std, MAD_med, med_std, med_med)
    mask = make_mask(MAD, median, med_over_sum, sig_mask)


    expand_mask = np.tile(mask,stacked.shape[1]).reshape(stacked.T.shape)


    pix_vals_masked = np.ma.masked_array(stacked.T, mask=expand_mask)
    clusters_masked = np.ma.masked_array(pix_vals_masked, mask=cmask)


    # with open('U2_mask_02_18.npy', 'wb') as f:
    #    np.save(f, mask)

    masked = NamedArray("masked spectrum ({}#sigma)".format(sig_mask), pix_vals_masked.flatten(),
                            np.median(pix_vals_masked), np.std(pix_vals_masked))

    cmasked = NamedArray("masked spectrum ({}#sigma)".format(sig_mask), clusters_masked.flatten(),
                            np.median(clusters_masked), np.std(clusters_masked))
    ROOThist(cmasked, bin_sz=0.01,lam=0.001,norm=cmasked.data.size,mean=0,noise=0.16)


if __name__ == '__main__':
    main()
