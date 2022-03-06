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
import re


CCDData = namedtuple("CCDData", "name data median mad")
PreProc = namedtuple("PreProc", "name data")
ImgInfo = namedtuple("ImgInfo", "name data ccd amp rows")


###Stack imgs row-wise
def stack_imgs(fitsfiles, n_amps):

    shape = None
    stacked_imgs = None
    stacked_sum = None
    cal = PreProc("cal", [])
    dc = PreProc("dc", [])
    noiseADU = PreProc("noiseADU", [])
    noise = PreProc("noise", [])


    for i in range(len(fitsfiles)):

        with fits.open(fitsfiles[i]) as hdulist:
            #hdulist.info()
            header = hdulist[0].header
            data = hdulist[0].data
        cal.data.append(float(header["CAL"]))
        noiseADU.data.append(float(header["NOISE"]))
        dc.data.append(float(header["LAMBDA"]))
        noise.data.append(float(header["NOISE"])/float(header["CAL"]))
        amp = header["AMP"]
        ccd = "6415"
        rows = header["NAXIS2"]

        data = np.array(data)
        #data = np.flip(data.T, axis=1)
        sum =  np.sum(data, axis=0)
        ###get total electrons per column
        #sum = np.sum(data, axis=1)
        #print("sum dimensions={}".format(sum.shape))
        #print(data.shape)
        if i==0:
            runids = re.findall("_(\d+)_{}".format(amp), ''.join(fitsfiles))
            id = ImgInfo("runid", [int(runids[i])], ccd, amp, rows)
            shape = data.shape
            stacked_imgs = data
            stacked_sum = sum
        else:
            if data.shape != shape:
                raise AssertionError("Fits files must be the same shape")
            id.data.append(int(runids[i]))
            stacked_imgs = np.vstack((stacked_imgs, data))
            stacked_sum = np.vstack((stacked_sum, sum))
    print(id.data)
    if amp == "U":
        stacked_imgs = np.flip(stacked_imgs, axis=1)
        stacked_sum = np.flip(stacked_sum, axis=1)
    #stacked_imgs = np.flip(stacked_imgs, axis=0)


    return stacked_imgs, stacked_sum, dc, noiseADU, cal, id, noise  #2D arrays of stacked images and stacked sum over cols


###Calculate 1-D projections: median, med_over_sum, MAD
def projection_x(stacked, sums, gain, min_col, max_col, runid):

    calibrated = stacked / gain

    ###Uncomment to flip ADU polarity
    #calibrated = np.amax(calibrated) - calibrated
    print("stacked = {}".format(stacked.shape))
    print("sums = {}".format(sums.shape))

    pix_vals = CCDData("pix_vals", calibrated.flatten(),
                       np.median(calibrated), stats.median_abs_deviation(calibrated))

    ###med1 is a simple median over all pixels in col over all imgs
    med1 = np.median(calibrated, axis = 0)
    median = CCDData("median", med1, np.median(med1[min_col:max_col]), stats.median_abs_deviation(med1[min_col:max_col]))

    mad = stats.median_abs_deviation(calibrated, axis = 0)
    MAD = CCDData("MAD", mad, np.median(mad[min_col:max_col]), stats.median_abs_deviation(mad[min_col:max_col]))

    ###med2 takes the sum of electrons in col per image and then takes median
    med2 = np.median(sums, axis = 0)
    med_over_sum = CCDData("median", med2, [], [])
    #med_over_sum = CCDData("median", med2, np.median(med2[min_col:max_col]), stats.median_abs_deviation(med2[min_col:max_col]))

    return (pix_vals, median, MAD, med_over_sum)

###Save to ROOT file signed medians as 1-D histos
def saveROOT(med1, med2, id, mask):

    title = "{}{}_masked_meds_RUNID{}-{}.root".format(id.ccd, id.amp, min(id.data),max(id.data))

    f = ROOT.TFile(title, "RECREATE");

    h1 = ROOT.TH1D("h1","medians", len(med1.data), 1, len(med1.data))

    h2 = ROOT.TH1D("h2","median sum/no. img rows", len(med2.data), 1, len(med2.data))
    ###If column i is masked, make median negative
    for i in range(len(med1.data)):
        if mask[i]:
            h1.Fill(i+1, -abs(med1.data[i]))
            h2.Fill(i+1, -abs(med2.data[i]/id.rows))   ###We want median sum/no. of rows in an img
            #h2.Fill(i+1, -abs(med2.data[i]))
        else:
            h1.Fill(i+1, abs(med1.data[i]))
            h2.Fill(i+1, abs(med2.data[i]/id.rows))    ###We want median sum/no. of rows in an img
            #h2.Fill(i+1, abs(med2.data[i]))

    h1.Draw()
    h1.Write()
    h2.Write()
    f.Close()

    return


def plot(data, id):
    fig, ax = plt.subplots(figsize=(10, 5.4))
    ax.plot(id.data, data.data, "k.")
    ax.set_xlabel("{}".format(id.name))
    ax.set_ylabel("{}".format(data.name))
    plt.show()
    return

def plot_img(img):

    plt.rcParams.update({'font.size': 15})
    plt.figure(figsize=(9,11))
    #fig,ax1 = plt.subplots()
    #image=image[1:10,3200:-1]
    plt.imshow(img,vmin=-1,vmax=5,
    #plt.imshow(img,vmin=np.mean(img)-0.5*np.std(img),vmax=np.mean(img)+0.5*np.std(img),
         aspect="auto",cmap="inferno")
    plt.ylim(0, img.shape[0])
    #plt.imshow(img, aspect="auto",cmap="inferno")
    plt.colorbar(label="e")
    plt.show()
    return


###Plot histogram and fit using ROOT
def ROOThist(data, bin_sz, lam, norm, noise, mean, id):

    ROOT.gStyle.SetOptFit(1111)
    ROOT.gStyle.SetOptStat(0)
    c = ROOT.TCanvas("canvas")

    n_bins = (np.amax(data.data)-np.amin(data.data))/bin_sz

    title = "{}{} masked spectrum \n RUNID {} - {}".format(id.amp, id.ccd, min(id.data),max(id.data))
    hist = ROOT.TH1F("hist",title, int(n_bins), np.amin(data.data), np.amax(data.data))
    s = np.random.poisson(6,3100)
    s = s*bin_sz

    for i in data.data:
        hist.Fill(i)

    """
    Fit parameters:
    [0]  norm (number of counts)
    [1]  lambda of poisson
    [2]  gauss mean offset
    [3]  pix noise (sigma of gauss)
    """

    #poiss = ROOT.TF1("poiss", "[0]*TMath::Poisson(x*[2],[1])", 0, 3)
    #poiss = ROOT.TF1("poiss", "TMath::Poisson(x,[0])")
    #poiss.SetParameters(3100,12,1/bin_sz)
    func = []

    for i in range(3):
        func.append("[0]*TMath::Poisson({0},[1])*TMath::Gaus(x,{0}+[2],[3],1)".format(i))

    f = ROOT.TF1("poiss_gauss", " + ".join(func))

    f.SetParNames("norm","#lambda","#mu","#sigma")
    f.SetParameters(norm, lam, mean, noise)


    hist.GetXaxis().SetTitle("{} (e)".format(data.name))
    hist.GetYaxis().SetTitle("Count")
    #hist.GetYaxis().SetRangeUser(0, 1000000)
    #hist.Fit(poiss)
    hist.Fit(f, "LSEM")
    hist.Sumw2()
    hist.Draw()

    std = hist.GetStdDev()
    mean = hist.GetMean()

    c.SetLogy()
    c.Update()


    input("Press Enter ........ ")

    return (std, mean)


###Generate mask
def make_mask(mad, med1, med2, strength, radius, id, savemask=False):


    #MAD_mask = np.ma.masked_where(mad.data > mad.median + mad.mad*strength, mad.data, copy=False)
    #med_mask = np.ma.masked_where(med2.data > med2.median + med2.mad*strength, med2.data, copy=False)
    #med_mask = np.ma.masked_where(med1.data > med1.median + med1.mad*strength, med1.data, copy=False)

    #med_mask = np.ma.masked_where(np.logical_or(med1.data > med1.median + med1.mad*strength,
    #                          med1.data < med1.median - med1.mad*strength), med1.data, copy=False)
    MAD_mask = np.ma.masked_where(np.logical_or(mad.data > mad.median + mad.mad*strength,
                             mad.data < mad.median - mad.mad*strength), mad.data, copy=False)
    med_mask = np.ma.masked_where(np.logical_or(med2.data > med2.median + med2.mad*strength,
                             med2.data < med2.median - med2.mad*strength), med2.data, copy=False)


    mask = np.any((med_mask.mask, MAD_mask.mask), axis=0)
    #mask = np.all((med_mask.mask, MAD_mask.mask), axis=0)

    masked = mask.sum()
    print("{} columns masked".format(masked))

    ###Mask if within radius columns of masked column on both sides
    while True:
        hit = False
        for i in range(radius, len(mask)-radius):
            if not mask[i]:
                for j in range(radius):
                    if mask[i-j]:
                        for k in range(radius):
                            if mask[i+k]:
                                mask[i] = True
                                hit = True
                                break
        if not hit:
            break

    ###Unmask isolated masked columns
    #for i in range(1, len(mask)-1):
    #    if mask[i]:
    #        if not mask[i-1] and not mask[i+1]:
    #            mask[i] = False



#########U
    # mask[0:146] = True
    # mask[310:321] = True
    # mask[185:190] = True
    # mask[200:203] = True
###########

# ##########L
    mask[300:320] = True
    mask[280:320] = True
############

    masked = mask.sum()
    print("{} columns masked".format(masked))


    fig, ax1 = plt.subplots(figsize=(10, 5.4))
    ax1.set_xlabel("Column number")
    ax1.plot(med2.data, "r+", label=med2.name, markersize=5)
    ax1.set_ylim(0,50)
    #ax1.set_ylim(min(med1.data),max(med1.data))
    #ax1.plot(med2.data, "r+", label=med2.name, markersize=1)
    ylab = med2.name + " (e)"
    ax1.set_ylabel(ylab)
    #ax1.set_yscale("log")
    ax1.legend(loc=2)

    ax2 = ax1.twinx()
    ax3 = ax1.twinx()
    plt.legend()
    ax2.imshow(np.expand_dims(mask, axis=0), aspect="auto", cmap='binary', alpha=0.4 )
    ax2.get_yaxis().set_visible(False)
    ax3.set_ylim(0.1,0.2)
    #ax3.set_ylim(min(mad.data),max(mad.data))
    ax3.plot(mad.data, "bx", label=mad.name, markersize=5 )
    #ax3.set_yscale("log")

    ylab = mad.name + " (e)"
    ax3.set_ylabel(ylab)
    plt.title(" {} {}  RUNID: {} - {}\n {} columns masked".format(id.ccd, id.amp,
                                min(id.data), max(id.data), masked))
    plt.legend()
    if savemask:
        with open('{}{}_mask_RUNID{}-{}.npy'.format(id.amp, id.ccd, min(id.data),max(id.data)), 'wb') as f:
            np.save(f, mask)
        plt.savefig('{}{}_mask_RUNID{}-{}.png'.format(id.amp, id.ccd, min(id.data),max(id.data)))

    plt.show()
    return mask

###Mask clusters -- masks all pixels above threshold and all leading pixels
###within radius and all trailing pixels within radius + vcti or hcti
def mask_clusters(data, threshold, radius, vcti, hcti):
    cols = data.shape[0]
    rows = data.shape[1]
    mask = np.zeros((cols, rows),dtype = bool)
    for i in range(cols):
        for j in range(rows):
            if(data[i,j] >= threshold):
                if(i < radius):  #pixel column is within radius of the left edge
                    if(j < radius):  #pixel row is within radius of the bottom
                        mask[:(i+radius+vcti),:(j+radius+hcti)] = True  #mask radius + cti above and to the right of pixel
                    elif((rows - j) <= radius):  #pixel column is within radius from the right edge
                        mask[:(i+radius+vcti),(j-radius):] = True  #mask radius + cti above pixel row and radius to the left of edge
                    else:
                        mask[:(i+radius+vcti),(j-radius):(j+radius+hcti)] = True #mask radius + cti to right of left edge and between radius leading and radius+ cti trailing of row
                elif((cols - i) <= radius):  #pix col is within radius of right edge
                    if(j < radius):  #pix row within radius of bottom
                        mask[(i-radius):,:(j+radius+hcti)] = True #mask radius away from right edge and radius + cti from bottom
                    elif((rows - j) <= radius): #pix within radius of top
                        mask[(i-radius):,(j-radius):] = True  #mask radius to left of pix and radius below
                    else:
                        mask[(i-radius):,(j-radius):(j+radius+hcti)] = True #mask radius to left and radius below and radius + cti above
                else:
                    if(j < radius): #pixel is within radius of bottom
                        mask[(i-radius):(i+radius+vcti),:(j+radius+hcti)] = True  #mask radius to left and radius +cti to right and radius + cti above
                    elif((rows - j) <= radius): #pix within radius of top
                        mask[(i-radius):(i+radius+vcti),(j-radius):] = True
                    else:
                        mask[(i-radius):(i+radius+vcti),(j-radius):(j+radius+hcti)] = True
    return mask

###Fit masked pixels, with clusters masked, to poisson-gauss distribution
def fit_masked(mask, stacked, cmask, id):

    ###Expand mask to 2D
    expand_mask = np.tile(mask,stacked.shape[0]).reshape(stacked.shape)

    ###Apply mask to pixel array
    pix_vals_masked = np.ma.masked_array(stacked, mask=expand_mask)

    ###Mask clusters found from mask_clusters
    clusters_masked = np.ma.masked_array(pix_vals_masked, mask=cmask)
    print("pixels masked as clusters = {}".format(cmask.sum()))

    masked = CCDData("masked spectrum ", pix_vals_masked.flatten(),
                        np.median(pix_vals_masked), stats.median_abs_deviation(pix_vals_masked))
    cmasked = CCDData("masked spectrum ", clusters_masked.flatten(),
                        np.median(clusters_masked), stats.median_abs_deviation(clusters_masked))

    ###Fit masked pixels to poiss-gauss
    ROOThist(cmasked, bin_sz=0.01,lam=0.01,norm=1300,mean=0,noise=0.19,id=id)
    return clusters_masked


def main():
    parser = argparse.ArgumentParser(description='Generate CCD Mask')
    parser.add_argument('-f','--files', metavar='FILES', type=str, help='FITS files',required=True, nargs='*')
    parser.add_argument('-s','--sigma', metavar='Noise',type=float,default=0.2, help='Noise parameter for single-e peaks fit')
    parser.add_argument('-l','--lam', metavar='Dark current',type=float,default=0.01, help='Dark current parameter for single-e peaks fit')
    parser.add_argument('-g','--gain',metavar='CCD gain', type=float,default=1., help='e to electron conversion for single-e peaks fit')
    args = parser.parse_args()

    stacked, sums, dc, noiseADU, cal, id, noise = stack_imgs(args.files, n_amps=1)
    pix_vals, median, MAD, med_over_sum = projection_x(stacked, sums, args.gain,
                              min_col=150, max_col=250, runid=id)


    # plot(dc, id)
    # plot(noise, id)
    # plot(cal, id)
    #plot_img(stacked)
    ROOThist(pix_vals, bin_sz=0.1,lam=0.1,norm=pix_vals.data.size,noise=0.16,mean=0,id=id)
    cmask = mask_clusters(stacked, threshold=10, radius=2, vcti=50, hcti=10)

    #plot_img(cmask)
    #mask = make_mask(MAD, median, med_over_sum, strength=7, radius=3, id=id, savemask=False)
    U ="U6415_mask_RUNID480-725.npy"
    L ="L6415_mask_RUNID480-725.npy"
    mask = np.load(L)

    #saveROOT(median, med_over_sum, id, mask)
    CCD_masked = fit_masked(mask, stacked, cmask, id)
    #plot_img(CCD_masked)
if __name__ == '__main__':
    main()
