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


CCDData = namedtuple("CCDData", "name data median std")
HeaderInfo = namedtuple("HeaderInfo", "name dataL dataU")
ImgInfo = namedtuple("ImgInfo", "name data ccd amp")

###Stack imgs row-wise
def stack_imgs(fitsfiles, n_amps):

    shape = None
    stacked_imgs = None
    stacked_sum = None
    cal = HeaderInfo("cal", [], [])
    dc = HeaderInfo("dc", [], [])
    noiseADU = HeaderInfo("noiseADU", [], [])
    #noise = ImgInfo("noise", [], [])
    id = []

    for i in range(len(fitsfiles)):
        with fits.open(fitsfiles[i]) as hdulist:
            #hdulist.info()
            header = hdulist[0].header
            data = hdulist[1].data
        cal.dataL.append(float(header["MEGAINL"]))
        cal.dataU.append(float(header["MEGAINU"]))
        noiseADU.dataL.append(float(header["MESIGL"]))
        noiseADU.dataU.append(float(header["MESIGU"]))
        dc.dataL.append(float(header["MEDCL"]))
        dc.dataU.append(float(header["MEDCU"]))



        ###get total electrons per column
        sum = np.sum(data, axis=1)

        if n_amps == 2:
            data = np.split(data, 2, axis=1)
            dataL = data[0]/float(header["MEGAINL"])
            dataU = data[1]/float(header["MEGAINU"])
            amp = "L"
            if amp == "L":
                data = dataL
            else:
                data = dataU

        if i==0:
            runids = re.findall("_(\d+)_ped", ''.join(fitsfiles))
            id = ImgInfo("runid", [int(runids[i])], "6415", amp)
            shape = data.shape
            stacked_imgs = data
            stacked_sum = sum
        else:
            if data.shape != shape:
                raise AssertionError("Fits files must be the same shape")
            id.data.append(int(runids[i]))
            stacked_imgs = np.vstack((stacked_imgs, data))
            stacked_sum = np.vstack((stacked_sum, sum))


    return stacked_imgs, stacked_sum, dc, noiseADU, cal, id   #2D arrays of stacked images and stacked sum over cols


###Calculate 1-D projections: median, med_over_sum, MAD
def projection_x(stacked, sums, gain, min_col, max_col, runid):

    calibrated = stacked / gain

    ###Uncomment to flip ADU polarity
    #calibrated = np.amax(calibrated) - calibrated

    pix_vals = CCDData("pix_vals", calibrated.flatten(),
                       np.median(calibrated), np.std(calibrated))

    ###med1 is a simple median over all pixels in col over all imgs
    med1 = np.median(calibrated, axis = 1)
    median = CCDData("median", med1, np.median(med1[min_col:max_col]), np.std(med1[min_col:max_col]))

    mad = stats.median_abs_deviation(calibrated, axis = 1)
    MAD = CCDData("MAD", mad, np.median(mad[min_col:max_col]), np.std(mad[min_col:max_col]))

    ###med2 takes the sum of electrons in col per image and then takes median
    med2 = np.median(sums, axis = 0)
    med_over_sum = CCDData("median", med2, np.median(med2), np.std(med2))

    return (pix_vals, median, MAD, med_over_sum)

###Save medians to ROOT files
def saveROOT(med1, med2, id):

    title = "medians_RUNID{}-{}".format(min(id.data),max(id.data))
    h = ROOT.TH1F("h",title, 3100, 1, 3100)
    for i in range(3100):
        h.Fill(i+1, med1.data[i])
    medians = ROOT.TFile(title, "RECREATE")
    h.Draw()
    h.Write()
    medians.Close()
    h.Reset()

    title = "medians_of_sums_RUNID{}-{}".format(min(id.data),max(id.data))
    h = ROOT.TH1F("h",title, 3100, 1, 3100)
    for i in range(3100):
        h.Fill(i+1, med2.data[i])
    meds_of_sums = ROOT.TFile(title, "RECREATE")
    h.Draw()
    h.Write()
    meds_of_sums.Close()
    h.Reset()
    return



def plot(data, id, amp):
    fig, ax = plt.subplots(figsize=(10, 5.4))
    if amp == "U":
         ax.plot(id, data.dataU, "k.")
    elif amp == "L":
         ax.plot(id, data.dataL, "k.")
    else:
         raise AssertionError("invalid amplifier")
    ax.set_xlabel("{}".format("RUNid"))
    ax.set_ylabel("{}".format(data.name))
    plt.show()
    return

def plot_img(img):

    plt.rcParams.update({'font.size': 15})
    plt.figure(figsize=(9,11))
    #fig,ax1 = plt.subplots()
    #image=image[1:10,3200:-1]
    plt.imshow(img,vmin=0,vmax=0.5,
        aspect="auto",cmap="inferno")
    #plt.imshow(img,vmin=np.mean(img)-0.2*np.std(img),vmax=np.mean(img)+0.2*np.std(img),
#        aspect="auto",cmap="inferno")
    #plt.imshow(image, aspect="auto",cmap="inferno")
    plt.colorbar(label="ADU")
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
    s = np.random.poisson(6,3100)
    s = s*bin_sz

    for i in data.data:
    #for i in s:
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

    for i in range(4):
        func.append("[0]*TMath::Poisson({0},[1])*TMath::Gaus(x,{0}+[2],[3],1)".format(i))

    f = ROOT.TF1("poiss_gauss", " + ".join(func))

    f.SetParNames("norm","#lambda","#mu","#sigma")
    f.SetParameters(norm, lam, mean, noise)

    hist.GetXaxis().SetTitle("{} (e)".format(data.name))
    hist.GetYaxis().SetTitle("Count")

    #hist.Fit(poiss)
    hist.Fit(f, "S L")  ###Use log likelihood method and print results of hte fit
    hist.Draw()

    std = hist.GetStdDev()
    mean = hist.GetMean()

    c.SetLogy()
    c.Update()


    input("Press Enter ........ ")

    return (std, mean)


###Generate mask
def make_mask(mad, med1, med2, strength, radius, id, savemask=False):


    #MAD_mask = np.ma.masked_where(mad.data > mad.median + mad.std*strength, mad.data, copy=False)
    #med_mask = np.ma.masked_where(med2.data > med2.median + med2.std*strength, med2.data, copy=False)
    #med_mask = np.ma.masked_where(med1.data > med1.median + med1.std*strength, med1.data, copy=False)



    med_mask = np.ma.masked_where(np.logical_or(med1.data > med1.median + med1.std*strength,
                             med1.data < med1.median - med1.std*strength), med1.data, copy=False)
    MAD_mask = np.ma.masked_where(np.logical_or(mad.data > mad.median + mad.std*strength,
                             mad.data < mad.median - mad.std*strength), mad.data, copy=False)
    # med_mask = np.ma.masked_where(np.logical_or(med2.data > med2.median + med2.std*strength,
    #                          med2.data < med2.median - med2.std*strength), med2.data, copy=False)


    mask = np.any((med_mask.mask, MAD_mask.mask), axis=0)
    #mask = np.all((med_mask.mask, MAD_mask.mask), axis=0)


    print("{} columns masked".format(mask.sum()))

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
            #if (mask[i-1] or mask[i-2] or mask[i-3] or mask[i-4] or mask[i-5]) and (mask[i+1] or mask[i+2] or mask[i+3] or mask[i+4] or mask[i+5]):
                #mask[i] = True
        if not hit:
            break


    if savemask:
        with open('mask_RUNID{}-{}.npy'.format(min(id.data),max(id.data)), 'wb') as f:
            np.save(f, mask)

    print("{} columns masked".format(mask.sum()))


    fig, ax1 = plt.subplots(figsize=(10, 5.4))
    ax1.set_xlabel("Column number")
    ax1.plot(med1.data, "r+", label=med1.name, markersize=1)
    #ax1.plot(med2.data, "r+", label=med2.name, markersize=1)
    ylab = med2.name + " (e)"
    ax1.set_ylabel(ylab)

    ax2 = ax1.twinx()
    ax3 = ax1.twinx()

    ax2.imshow(np.expand_dims(mask, axis=0), aspect="auto", cmap='binary', alpha=0.4 )
    ax2.get_yaxis().set_visible(False)
    ax3.plot(mad.data, "b+", label=mad.name, markersize=1 )
    ylab = mad.name + " (e)"
    ax3.set_ylabel(ylab)
    plt.title("{} columns masked \n RUNID: {} - {}".format(mask.sum(),min(id.data), max(id.data)))
    plt.legend()
    plt.show()
    return mask

###Mask clusters -- masks all pixels above threshold and all leading pixels
###within radius and all trailing pixels within radius + cti
def mask_clusters(data, threshold, radius, vcti, hcti):
    print(data.shape)
    cols = data.shape[0]
    print(cols)
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
def fit_masked(mask, stacked, cmask):

    ###Expand mask to 2D
    expand_mask = np.tile(mask,stacked.shape[1]).reshape(stacked.T.shape)

    ###Apply mask to pixel array
    pix_vals_masked = np.ma.masked_array(stacked.T, mask=expand_mask)

    ###Mask clusters found from mask_clusters
    clusters_masked = np.ma.masked_array(pix_vals_masked, mask=cmask.T)
    print("pixels masked as clusters = {}".format(cmask.sum()))

    masked = CCDData("masked spectrum", pix_vals_masked.flatten(),
                        np.median(pix_vals_masked), np.std(pix_vals_masked))
    cmasked = CCDData("masked spectrum", clusters_masked.flatten(),
                        np.median(clusters_masked), np.std(clusters_masked))

    ###Fit masked pixels to poiss-gauss
    ROOThist(cmasked, bin_sz=0.01,lam=0.001,norm=cmasked.data.size,mean=0,noise=0.16)
    return


def main():
    parser = argparse.ArgumentParser(description='Generate CCD Mask')
    parser.add_argument('-f','--files', metavar='FILES', type=str, help='FITS files',required=True, nargs='*')
    parser.add_argument('-s','--sigma', metavar='Noise',type=float,default=0.2, help='Noise parameter for single-e peaks fit')
    parser.add_argument('-l','--lam', metavar='Dark current',type=float,default=0.01, help='Dark current parameter for single-e peaks fit')
    parser.add_argument('-g','--gain',metavar='CCD gain', type=float,default=1., help='e to electron conversion for single-e peaks fit')
    args = parser.parse_args()

    stacked, sums, dc, noiseADU, cal, id = stack_imgs(args.files, n_amps=2)
    pix_vals, median, MAD, med_over_sum = projection_x(stacked, sums, args.gain, min_col=0, max_col=310, runid=id)
    #saveROOT(median, med_over_sum, id)

    #plot(dc, id)
    #plot(noiseADU, id)
    #plot(cal, id)
    plot_img(stacked)
    #ROOThist(pix_vals, bin_sz=0.1,lam=0.1,norm=pix_vals.data.size,noise=0.16,mean=0)
    cmask = mask_clusters(stacked, threshold=10, radius=2, vcti=80, hcti=5)
    plot_img(cmask)


    mask = make_mask(MAD, median, med_over_sum, strength=3, radius=0, id=id, savemask=False)
    #mask = np.load("mask_RUN_ID_040-047/L2/L2_mask_325cols.npy")

    fit_masked(mask, stacked, cmask)



if __name__ == '__main__':
    main()
