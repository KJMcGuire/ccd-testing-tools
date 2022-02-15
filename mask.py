#!/usr/bin/env python3

######################################################
##Author:  Kellie McGuire     kellie@kelliejensen.com
##

######################################################


import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy.ma as ma
from scipy import stats
import argparse
import ROOT


###Compute mean of multi-skip images; stack imgs by row
def avg_skp(fitsfiles, n_amps):

    shape = None
    stacked_imgs = None
    for i in range(len(fitsfiles)):
        with fits.open(fitsfiles[i]) as hdulist:
            #hdulist.info()
            header = hdulist[0].header
            data = hdulist[0].data
        rows = header["NAXIS2"]
        skips = header["NDCMS"]
        cols = header["NAXIS1"] // skips
        if n_amps == 2:
            data = np.split(data, 2, axis=1)

        #data = np.reshape(data, (cols*skips,rows))
        data = np.reshape(data, (cols, rows, skips))
        # data = np.array(data)
        # print(data.shape)

        #print(data.shape)
        if i==0:
            shape = data.shape
            data = np.mean(data, axis=2)
            stacked_imgs = data
        else:
            if data.shape != shape:
                raise AssertionError("Fits files must be the same shape")
            stacked_imgs = np.hstack((stacked_imgs, np.mean(data, axis=2)))

    return stacked_imgs  #2D array of stacked images


###Calculate 1-D projections (sum, median, MAD) and plot results
def projection_x(stacked, gain):

    calibrated = stacked / gain
    e_tot = np.sum(calibrated, axis = 1)
    median = np.median(calibrated[0:3100,:], axis = 1)
    median = np.amax(median)-median
    #print(median)
    MAD = stats.median_abs_deviation(calibrated[0:3100,:], axis = 1)

    fig, ax = plt.subplots(figsize=(10, 5.4))
    #print(MAD[110:120])
    #ax.plot(e_tot)
    #print(calibrated[1,:])
    #ax.plot(MAD[110:120])
    return (e_tot, median, MAD)


###Plot histograms and fit to poisson-gauss
def histogram(data, bin_sz):

    range=(np.amin(data),np.amax(data))
    bins = (np.amax(data)-np.amin(data))/bin_sz
    hist, bin_edges = np.histogram(data,bins=int(bins),range=range)
    bin_centers = 0.5*(bin_edges[1:]+bin_edges[:1])
    plt.rcParams.update({'font.size': 15})
    fig, ax = plt.subplots(figsize=(10,5.4))
    #ax.errorbar(bin_centers, hist, fmt='k_', yerr=hist**0.5)
    n = ax.hist(bin_edges[:-1],bin_edges,align='left',weights=hist,histtype='step')
    return n

###Plot histogram and fit using ROOT
def ROOThist(data, bin_sz,):

    ROOT.gStyle.SetOptFit(111)
    c = ROOT.TCanvas("canvas")

    n_bins = (np.amax(data)-np.amin(data))/bin_sz
    print("max = {}".format(np.amax(data)))
    hist = ROOT.TH1F("hist","hist", int(n_bins), np.amin(data), np.amax(data))
    s = np.random.poisson(6,3100)
    s = s*bin_sz

    for i in data:
        hist.Fill(i)

    """
    Fit parameters:
    [0]  Normalization (number of counts)
    [1]  lambda of poisson
    [2]  scaling factor (1/bin_sz)

    """

    func = ROOT.TF1("func", "[0]*TMath::Poisson(x*[2],[1])", 0, 100)
    #func = ROOT.TF1("poiss_gauss","TMath::Poisson(x,[2])*TMath::Gaus(x,[1],[2])")

    #func.SetParNames("norm","#lambda_{poisson}","scale")
    func.SetParameters(3100,12,1/bin_sz)

    hist.GetXaxis().SetTitle(" (ADU)")
    hist.GetYaxis().SetTitle("No. Cols")


    hist.Fit(func)
    hist.Draw()

    hist.GetXaxis().SetRangeUser(0,100)
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


def testROOT():
    f1= ROOT.TF1("func1","sin(x)", 0, 10)
    f1.Draw()
    input("Press Enter ........ ")
    return



###Generate mask
def mask(data, stddev, mean):


    #print(data[110:120])

    #mask = np.where(condition[data>2])
    mask = np.ma.masked_where(np.logical_or(data>mean+stddev*3,data<mean-stddev*3),data,copy=False)
    #mask = np.ma.masked_where(data>mean+stddev*2,data,copy=False)
    #mask_0 = np.ma.MaskedArray(data, mask=111)
    #mask_0 = np.ma.MaskedArray(data, mask=np.ma.masked_where(data>mean+stddev*1,data,copy=True))

    #print(mask[110:120])
    #print(mask.mask[110:120])
    #print(mask_0[110:120])

    #print(mask0[100:200])
    plt.rcParams.update({'font.size': 15})

    plt.plot(mask.mask)
    # plt.title("")
    plt.xlabel("Columns")
    plt.show()
    return mask




def main():
    parser = argparse.ArgumentParser(description='Compress multi-skip FITS image and fit single-e peaks.')
    parser.add_argument('-f','--files', metavar='FILES', type=str, help='FITS files',required=True, nargs='*')
    parser.add_argument('-s','--sigma', metavar='Noise',type=float,default=0.2, help='Noise parameter for single-e peaks fit')
    parser.add_argument('-l','--lamb', metavar='Dark current',type=float,default=0.01, help='Dark current parameter for single-e peaks fit')
    parser.add_argument('-g','--gain',metavar='CCD gain', type=float,default=1., help='ADU to electron conversion for single-e peaks fit')
    args = parser.parse_args()

    stacked = avg_skp(args.files, n_amps=2)
    e_tot, median, MAD = projection_x(stacked, args.gain)
    #std, mean = ROOThist(MAD, 0.1)
    #masked_MAD = mask(MAD, std, mean)
    #ROOThist(masked_MAD, 0.1)
    std, mean = ROOThist(median, 0.1)
    masked_median = mask(median, std, mean)
    ROOThist(masked_median, 0.1)

    #histogram(median,1)
    #plt.imshow(stacked, vmin=np.nanmean(stacked)-np.nanstd(stacked), vmax=np.nanmean(stacked)+np.nanstd(stacked), cmap="viridis")
    #print(stacked.shape)
    #plt.plot(figsize=(10, 5.4))
    #plt.imshow(stacked.T, vmin=np.nanmean(stacked)-np.nanstd(stacked), vmax=np.nanmean(stacked)+np.nanstd(stacked), aspect='auto', cmap="viridis")

    #plt.imshow(stacked.T[:,3200:-1], vmin=18640, vmax=18645, aspect='auto', cmap="gist_heat")
    #plt.colorbar(label="ADU")



    plt.show()

if __name__ == '__main__':
    main()
