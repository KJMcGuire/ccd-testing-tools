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
from fit_SE_peaks import plot_img



CCDData = namedtuple("CCDData", "name data median mad")
PreProc = namedtuple("PreProc", "name data")
ImgInfo = namedtuple("ImgInfo", "name data ccd amp")


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
        #rows = int(header["NROW"])
        #skips = int(header["NSAMP"])
        #cols = int(header["NCOL"]) // skips
        cal.data.append(float(header["CAL"]))
        noiseADU.data.append(float(header["NOISE"]))
        dc.data.append(float(header["LAMBDA"]))
        noise.data.append(float(header["NOISE"])/float(header["CAL"]))

        if n_amps == 2:
            data = np.split(data, 2, axis=0)

        data = np.array(data)
        data = np.flip(data.T, axis=1)

        ###get total electrons per column
        sum = np.sum(data, axis=1)

        #print(data.shape)
        if i==0:
            id = ImgInfo("runid", [int(header["RUNID"])], str(header["LTANAME"]), str(header["AMP"]))
            shape = data.shape
            stacked_imgs = data
            stacked_sum = sum
        else:
            if data.shape != shape:
                raise AssertionError("Fits files must be the same shape")
            id.data.append(int(header["RUNID"]))
            stacked_imgs = np.hstack((stacked_imgs, data))
            stacked_sum = np.vstack((stacked_sum, sum))

    return stacked_imgs, stacked_sum, dc, noiseADU, cal, id, noise  #2D arrays of stacked images and stacked sum over cols


###Calculate 1-D projections: median, med_over_sum, MAD
def projection_x(stacked, sums, gain, min_col, max_col, runid):

    calibrated = stacked / gain

    ###Uncomment to flip ADU polarity
    #calibrated = np.amax(calibrated) - calibrated

    pix_vals = CCDData("pix_vals", calibrated.flatten(),
                       np.median(calibrated), stats.median_abs_deviation(calibrated))

    ###med1 is a simple median over all pixels in col over all imgs
    med1 = np.median(calibrated, axis = 1)
    median = CCDData("median", med1, np.median(med1[min_col:max_col]), stats.median_abs_deviation(med1[min_col:max_col]))

    mad = stats.median_abs_deviation(calibrated, axis = 1)
    MAD = CCDData("MAD", mad, np.median(mad[min_col:max_col]), stats.median_abs_deviation(mad[min_col:max_col]))

    ###med2 takes the sum of electrons in col per image and then takes median
    med2 = np.median(sums, axis = 0)

    med_over_sum = CCDData("median", med2, np.median(med2), stats.median_abs_deviation(med2))
    #med_over_sum = CCDData("median", med2, np.median(med2[min_col:max_col]), stats.median_abs_deviation(med2[min_col:max_col]))

    return (pix_vals, median, MAD, med_over_sum)

###Save medians to ROOT files
def saveROOT(med1, mads, id):

    title = "{}{}_medians_RUNID{}-{}.root".format(id.amp, id.ccd, min(id.data),max(id.data))
    h = ROOT.TH1F("h",title, 3100, 1, 3100)
    for i in range(3100):
        h.Fill(i+1, med1.data[i])
    medians = ROOT.TFile(title, "RECREATE")
    h.Draw()
    h.Write()
    medians.Close()
    h.Reset()

    title = "{}{}_MADs_RUNID{}-{}.root".format(id.amp, id.ccd,min(id.data),max(id.data))
    h = ROOT.TH1F("h",title, 3100, 1, 3100)
    for i in range(3100):
        h.Fill(i+1, mads.data[i])
    mads = ROOT.TFile(title, "RECREATE")
    h.Draw()
    h.Write()
    mads.Close()
    h.Reset()
    return



def plot(data, id):
    fig, ax = plt.subplots(figsize=(10, 5.4))
    ax.plot(id.data, data.data, "k.")
    ax.set_xlabel("{}".format(id.name))
    ax.set_ylabel("{}".format(data.name))
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

    for i in range(4):
        func.append("[0]*TMath::Poisson({0},[1])*TMath::Gaus(x,{0}+[2],[3],1)".format(i))

    f = ROOT.TF1("poiss_gauss", " + ".join(func))

    f.SetParNames("norm","#lambda","#mu","#sigma")
    f.SetParameters(norm, lam, mean, noise)


    hist.GetXaxis().SetTitle("{} (e)".format(data.name))
    hist.GetYaxis().SetTitle("Count")

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
def make_mask(mad, med1, med2, strength, radius, id, tail, savemask=False):


    #MAD_mask = np.ma.masked_where(mad.data > mad.median + mad.std*strength, mad.data, copy=False)
    #med_mask = np.ma.masked_where(med2.data > med2.median + med2.std*strength, med2.data, copy=False)
    #med_mask = np.ma.masked_where(med1.data > med1.median + med1.std*strength, med1.data, copy=False)



    #med_mask = np.ma.masked_where(np.logical_or(med1.data > med1.median + med1.std*strength,
    #                         med1.data < med1.median - med1.std*strength), med1.data, copy=False)
    MAD_mask = np.ma.masked_where(np.logical_or(mad.data > mad.median + mad.mad*strength,
                             mad.data < mad.median - mad.mad*strength), mad.data, copy=False)
    med_mask = np.ma.masked_where(np.logical_or(med2.data > med2.median + med2.mad*strength,
                              med2.data < med2.median - med2.mad*strength), med2.data, copy=False)


    #mask = med_mask.mask
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
            #if (mask[i-1] or mask[i-2] or mask[i-3] or mask[i-4] or mask[i-5]) and (mask[i+1] or mask[i+2] or mask[i+3] or mask[i+4] or mask[i+5]):
                #mask[i] = True
        if not hit:
            break

    ###Unmask isolated masked columns
    #for i in range(1, len(mask)-1):
    #    if mask[i]:
    #        if not mask[i-1] and not mask[i+1]:
    #            mask[i] = False



    ###Mask tail number of columns trailing masked column if column number <1500
    ### and tail*2 columns if column number >1500
    skipping = 0
    #for i in range(0, 1500):
    #    if skipping > 0:
    #        skipping -= 1
    #        continue
    #    if mask[i]: 
    #        for j in range(1, tail):
    #            mask[i+j] = True
    #            skipping += 1
    #skipping = 0
    for i in range(1500,len(mask)-tail*2):
        #if med2.data[i] > med2.median + med2.std*7.:
        if skipping > 0:
            skipping -= 1
            continue
        if mask[i]:
            for j in range(1,tail*2):
                mask[i+j] = True
                skipping += 1
     


    mask[0:10] = True
    mask[3082:-1] = True
##########U2
    #mask[0:1300] = True
    #mask[1330:1350] = False
    # mask[1012:1018] = True
    # mask[1138:1142] = True
    # mask[1730:1740] = True
############

#########U1
    #mask[0:900] = True
    mask[0:900] = True
    #mask[1360:1380] = True
    #mask[1540:1547] = True
    #mask[2276:2287] = True
    #mask[2760:2808] = False
###########

# ##########L1
    #mask[0:100] = True
    #mask[1475:1478] = True
    #mask[1720:1750] = True
    #mask[1940:1980] = True
    #mask[2856:2910] = True
    #mask[2030:2050] = True
    #mask[2160:2185] = True
    #mask[2700:2720] = True
    #mask[2660:2685] = False
############

##########L2
    #mask[0:10] = True
    #mask[1104:1107] = True
    #mask[1265:1270] = True
    #mask[1401:1403] = True
    #mask[1896:1898] = True
    #mask[2670:2672] = True
    #mask[2240:2253] = True
    #mask[2855:2870] = False
############


    masked = mask.sum()
    print("{} columns masked".format(masked))


    fig, ax1 = plt.subplots(figsize=(10, 5.4))
    ax1.set_xlabel("Column number")
    ax1.set_ylim(-10,100)
    #ax1.plot(med1.data, "r+", label=med1.name, markersize=1)
    ax1.plot(med2.data, "r+", label=med2.name, markersize=1)
    
    ylab = med2.name + " (e)"
    ax1.set_ylabel(ylab)
    ax1.legend(loc=2)

    ax2 = ax1.twinx()
    ax3 = ax1.twinx()

    ax2.imshow(np.expand_dims(mask, axis=0), aspect="auto", cmap='binary', alpha=0.4 )
    ax2.get_yaxis().set_visible(False)
    ax3.plot(mad.data, "b+", label=mad.name, markersize=1 )
    ax3.set_ylim(-0.05,0.3)
    ylab = mad.name + " (e)"
    ax3.set_ylabel(ylab)
    plt.title(" {}{}  RUNID: {} - {}\n {} columns masked".format(id.amp, id.ccd,
                                min(id.data), max(id.data), masked))
    plt.legend()
    if savemask:
        with open('{}{}_mask_RUNID{}-{}.npy'.format(id.amp, id.ccd, min(id.data),max(id.data)), 'wb') as f:
            np.save(f, mask)
        plt.savefig('{}{}_mask_RUNID{}-{}.png'.format(id.amp, id.ccd, min(id.data),max(id.data)))


    plt.show()
    return mask

###Mask clusters -- masks all pixels above threshold and all leading pixels
###within radius and all trailing pixels within radius + cti
def mask_clusters(data, threshold, radius, cti):
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
                        mask[:(i+radius+cti),:(j+radius+cti)] = True  #mask radius + cti above and to the right of pixel
                    elif((rows - j) <= radius):  #pixel column is within radius from the right edge
                        mask[:(i+radius+cti),(j-radius):] = True  #mask radius + cti above pixel row and radius to the left of edge
                    else:
                        mask[:(i+radius+cti),(j-radius):(j+radius+cti)] = True #mask radius + cti to right of left edge and between radius leading and radius+ cti trailing of row
                elif((cols - i) <= radius):  #pix col is within radius of right edge
                    if(j < radius):  #pix row within radius of bottom
                        mask[(i-radius):,:(j+radius+cti)] = True #mask radius away from right edge and radius + cti from bottom
                    elif((rows - j) <= radius): #pix within radius of top
                        mask[(i-radius):,(j-radius):] = True  #mask radius to left of pix and radius below
                    else:
                        mask[(i-radius):,(j-radius):(j+radius+cti)] = True #mask radius to left and radius below and radius + cti above
                else:
                    if(j < radius): #pixel is within radius of bottom
                        mask[(i-radius):(i+radius+cti),:(j+radius+cti)] = True  #mask radius to left and radius +cti to right and radius + cti above
                    elif((rows - j) <= radius): #pix within radius of top
                        mask[(i-radius):(i+radius+cti),(j-radius):] = True
                    else:
                        mask[(i-radius):(i+radius+cti),(j-radius):(j+radius+cti)] = True
    return mask

###Fit masked pixels, with clusters masked, to poisson-gauss distribution
def fit_masked(mask, stacked, cmask, id):

    ###Expand mask to 2D
    expand_mask = np.tile(mask,stacked.shape[1]).reshape(stacked.T.shape)

    ###Apply mask to pixel array
    pix_vals_masked = np.ma.masked_array(stacked.T, mask=expand_mask)

    ###Mask clusters found from mask_clusters
    clusters_masked = np.ma.masked_array(pix_vals_masked, mask=cmask.T)
    print("pixels masked as clusters = {}".format(cmask.sum()))

    masked = CCDData("masked spectrum ", pix_vals_masked.flatten(),
                        np.median(pix_vals_masked), np.std(pix_vals_masked))
    cmasked = CCDData("masked spectrum ", clusters_masked.flatten(),
                        np.median(clusters_masked), np.std(clusters_masked))



    ###Fit masked pixels to poiss-gauss
    ROOThist(cmasked, bin_sz=0.01,lam=0.001,norm=cmasked.data.size,mean=0,noise=0.16,id=id)
   
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
                              min_col=1500, max_col=3000, runid=id)
    #saveROOT(median, MAD, id)

    #plot(dc, id)
    #plot(noise, id)
    #plot(cal, id)

    #ROOThist(pix_vals, bin_sz=0.1,lam=0.1,norm=pix_vals.data.size,noise=0.16,mean=0)
    cmask = mask_clusters(stacked, threshold=20, radius=1, cti=50)

    #MAD_std, MAD_med = ROOThist(MAD, 0.001, 0.1, norm=MAD.data.size,noise=0.16,mean=0)
    #med_std, med_med = ROOThist(med_over_sum, 0.001, 0, norm=med_over_sum.data.size,noise=0.16,mean=0)
    #med_sum_std, med_sum_med = ROOThist(med_over_sum, 0.1, 0, norm=median.data.size,noise=0.16,mean=0)

    mask = make_mask(MAD, median, med_over_sum, strength=6, radius=5, tail = 0, id=id, savemask=False)

    L2 = "L2_mask_RUNID160-210.npy"
    L1 = "L1_mask_RUNID160-210.npy"
    #U2 = "mask_RUN_ID_040-047/U2/U2_mask_1510cols.npy"
    U2 = "mask_RUNID_133-139/U2_mask_RUNID133-139.npy"
    U1 = "U1_mask_RUNID133-158.npy"
    #mask = np.load(L1)

    img_masked = fit_masked(mask, stacked, cmask, id)
    #plot_img(img_masked)


if __name__ == '__main__':
    main()
