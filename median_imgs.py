#!/usr/bin/env python3

######################################################
##Author:  Kellie McGuire     kellie@kelliejensen.com
##
##Takes FITS files and computes median value for each
##pixel and median value after eliminating outliers.
##Saves results to root histograms.
######################################################


import numpy as np
from astropy.io import fits
import argparse
import sys
import matplotlib.pyplot as plt
import ROOT
import time
import datetime
import numpy.ma as ma


###Create array of the data from FITS file
def get_data(fitsfiles,ext=0):
    for i in range(len(fitsfiles)):
        hdulist = fits.open(fitsfiles[i])
        hdulist.info()
        fitsfiles[i] = hdulist[ext].data
        #if i == len(fitsfiles):
        #    hdulist.close()
    fitsfiles = np.array(fitsfiles)
    hdulist.close()
    return fitsfiles #3d array

###Flattens 3d array into 2d
def flatten_files(fitsfiles):
    n_files, width, height = fitsfiles.shape
    files = np.zeros((n_files, width*height))
    for n in range(n_files):
        files[n] = fitsfiles[n].flatten()
    return files #2d array


#########################################################
###Calculates the median and stddev of each pixel over
###all files. Creates arrays populated with median
###and median after omitting outliers and saves to .txt
#########################################################
def median_imgs(flatfiles, outfile, width, height):
    print(flatfiles.shape)
    n_files, file_length = flatfiles.shape

    median_img = np.median(flatfiles, axis=0)

    stddev = np.std(flatfiles,axis=0)
    masked = np.ma.masked_where(np.logical_or(flatfiles<median_img-stddev,flatfiles>median_img+stddev),flatfiles,copy=False)
    median_sans_outliers = np.ma.median(masked, axis=0)

    median_img = median_img.reshape(width, height)
    median_sans_outliers = median_sans_outliers.reshape(width, height)

    title = outfile+".root"
    f = ROOT.TFile(title, "RECREATE")
    h = ROOT.TH2D("med","median img", height, 0.5, height+0.5, width, 0.5, width+0.5)
    h2 = ROOT.TH2D("sansout","median img sans outliers", height, 0.5, height+0.5, width, 0.5, width+0.5)

    for i in range(height):
        for j in range(width):
            h.Fill(i+1, j+1, median_img[j,i])
            h2.Fill(i+1, j+1, median_sans_outliers[j,i])

    h.Write()
    h2.Write()
    f.Close()
    #np.savetxt('median_img.txt', median_img, delimiter=',')
    #np.savetxt('median_sans_outliers.txt', median_sans_outliers, delimiter=',')
    return




def main():

    parser = argparse.ArgumentParser(description='Generate median CCD img')
    parser.add_argument('-f','--files', metavar='FILES', type=str, help='FITS files',required=True, nargs='*')
    parser.add_argument('-e','--ext', metavar='Fits Extention', type=int, default=0, help='Extension of FITS file')
    args = parser.parse_args()


    start_time = time.time()

    data = get_data(args.files, args.ext)
    flatfiles = flatten_files(data)
    n_files, width, height = data.shape
    median_imgs(flatfiles,"20220702_median",width,height)


    print(f"time: {datetime.timedelta(seconds=int(time.time() - start_time))}")





if __name__ == '__main__':
    main()
    import doctest
    #doctest.testmod()
