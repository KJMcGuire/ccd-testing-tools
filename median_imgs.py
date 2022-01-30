#!/usr/bin/env python3

######################################################
##Author:  Kellie McGuire     kellie@kelliejensen.com
##
##Takes FITS files and computes median value for each
##pixel and median value after eliminating outliers.
##Saves results to .txt files.
######################################################


from glob import glob
import numpy as np
from astropy.io import fits
import os.path
from os import path
import sys
import matplotlib.pyplot as plt
import time
import datetime
import numpy.ma as ma


###Create array of the data from FITS file
def get_data(fitsfiles):
    for i in range(len(fitsfiles)):
        hdulist = fits.open(fitsfiles[i])
        hdulist.info()
        fitsfiles[i] = hdulist[0].data
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
def median_imgs(flatfiles, width, height):
    n_files, file_length = flatfiles.shape

    median_img = np.median(flatfiles, axis=0)
    stddev = np.std(flatfiles,axis=0)
    masked = np.ma.masked_where(np.logical_or(flatfiles<median_img-stddev,flatfiles>median_img+stddev),flatfiles,copy=True)
    median_sans_outliers = np.ma.median(masked, axis=0)

    median_img = median_img.reshape(width, height)
    median_sans_outliers = median_sans_outliers.reshape(width, height)
    
    np.savetxt('median_img.txt', median_img, delimiter=',')
    np.savetxt('median_sans_outliers.txt', median_sans_outliers, delimiter=',')
    return




def main():
    if len(sys.argv)<2:

        print("Error: first argument must be a FITS file")
        exit(1)

    if path.exists(sys.argv[1]):
        if sys.argv[1].endswith('.fits'):
            fitsfiles = glob("*.fits")
            print(fitsfiles)
        else:
            print("Enter a valid FITS file")
            exit(1)

    else:
        print("Enter a valid FITS file")
        exit(1)


    start_time = time.time()


    data = get_data(fitsfiles)
    flatfiles = flatten_files(data)
    n_files, width, height = data.shape

    median_imgs(flatfiles,width,height)


    print(f"time: {datetime.timedelta(seconds=int(time.time() - start_time))}")





if __name__ == '__main__':
    main()
    import doctest
    #doctest.testmod()
