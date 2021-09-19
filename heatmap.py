######################################################
##Author:  Kellie McGuire     kellie@kelliejensen.com
##
##Takes FITS files or .txt file and creates a heatmap
##of the pixel values
######################################################


import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import sys


def fits_to_txt(fitsfile):
    hdulist = fits.open(fitsfile)
    hdulist.info()
    data = hdulist[0].data
    hdulist.close()
    return(data)


def heatmap(data,equalize=False):
    print(data.shape)
    height, width = data.shape

    if equalize:
        Umed=np.nanmedian(data[0:-1,int(width/2):width])
        Lmed=np.nanmedian(data[0:-1,0:int(width/2)])
        diff=Umed-Lmed
        ###Set U and L to have same baseline
        subtract = np.zeros((height,int(width/2)))
        difference = np.full((height,int(width/2)),diff)
        subtract = np.append(subtract, difference, axis=1)
        data = data-subtract


    #print("pix array: {}".format(np.nanmedian(data[500:4000,2964:3082])))
    #print("overscan: {}".format(np.nanmedian(data[500:4000,3082:3100])))
    #print("overscan: {}".format(np.nanmedian(data[500:4000,3100:3118])))
    #print("pix array: {}".format(np.nanmedian(data[500:4000,3118:3218])))



    plt.rcParams.update({'font.size': 15})
    plt.imshow(data, vmin=np.nanmean(data)-np.nanstd(data), vmax=np.nanmean(data)+np.nanstd(data), cmap="viridis")
    #plt.imshow(data, cmap="viridis")
    plt.title("UW6416D  20-min exposures (median)")
    plt.colorbar(label="ADU")
    plt.ylim(0,height)
    plt.ylabel("Row")
    plt.xlabel("Column")
    plt.show()


if __name__ == '__main__':
    if sys.argv[1].endswith('.fits'):
        input_file = fits_to_txt(sys.argv[1])
    elif sys.argv[1].endswith('.txt'):
        input_file = np.genfromtxt(sys.argv[1], delimiter=',')
    else:
        print("Enter a valid FITS or txt file")
        exit(1)

    heatmap(input_file,equalize=False)
