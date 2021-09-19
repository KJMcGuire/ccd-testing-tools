from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.optimize import curve_fit
from scipy import stats
import os.path
from os import path
import sys


###########################################
###Create arrays of the data from FITS file
###########################################
def get_data(fitsfiles):
    for i in range(len(fitsfiles)):
        hdulist = fits.open(fitsfiles[i])
        hdulist.info()
        fitsfiles[i] = hdulist[0].data
    data = np.array(fitsfiles)
    print(data.shape)
    hdulist.close()
    return data ##3d array of all fits data

#########################################################
###Reshape 3D data array into 2D for horizontal CTI calcs
#########################################################
def horiz_reshape(data):

    n_files, rows, cols = data.shape
    horiz = np.arange(0,cols,1)

    ###Stack images vertically; use for horiz. transfer
    h_data = data.reshape(rows*n_files, cols)
    h_data = h_data.T

    #h_data = h_data[3100:-1,0:-1] #To use only some of the columns

    ###Create 1D array of rows*n_files zeros, ones, twos, ...
    col_n = np.repeat(horiz, len(h_data[0,:]))

    return h_data,col_n

#########################################################
###Reshape 3D data array into 2D for vertical CTI calcs
#########################################################
def vert_reshape(data):
    n_files, rows, cols = data.shape
    vert = np.arange(0,rows,1)

    data = data[0:-1,0:-1,1000:2000]  #Select slice; dim: [img#,rows,cols]

    ###Stack images horizontally; use for vert. transfer
    v_data = np.array([np.concatenate((q)) for q in zip(*[data[a] for a in range(len(data))])])

    ###Create 1D array of rows*n_files zeros, ones, twos, ...
    row_n = np.repeat(vert, len(v_data[0,:]))

    return v_data,row_n


######################################################
###Fit binned pixel distrubution within specified window
###to gaussian; return means from the fit results
######################################################
def get_means(data_arr,bin_sz=100,window=(2000,4000),peak=3000,gain=10,plot=True):

    bins = len(data_arr) // bin_sz
    means = np.empty([0])

    ADU_max = window[1]  ##Max ADU value to define window containing x-ray events
    ADU_min = window[0]  ##Min ADU value to define window containing x-ray events

    ###Define function to be used to fit to the data
    def gauss(x, *p):
        A, mu, sigma = p
        return A*np.exp(-(x-mu)**2/(2.*sigma**2))

    for i in range(bins):
        ###Create empty array for holding selected values
        filter_pix = np.empty([0],dtype=int)
        rng=(i*bin_sz,(i*bin_sz)+bin_sz)
        ###Iterate over all pixels in bin and select those that fall within window
        for element in data_arr[rng[0]:rng[1],:].flatten():
            if element < ADU_max:
                if element > ADU_min:
                    filter_pix = np.append(filter_pix, element)

        gauss_bins=int(((ADU_max-ADU_min)/gain)) ##Number of bins for fitting to gauss; chosen so each bin corresponds to roughly one electron
        hist, bin_edges = np.histogram(filter_pix,bins=gauss_bins)
        print(bin_edges)
        bin_centers = (bin_edges[:-1] + bin_edges[1:])/2

        p0 = [200.,peak,100.]
        coeff, var_matrix = curve_fit(gauss,bin_centers,hist, p0=p0)
        perr = np.sqrt(np.diag(var_matrix))
        means = np.append(means,(coeff[1],perr[1]))


        if plot:
            n,bins,patches = plt.hist(data_arr[rng[0]:rng[1],:].flatten(),bins=int(4000/gain), range=(3000,7000))
            gauss_fit = gauss(bins, *coeff)
            plt.plot(bins,gauss_fit, color='red')
            #plt.xlim(1,15000)
            plt.title("Pixel distribution")
            plt.xlabel("Pix val (ADU)")
            plt.ylabel("Count")
            plt.show()


    return means.reshape((-1,2))

###################################################
###Plot raw spectrum
###################################################
def spectrum(raw_data):

        plt.hist(raw_data.flatten(),bins=15000, range=(1,15000))
        plt.show()



###################################################
###Plot 2D histogram of data
###################################################
def plot2Dhist(data, transfers):

    plt.hist2d(transfers,data.flatten(),bins=[31,2000], range=[[100,3000],[1,15000]], cmax = 15000)
    #plt.hist2d(transfers,data.flatten(),bins=[31,2000])

    #plt.imshow(data, vmax=19000, cmap="viridis")
    plt.colorbar()
    plt.title("UW6415D Horizontal CTE")
    plt.xlabel("Transfers")
    plt.ylabel("Pix val (ADU)")
    #plt.legend()
    plt.show()


###################################################
###Plot linear fit to mean vals
###################################################

def fit_means(means,data,row_n,start=1,stop=-1):

    ###Average number of transfers for each mean pix value
    transfers = np.linspace(0,len(data),len(means[:,0]))

    m,b,r,p,stderr = stats.linregress(transfers[start:stop],means[start:stop,0])
    CTE = 1 - m/b
    CTE_err = stderr/b
    print(CTE_err)
    plt.figure(figsize=(9,3))
    plt.errorbar(transfers[start:stop],means[start:stop,0],yerr=means[start:stop,1],marker='.',linewidth=0, elinewidth=1)
    plt.plot(transfers, m*transfers + b, color='gray', linestyle='--',label='CTE: {:.7f} +- {:.7f}'.format(CTE,CTE_err))
    plt.title("UW6414D Vert. CTE")
    plt.xlabel("Transfers")
    plt.ylabel("Pix val (ADU)")
    plt.legend()
    plt.show()




if __name__ == '__main__':
    if len(sys.argv)<2:

        print("Error: first argument must be a FITS file")
        exit(1)

    if path.exists(sys.argv[1]):
        if sys.argv[1].endswith('.fits'):
            #fitsfiles = sys.argv[1]
            fitsfiles = glob("*.fits")
            print(fitsfiles)
        else:
            print("Enter a valid FITS file")
            exit(1)

    else:
        print("Enter a valid FITS file")
        exit(1)

    raw_data = get_data(fitsfiles)
    data_to_fit, transfers = horiz_reshape(raw_data)
    #data_to_fit, transfers = vert_reshape(raw_data)

    #spectrum(data_to_fit)
    means = get_means(data_to_fit,bin_sz=200,window=(4700,5300),peak=5000,gain=10,plot=False)

    fit_means(means,data_to_fit,transfers,start=1,stop=-1)

    #plot2Dhist(data_to_fit,transfers)
