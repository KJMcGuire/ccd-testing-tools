# ccd-testing-tools
Tools for CCD analysis. 
## CTI.py   
Takes as input FITS files. Applies binning along cols(rows) and finds the mean pixel value within a specified ADU window by fitting pixel vals to a gaussian. Fits the resulting means to a linear function versus number of cols(rows) to determine the horiz(vert) CTE of the CCD. Also creates a 2D histogram of the pixel vals collapsed along the horiz(vert) axis.
##median_imgs.py
Takes as input FITS files. Computes the median value for each pixel across all files before and after eliminating outliers. Saves resulting median images to .txt files.
##heatmap.py
Takes as input FITS file or .txt file. Creates a heatmap of the pixel values.
##CompositeCluster.C
Takes as input .root files generated from the cluster finder method of the [pysimdamicm software](https://ncastell.web.cern.ch/ncastell/pysimdamicm/) and generates a composite cluster within a specified energy range.
