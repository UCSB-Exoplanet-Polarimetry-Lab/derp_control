'''
bad_pix_module.py

This module contains functions for testing the efficacy of bad pixel maps for the CRED2 at various exposure times.
Functions are intended for use ith the jupyter notebook bad_pix_analysis
Functions in this module rely on functions from derpy as a pip insalled module
'''

################################## imports ##################################
import derpy as dp
import numpy as np
import sys
sys.path.append("C:\Program Files\FirstLightImaging\FliSdk\Python\demo")
import FliSdk_V2 as sdk
#############################################################################

def collect_data(exp_times, num_ims, output_dir, cam):
    '''
    This function takes a data set comprised of bad pixel corrected and non corrected dark frames, then sends them to the desginated output directory
    in:
        exp_times: list of exposure times the camera will use 
        num_ims: number of images camera will take at each exposure for averaging
        outpu_dir: folder where fits files will be stored
        cam: object created in derpy module 'camera.py'; this is the CRED2
    out:
        .fits files to output_dir
        or
        Error message 
    '''

    # loop through exposure times 
    for time in exp_times:

        ## initialize a name for each fits file
        name_corr = 'Corrected_' + str(time) + '_' + str(num_ims)
        name_uncorr = 'Uncorrected_' + str(time) + '_' + str(num_ims)

        ## call 'take_median_image' from derpy to tkae images for each condition

        # turning off correction and taking uncorrected images
        sdk.FliCredTwo.EnableBadPixel(cam.context, False)
        cam.take_median_image(10)