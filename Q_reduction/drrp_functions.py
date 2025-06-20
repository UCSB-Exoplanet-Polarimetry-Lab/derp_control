# Functions used to run Dual Rotating Retarder Polarimeter (DRRP) measurements and analysis
# This is the new version of DRRP_Functions.py that uses Q measurements instead of I, and some additional features

import numpy as np
from numpy.linalg import inv
from astropy.io import fits
import os
import re
from scipy.optimize import curve_fit
import glob
import math
import json

# Mueller matrix for a linear polarizer, with angle a between transmission axis and horizontal (radians)
def linear_polarizer(a):
    """
    Mueller matrix for a linear polarizer
    Inputs:
    a: angle of the transmission axis, in raidans.
    """
    M01 = np.cos(2*a)
    M02 = np.sin(2*a)
    M10 = np.cos(2*a)
    M11 = np.cos(2*a)**2
    M12 = np.cos(2*a)*np.sin(2*a)
    M20 = np.sin(2*a)
    M21 = np.cos(2*a)*np.sin(2*a)
    M22 = np.sin(2*a)**2

    return 0.5*np.array([[1, M01, M02, 0], 
                         [M10, M11, M12, 0], 
                         [M20, M21, M22, 0], 
                         [0, 0, 0, 0]])


def linear_retarder(a, r):
    """
    Mueller matrix for a linear retarder (waveplate)
    Inputs:
    a: value for the angle of the fast axis, in radians. 
    r: value for the retardance/phase shift, in radians. 
    """
    M11 = np.cos(2*a)**2 + np.cos(r)*np.sin(2*a)**2
    M12 = np.cos(2*a)*np.sin(2*a)*(1-np.cos(r))
    M13 = -np.sin(2*a)*np.sin(r)
    M21 = M12
    M22 = np.sin(2*a)**2 + np.cos(2*a)**2*np.cos(r)
    M23 = np.cos(2*a)*np.sin(r)
    M31 = -M13
    M32 = -M23
    M33 = np.cos(r)

    return np.array([[1, 0, 0, 0], 
                     [0, M11, M12, M13], 
                     [0, M21, M22, M23], 
                     [0, M31, M32, M33]])


def extract_number(filename):
    """
    Sorting function to organize filenames based on the last number in the filename. 
    For DRRP data, the last number should be the angle of rotation. 
    """
    match = re.findall(r'\d+(?:\.\d+)?', filename)
    if match:
        return float(match[-1])


def dark_subtraction(image_file, dark_file, old_directory, new_directory):
    """
    Subtract a dark frame from a raw image in fits file format. 
    The new "reduced" images are saved to a different folder. 
    Inputs:
    image_file: string containing the first part of the file name, such as "Data_1100nm_"
    dark_file: full file name of the dark frame
    old_directory: file path to the folder containing the image file
    new_directory: file path to a folder where the dark subtracted image will be saved. 
    """
    # Open the dark image and extract pixel values
    fits.open(dark_file)
    dark = fits.getdata(dark_file)
    dark_median = np.median(dark, axis=0)

    # Search through the desired raw data folder
    for filename in os.listdir(old_directory):
        if filename.startswith(image_file):                                # Call specific files starting with the desired name
            with fits.open(os.path.join(old_directory, filename)) as hdul:
                img_data = hdul[0].data
                img_median = np.median(img_data, axis=0)
                reduced_data = img_median - dark_median

            # Save the newly reduced image to a reduced data folder
            new_filename = f"Reduced_{filename}"
            new_filepath = os.path.join(new_directory, new_filename)
            fits.writeto(new_filepath, reduced_data, overwrite=True)


def dark_subtract(wavelengths, dark_file_path, foldername_base):
    """
    Function to perform dark subtraction of fits file images. Saves the reduced images to a new folder. Can be performed over multiple wavelengths at once!
    
    Parameters:
    wavelengths: Enter wavelengths like [1100, 1850] in nm.
    dark_file_path: Path to the desired dark image file. 
    foldername_base: Path to the base folder where other folders for raw and reduced (dark subtracted) images are stored.
    
    Returns:
    None

    """
    
    print("Reducing images...")
    for idx, wavelength in enumerate(wavelengths):
        try:
            foldername = os.path.join(foldername_base, f"Cal_{wavelength}_Raw\\")
            new_directory = os.path.join(foldername_base, f"Cal_{wavelength}_Reduced\\")
            os.makedirs(new_directory, exist_ok=True)

            # Clear out old files in the directory
            old_files = glob.glob(os.path.join(new_directory, '*'))
            for f in old_files:
                os.remove(f)

            # Open the dark image
            dark = fits.getdata(dark_file_path)
            dark_median = np.median(dark, axis=0)

            # Search through raw data folder
            for file in os.listdir(foldername):
                if file.startswith("DRRP_"):
                    with fits.open(os.path.join(foldername, file)) as hdul:
                        img_data = hdul[0].data
                        reduced_data = img_data - dark_median
                        hdul.close() # see if this makes it faster

                    # Save the reduced image
                    new_filename = f"Reduced_{file}"
                    fits.writeto(os.path.join(new_directory, new_filename), reduced_data, overwrite=True)

                    # Add comments to reduced image header
                    with fits.open(os.path.join(new_directory, new_filename), mode='update') as hdu:
                        header = hdu[0].header
                        header['COMMENT1'] = "Reduced image taken using CRED2 ER performing DRRP measurements."
                        # header['COMMENT2'] = f"Camera temperature: {cam.set_temp}C. Framerate: {cam.fps}fps. Exposure time: {cam.tint}ms."
                        hdu.flush()

            print(f"Images have been reduced for wavelength {wavelength}. Process finished.")
        except Exception as e:
            print(f"Error during reduction for wavelength {wavelength}: {e}")


def find_pixels(img_file, value=16383):
    """
    This function iterates over all the pixels in a fits image and returns the indices of the pixels that match the given value. Useful for checking for saturation. 
    
    Parameters:
    img_file: Path to the fits file.
    value: The value to search for in the img_median. Defaults to 16,383, which is the saturation limit for the CRED-2.
    
    Returns:
    list: A list of indices (y,x) where the value is found.
    """
    saturated_indices = []

    # Works if img_file is a cube of images, not just one frame
    with fits.open(img_file) as hdul:
        img_data = hdul[0].data
        hdul.close()
        img_median = np.median(img_data, axis=0)
        # print(img_median.shape)

        for i in range(img_median.shape[0]):
            for j in range(img_median.shape[1]):
                if img_median[i, j] == value:
                    saturated_indices.append((i, j))

    if len(saturated_indices) == 0:
        print(f"No pixels found in {img_file} with value {value}.")
    else:
        # returns indices in the form (y,x)
        return saturated_indices


def extract_intensities(reduced_filename, reduced_folder, lcenter, rcenter, maxradius, cutoff=5000):
    """
    Get intensity from an image as the sum of pixels within a certain radius on the detector. 
    Inputs:
    reduced_filename: a string indicating the first part of the file name that is the same for each file, like 'Reduced_DRRP_'.
    reduced_folder: a string indicating the folder where these files are located. 
    lcenter and rcenter: the coordinates [y, x] for the location of each beam on the detector. 
    maxradius: an integer number of pixels to define as the radius of the beam.
    Cutof: the value of the sum of pixels in the spot. 
    If the measured value is less than this cutoff threshold, there is likely an error with that image and it will raise a warning.
    
    Outputs:
    I_left and I_right: pixel sum of each spot, the intensity.
    new_angles: the QWP angles with usable data. Angles where I<cutoff are excluded from this list.
    bad_indices: list of the bad angles that were excluded. """

    I_left = np.array([])
    I_right = np.array([])
    bad_indices = np.array([])
    # longtheta = np.linspace(0, np.pi, 46)
    angles = np.array([])

    for filename in sorted(os.listdir(reduced_folder), key = extract_number):
        if filename.startswith(reduced_filename):
            angle = np.radians(extract_number(filename)/5) # get QWP 1 angle in radians
            with fits.open(os.path.join(reduced_folder, filename)) as hdul:
                angles = np.append(angles, angle)
                reduced_img_data = hdul[0].data
                ys, xs, = np.indices(reduced_img_data.shape)
                lradius = np.sqrt((ys-lcenter[0])**2+(xs-lcenter[1])**2)
                rradius = np.sqrt((ys-rcenter[0])**2+(xs-rcenter[1])**2)

                lbackground_mask = (lradius > maxradius+5) & (lradius < maxradius+10)
                rbackground_mask = (rradius > maxradius+5) & (rradius < maxradius+10)   # Index the background around each spot, take the median value

                background_lmedian = np.median(reduced_img_data[lbackground_mask])
                background_rmedian = np.median(reduced_img_data[rbackground_mask])

                lflux = np.sum(reduced_img_data[lradius < maxradius] - background_lmedian)   # Now take the flux with the background mask subtracted
                rflux = np.sum(reduced_img_data[rradius < maxradius] - background_rmedian)
                I_left = np.append(I_left, lflux)
                I_right = np.append(I_right, rflux)

                if lflux+rflux < cutoff:
                    print("Warning: low flux detected, check the image " + filename + ", index: " + str(sorted(os.listdir(reduced_folder), key = extract_number).index(filename)))
                    bad_indices = np.append(bad_indices, sorted(os.listdir(reduced_folder), key = extract_number).index(filename))
                else:
                    continue 

    # Makes the array a list of integers that can be used to index the other array
    bad_indices = bad_indices.astype(int)
    # Deletes the bad indices (caused by camera glitch or some other complication) from the data
    I_left = np.delete(I_left, bad_indices)
    I_right = np.delete(I_right, bad_indices)
    # new_angles = np.delete(longtheta, bad_indices)

    return I_left, I_right, angles, bad_indices


def condition_number(matrix):
    """
    Gives the condition number of eventual Mueller matrix. 
    """
    minv = np.linalg.pinv(matrix)

    # Compute maximum norm
    norm = np.linalg.norm(matrix, ord=np.inf)
    ninv = np.linalg.norm(minv, ord=np.inf)

    return norm*ninv


def q_calibrated_full_mueller_polarimetry(thetas, a1, w1, w2, r1, r2, I_vert, I_hor, M_in=None):
    """
    Full Mueller polarimetry using measurements of Q and calibration parameters. 
    Gives a calibrated Mueller matrix with the parameters, or set a1, w1, w2, r1, and r2 to zero for an uncalibrated matrix.
    Parameters
    ----------
    thetas : array
        angles of the first quarter wave plate
    a1 : float
        calibration parameter for the offset angle of the first linear polarizer
    w1 : float
        calibration parameter for the offset angle of the first quarter-wave plate fast axis.
    w2 : float
        calibration parameter for the offset angle of the second quarter-wave plate fast axis.
    r1 : float
        calibration parameter for the retardance offset of the first quarter-wave plate. 
    r2 : float
        calibration parameter for the retardance offset of the second quarter-wave plate.
    I_hor : array
        measured intensity of the horizontal polarization spot from the Wollaston prism
    I_vert : array
        measured intensity of the vertical polarization spot from the Wollaston prism
    Returns
    -------
    M : array
        4x4 Mueller matrix for the measured sample.
    """
    nmeas = len(thetas)  # Number of measurements
    Wmat1 = np.zeros([nmeas, 16])
    Pmat1 = np.zeros([nmeas])
    Wmat2 = np.zeros([nmeas, 16])
    Pmat2 = np.zeros([nmeas])
    th = thetas

    # Values normalized by q_ultimate_polarimetry
    # values don't need to be normalized here, assuming the input intensities are already normalized by total intensity
    I_total = I_vert + I_hor
    Q = I_hor - I_vert

    for i in range(nmeas):
        # Mueller Matrix of generator (linear polarizer and a quarter wave plate)
        Mg = linear_retarder(th[i]+w1, np.pi/2+r1) @ linear_polarizer(0+a1)

        # Mueller Matrix of analyzer (one channel of the Wollaston prism is treated as a linear polarizer)
        Ma = linear_retarder(th[i]*5+w2, np.pi/2+r2)

        # Data reduction matrix. Taking the 0 index ensures that intensity is the output
        Wmat1[i,:] = np.kron((Ma)[0,:], Mg[:,0]) # for the top row, using intensities
        Wmat2[i,:] = np.kron((Ma)[1,:], Mg[:,0]) # for the bottom 3 rows, using Q

        # M_in is some example Mueller matrix. Providing this input will test theoretical Mueller matrix. Otherwise, the raw data is used
        if M_in is not None:
            Pmat1[i] = (Ma[0,:] @ M_in @ Mg[:,0])
            Pmat2[i] = (Ma[1,:] @ M_in @ Mg[:,0])
        else:
            Pmat1[i] = I_total[i]  #Pmat is a vector of measurements (either I or Q)
            Pmat2[i] = Q[i] 

    # Compute Mueller matrix using Moore-Penrose pseudo invervse
    M1 = np.linalg.pinv(Wmat1) @ Pmat1
    M1 = np.reshape(M1, [4,4])

    M2 = np.linalg.pinv(Wmat2) @ Pmat2
    M2 = np.reshape(M2, [4,4])

    M = np.zeros([4,4])
    M[0,:] = M1[0,:]
    M[1:4,:] = M2[1:4,:]

    return M


# Define the identity matrix and other matrices which are useful for the Mueller calculus
M_identity = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
A = np.array([1, 0, 0, 0])
B = np.array([[1], [0], [0], [0]])
C = np.array([0, 1, 0, 0])


# Calculate the root-mean-square error of the calibration matrix by comparing with the identity matrix
def RMS_calculator(calibration_matrix):
    """Calculates the root-mean-square error of a calibration matrix by comparing with the identity matrix.

    Parameters
    ----------
    calibration_matrix : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    M_identity = np.eye(4) # 4x4 identity matrix
    differences = []

    for i in range(0, 4):
        for j in range(0, 4):
            differences.append(calibration_matrix[i, j]-M_identity[i, j])

    differences_squared = [x**2 for x in differences]
    RMS = np.sqrt(sum(differences_squared)/16)
    return RMS


# Calculate the retardance error by standard error propogation using RMS in the matrix elements from calibration
def propagated_error(M_R, RMS):
    """Propagates error in the Mueller matrix to error in the retardance. 

    Parameters
    ----------
    M_R : 4x4 array for the Mueller matrix of a retarder
    RMS : float. Root-mean-square error of the calibration matrix

    Returns
    -------
    float
        RMS error in the retardance value in radians. 
    """
    # return RMS/np.sqrt(1-(np.trace(M_R)/2-1)**2) # These two equations are equivalent
    x = np.trace(M_R)
    return 2*RMS/np.sqrt(4*x-x**2) # Value in radians



# Functions from Jaren's katsu code on Polar decomposition. Inspired by Lu & Chipman 1996 https://doi.org/10.1364/JOSAA.13.001106

def broadcast_outer(a,b):
    """
    Broadcasted outer product of two A,B,...,N vectors. Used for polarimetric data reduction
    where out is a A,B,...,N,N matrix. While in principle this does not require vectors of different length, it is not tested
    to produce anything other than square matrices.
    Parameters
    ----------
    a : numpy.ndarray
        A,B,...,N vector 1
    b : numpy.ndarray
        A,B,...,N vector 2
    Returns
    -------
    numpy.ndarray
        outer product matrix
    """
    return np.einsum('...i,...j->...ij',a,b)


def _empty_mueller(shape):
    """
    Returns an empty array to populate with Mueller matrix elements.

    Parameters
    ----------
    shape : list
        shape to prepend to the mueller matrix array. shape = [32,32] returns
        an array of shape [32,32,4,4] where the matrix is assumed to be in the
        last indices. Defaults to None, which returns a 4x4 array.

    Returns
    -------
    numpy.ndarray
        The zero array of specified shape

    Notes
    -----
    The structure of this function was taken from prysm.x.polarization,
    which was written by Jaren Ashcraft
    """
    if shape is None:
        shape = (4, 4)

    else:
        shape = (*shape, 4, 4)

    return np.zeros(shape)


def decompose_diattenuator(M, normalize=False):
    """
    Decompose M into a diattenuator using the Polar decomposition

    from Lu & Chipman 1996 https://doi.org/10.1364/JOSAA.13.001106

    Parameters
    ----------
    M : numpy.ndarray
        Mueller Matrix to decompose

    Returns
    -------
    numpy.ndarray
        Diattenuator component of mueller matrix
    """

    # First, determine the diattenuator
    T = M[..., 0, 0]

    if np.__name__ == "jax.numpy":
        if M.ndim > 2:
            diattenuation_vector = M[..., 0, 1:] / (T.at[..., np.newaxis])
        else:
            diattenuation_vector = M[..., 0, 1:] / (T)

        # D = np.sqrt(np.sum(np.matmul(diattenuation_vector, diattenuation_vector), axis=-1))
        D = np.sqrt(np.sum(diattenuation_vector * diattenuation_vector, axis=-1))
        mD = np.sqrt(1 - D ** 2)

        if M.ndim > 2:
            diattenutation_norm = diattenuation_vector / (D.at[..., np.newaxis])
        else:
            diattenutation_norm = diattenuation_vector / (D)

        DD = broadcast_outer(diattenutation_norm, diattenutation_norm)

        # create diattenuator
        I = np.identity(3)

        if M.ndim > 2:
            I = np.broadcast_to(I, [*M.shape[:-2], 3, 3])
            mD = mD[..., np.newaxis, np.newaxis] 

        inner_diattenuator = mD * I + (1 - mD) * DD  # Eq. 19 Lu & Chipman

        Md = _empty_mueller(M.shape[:-2])

        # Eq 18 Lu & Chipman
        Md = Md.at[..., 0, 0].set(1.)
        Md = Md.at[..., 0, 1:].set(diattenuation_vector)
        Md = Md.at[..., 1:, 0].set(diattenuation_vector)
        Md = Md.at[..., 1:, 1:].set(inner_diattenuator)

        if M.ndim > 2:
            Md = Md * T[..., np.newaxis, np.newaxis]
        else:
            Md = Md * T

        if normalize:
            return Md/np.max(np.abs(Md))
        else:
            return Md

    else:
        if M.ndim > 2:
            diattenuation_vector = M[..., 0, 1:] / T[..., np.newaxis]
        else:
            diattenuation_vector = M[..., 0, 1:] / T
        
        D = np.sqrt(np.sum(diattenuation_vector * diattenuation_vector, axis=-1))
        mD = np.sqrt(1 - D**2)

        if M.ndim > 2:
            diattenutation_norm = diattenuation_vector / D[..., np.newaxis]
        else:
            diattenutation_norm = diattenuation_vector / D

        DD = broadcast_outer(diattenutation_norm, diattenutation_norm)

        # create diattenuator
        I = np.identity(3)

        if M.ndim > 2:
            I = np.broadcast_to(I, [*M.shape[:-2], 3, 3])
            mD = mD[..., np.newaxis, np.newaxis]

        inner_diattenuator = mD * I + (1 - mD) * DD  # Eq. 19 Lu & Chipman

        Md = _empty_mueller(M.shape[:-2])

        # Eq 18 Lu & Chipman
        Md[..., 0, 0] = 1.
        Md[..., 0, 1:] = diattenuation_vector
        Md[..., 1:, 0] = diattenuation_vector
        Md[..., 1:, 1:] = inner_diattenuator

        if M.ndim > 2:
            Md = Md * T[..., np.newaxis, np.newaxis]
        else:
            Md = Md * T

        if normalize:
            return Md/np.max(np.abs(Md))
        else:
            return Md


def decompose_retarder(M, return_all=False, normalize=False):
    """
    Decompose M into a retarder using the Polar decomposition

    from Lu & Chipman 1996 https://doi.org/10.1364/JOSAA.13.001106

    Note: this doesn't work if the diattenuation can be described by a pure
    polarizer, because the matrix is singular and therefore non-invertible

    Parameters
    ----------
    M : numpy.ndarray
        Mueller Matrix to decompose
    return_all : bool
        Whether to return the retarder and diattenuator vs just the retarder.
        Defaults to False, which returns both

    Returns 
    -------
    numpy.ndarray
        Retarder component of mueller matrix
    """
    if normalize:
        Md = decompose_diattenuator(M, normalize=True)
    else:
        Md = decompose_diattenuator(M)

    # Then, derive the retarder
    Mr = M @ np.linalg.inv(Md)

    if normalize:
        Mr = Mr/np.max(np.abs(Mr)) 
    else:
        Mr = Mr

    if return_all:
        return Mr, Md
    else:
        return Mr
    

def q_output_simulation_function(t, a1, w1, w2, r1, r2, M_in=None):
    """
    Function that models the Mueller calculus for the DRRP system and is used to calculate the calibration parameters.
    Parameters
    ----------
    t : array
        angles of the first quarter wave plate
    a1 : float
        calibration parameter for the offset angle of the first linear polarizer
    w1 : float
        calibration parameter for the offset angle of the first quarter-wave plate fast axis.
    w2 : float
        calibration parameter for the offset angle of the second quarter-wave plate fast axis.
    r1 : float
        calibration parameter for the retardance offset of the first quarter-wave plate. 
    r2 : float
        calibration parameter for the retardance offset of the second quarter-wave plate.
    M_in : array
        optional 4x4 Mueller matrix to simulate data. By default None, which uses the identity matrix for air. 
    Returns
    -------
    prediction : array
        An array of predictions for measured Q values.
    """
    if M_in is None:
        M = M_identity
    else:
        M = M_in

    prediction = [None]*len(t)
    for i in range(len(t)):
        prediction[i] = float(C @ linear_retarder(5*t[i]+w2, np.pi/2+r2) @ M @ linear_retarder(t[i]+w1, np.pi/2+r1) @ linear_polarizer(a1) @ B)
    return prediction


# Function that is useful for generating intensity values for a given sample matrix and offset parameters
def I_total_output_simulation_function(t, a1, w1, w2, r1, r2, M_in=None):
    """
    Function to generate TOTAL intensity values measured with a given Mueller matrix and offset parameters.
    Parameters
    ----------
    t : array
        angles of the first quarter wave plate
    a1 : float
        calibration parameter for the offset angle of the first linear polarizer
    w1 : float
        calibration parameter for the offset angle of the first quarter-wave plate fast axis.
    w2 : float
        calibration parameter for the offset angle of the second quarter-wave plate fast axis.
    r1 : float
        calibration parameter for the retardance offset of the first quarter-wave plate. 
    r2 : float
        calibration parameter for the retardance offset of the second quarter-wave plate.
    M_in : array
        optional 4x4 Mueller matrix to simulate data. By default None, which uses the identity matrix for air. 
    Returns
    -------
    prediction : array
        An array of predictions for measured Q values.
    """
    if M_in is None:
        M = M_identity
    else:
        M = M_in

    prediction = [None]*len(t)
    for i in range(len(t)):
        prediction[i] = float(A  @ linear_retarder(5*t[i]+w2, np.pi/2+r2) @ M @ linear_retarder(t[i]+w1, np.pi/2+r1) @ linear_polarizer(a1) @ B)
    return prediction


# Basically the same as above, but with an optional input matrix to simulate data
def single_output_simulation_function(t, a1, a2, w1, w2, r1, r2, LPA_angle=0, M_in=None):
    """
    Function to generate intensity values for one polarization at a time. Default is horizontal, with LPA=0. For vertical, set LPA=pi/2.
    Parameters
    ----------
    t : array
        angles of the first quarter wave plate
    a1 : float
        calibration parameter for the offset angle of the first linear polarizer
    a2 : float
        calibration parameter for the offset angle of the second linear polarizer (could be just one channel of the Wollaston prism)
    w1 : float
        calibration parameter for the offset angle of the first quarter-wave plate fast axis.
    w2 : float
        calibration parameter for the offset angle of the second quarter-wave plate fast axis.
    r1 : float
        calibration parameter for the retardance offset of the first quarter-wave plate. 
    r2 : float
        calibration parameter for the retardance offset of the second quarter-wave plate.
    LPA_angle : float
        angle of the analyzing linear polarizer. Default is 0 for horizontal. Set to pi/2 for vertical.
    M_in : array
        optional 4x4 Mueller matrix to simulate data. By default None, which uses the identity matrix for air. 
    Returns
    -------
    prediction : array
        An array of predictions for measured Q values.
    """
    if M_in is None:
        M = M_identity
    else:
        M = M_in

    prediction = [None]*len(t)
    for i in range(len(t)):
        prediction[i] = float(A @ linear_polarizer(LPA_angle+a2) @ linear_retarder(5*t[i]+w2, np.pi/2+r2) @ M @ linear_retarder(t[i]+w1, np.pi/2+r1) @ linear_polarizer(a1) @ B)
    return prediction


def q_ultimate_polarimetry(cal_angles, cal_vert_intensity, cal_hor_intensity, sample_angles, sample_vert_intensity, sample_hor_intensity):
    """
    Function that calculates the Mueller matrix of a sample and other relevant information.
    cal_angles and sample_angles could be the same, or could be different.
    Parameters
    ----------
    cal_angles : array
        angles of the first quarter wave plate for calibration
    cal_vert_intensity : array
        measured intensity of the vertical polarization spot from the Wollaston prism for calibration
    cal_hor_intensity : array
        measured intensity of the horizontal polarization spot from the Wollaston prism for calibration
    sample_angles : array
        angles of the first quarter wave plate when taking data with the sample
    sample_vert_intensity : array
        measured intensity of the vertical polarization spot from the Wollaston prism when taking data with the sample
    sample_hor_intensity : array
        measured intensity of the horizontal polarization spot from the Wollaston prism when taking data with the sample
    Returns
    -------
    M_Sample : array
        4x4 Mueller matrix for the sample
    retardance : float
        extracted retardance of the sample in waves
    M_Cal : array
        4x4 Mueller matrix for the calibration (should resemble the identity matrix)
    RMS_Error : float
        root mean square error of the calibration matrix
    Retardance_Error : float
        error of the retardance value, assuming the RMS error from the calibration matrix is the same for all elements of the sample matrix.
    """
    ICal = cal_hor_intensity + cal_vert_intensity
    QCal = cal_hor_intensity - cal_vert_intensity 
    cal_vert_intensity_normalized = cal_vert_intensity/ICal
    cal_hor_intensity_normalized = cal_hor_intensity/ICal
    initial_guess = [0, 0, 0, 0, 0]
    parameter_bounds = ([-np.pi, -np.pi, -np.pi, -np.pi/2, -np.pi/2], [np.pi, np.pi, np.pi, np.pi/2, np.pi/2])

    # Find parameters from calibration 
    normalized_QCal = QCal/(2*ICal) # This should correctly normalize from 0 to 0.5 for each angle (first linear polarizer reduces input intensity by half)
    popt, pcov = curve_fit(q_output_simulation_function, cal_angles, normalized_QCal, p0=initial_guess, bounds=parameter_bounds)
    # print(popt, "Fit parameters for a1, w1, w2, r1, and r2. 1 for generator, 2 for analyzer")

    # The calibration matrix (should be close to identity) to see how well the parameters compensate
    MCal = q_calibrated_full_mueller_polarimetry(cal_angles, popt[0], popt[1], popt[2], popt[3], popt[4], cal_vert_intensity_normalized, cal_hor_intensity_normalized)
    MCal = MCal/MCal[0,0] # Normalize by the first element
    # MCal = MCal/np.max(np.abs(MCal))
    RMS_Error = RMS_calculator(MCal)

    # Normalize the sample intensities
    sample_I_total = sample_hor_intensity + sample_vert_intensity
    sample_vert_intensity_normalized = sample_vert_intensity/sample_I_total
    sample_hor_intensity_normalized = sample_hor_intensity/sample_I_total

    # Use the parameters found above from curve fitting to construct the actual Mueller matrix of the sample
    MSample = q_calibrated_full_mueller_polarimetry(sample_angles, popt[0], popt[1], popt[2], popt[3], popt[4], sample_vert_intensity_normalized, sample_hor_intensity_normalized)
    MSample = MSample/MSample[0,0] # Normalize by the first element
    # MSample = MSample/np.max(np.abs(MSample))

    np.set_printoptions(suppress=True) # Suppresses scientific notation, keeps decimal format

    M_r = decompose_retarder(MSample, normalize=True) # Use the polar decomposition to get retarder matrix
    trace_argument = np.trace(M_r)/2 - 1
    # trace_argument = np.clip(trace_argument, -1, 1) # prevent nan outputs by limiting values to the domain of arccos
    with np.errstate(invalid='ignore'): # Suppress the RuntimeWarning for invalid values in arccos
        retardance = np.arccos(trace_argument)/(2*np.pi)  # Value in waves
    # retardance = np.arccos(np.trace(retarder_decomposed_MSample)/2 - 1)/(2*np.pi) # Value in waves

    Retardance_Error = propagated_error(M_r, RMS_Error)/(2*np.pi) # value in waves

    # if retardance and error can't be calculated with regular method, use another method
    if math.isnan(retardance) or math.isnan(Retardance_Error):
        M_r_normalized = M_r/np.max(np.abs(M_r))
        z = M_r_normalized[3, 3]
        sigmaz = 1-np.abs(MCal[3, 3]) # difference between last element of calibration matrix and identity matrix (1)
        retardance = np.arccos(M_r_normalized[3, 3])/(2*np.pi)
        Retardance_Error = np.sqrt((-1/np.sqrt(1-z**2))**2 * sigmaz**2)/(2*np.pi) # propagate error from arccos(z), convert radians to waves
    
    # retardance and Retardance_Error in waves
    return MSample, retardance, MCal, RMS_Error, Retardance_Error, popt, ICal


def I_calibrated_full_mueller_polarimetry(thetas, a1, a2, w1, w2, r1, r2, I_meas=1, LPA_angle=0, return_condition_number=False, M_in=None):
    """
    Function that takes measurements and calibration parameters to construct a Mueller matrix using I reduction.
    """
    
    nmeas = len(thetas)
    Wmat = np.zeros([nmeas, 16])
    Pmat = np.zeros([nmeas])
    th = thetas

    for i in range(nmeas):
        # Mueller Matrix of generator (linear polarizer and a quarter wave plate)
        Mg = linear_retarder(th[i]+w1, np.pi/2+r1) @ linear_polarizer(0+a1)

        # Mueller Matrix of analyzer (one channel of the Wollaston prism is treated as a linear polarizer. The right spot is horizontal (0) and the left spot is vertical(pi/2))
        Ma = linear_polarizer(LPA_angle+a2) @ linear_retarder(th[i]*5+w2, np.pi/2+r2)

        # Data reduction matrix. Taking the 0 index ensures that intensity is the output
        Wmat[i,:] = np.kron(Ma[0,:], Mg[:,0])

        # M_in is some example Mueller matrix. Providing this input will test theoretical Mueller matrix. Otherwise, the raw data is used
        if M_in is not None:
            Pmat[i] = (Ma[0,:] @ M_in @ Mg[:,0]) * I_meas
        else:
            Pmat[i] = I_meas[i]

    # Compute Mueller matrix using Moore-Penrose pseudo invervse
    M = np.linalg.pinv(Wmat) @ Pmat
    M = np.reshape(M,[4,4])

    if return_condition_number == True:
        return M, condition_number(Wmat)
    else:
        return M
    

def I_vertical_calibration_function(t, a1, a2, w1, w2, r1, r2):
    """
    Simulate the horizontal channel intensity.
    
    Parameters:
      xdata : tuple
          A tuple containing (t, I_meas), where
              t      : array of angles
              I_total_meas : array of measured intensities, summing both vertical and horizontal channels. Should have the same length as t.
      a1, a2, w1, w2, r1, r2 : float
          Calibration parameters.
    
    Returns:
      Array of predicted output intensities. Values range from 0 to 0.5 (at most 1/2 the input intensity, because the first element is a linear polarizer)
    """
    prediction = np.empty_like(t, dtype=float)
    
    for i in range(len(t)):
        prediction[i] = float(
            A @ linear_polarizer(a2+np.pi/2) @ linear_retarder(5 * t[i] + w2, np.pi/2 + r2) @ 
            M_identity @ linear_retarder(t[i] + w1, np.pi/2 + r1) @ linear_polarizer(a1) @ B 
        )
    
    return prediction


def I_horizontal_calibration_function(t, a1, a2, w1, w2, r1, r2):
    """
    Simulate the horizontal channel intensity.
    
    Parameters:
      xdata : tuple
          A tuple containing (t, I_meas), where
              t      : array of angles
              I_total_meas : array of measured intensities, summing both vertical and horizontal channels. Should have the same length as t.
      a1, a2, w1, w2, r1, r2 : float
          Calibration parameters.
    
    Returns:
      Array of predicted output intensities. Values range from 0 to 0.5 (at most 1/2 the input intensity, because the first element is a linear polarizer)
    """
    prediction = np.empty_like(t, dtype=float)
    
    for i in range(len(t)):
        prediction[i] = float(
            A @ linear_polarizer(a2) @ linear_retarder(5 * t[i] + w2, np.pi/2 + r2) @ 
            M_identity @ linear_retarder(t[i] + w1, np.pi/2 + r1) @ linear_polarizer(a1) @ B 
        )
    
    return prediction


def I_ultimate_polarimetry(cal_angles, cal_left_intensity, cal_right_intensity, sample_angles, sample_left_intensity, sample_right_intensity):
    """
    New version!!! Now the calibration function takes intensity values that are normalized by the total intensity at each angle.
    
    Function to take input measurements and construct mueller matrices for the left and right beams using I reduction.
    popt gives best fit parameters for a1, a2, w1, w2, r1, and r2. 1 for generator, 2 for analyzer components. 
    """

    initial_guess = [0, 0, 0, 0, 0, 0]
    parameter_bounds = ([-np.pi, -np.pi, -np.pi, -np.pi, -np.pi/2, -np.pi/2], [np.pi, np.pi, np.pi, np.pi, np.pi/2, np.pi/2])

    Cal_I_total_meas = cal_left_intensity + cal_right_intensity
    cal_lnormalized_intensity = cal_left_intensity/(2*Cal_I_total_meas) # normalize values from 0 to 0.5 to match the output of the simulation function
    cal_rnormalized_intensity = cal_right_intensity/(2*Cal_I_total_meas)

    # Find parameters from calibration of the left spot
    lpopt, lpcov = curve_fit(I_vertical_calibration_function, cal_angles, cal_lnormalized_intensity, p0=initial_guess, bounds=parameter_bounds)

    # Find parameters from calibration of the right spot
    rpopt, rpcov = curve_fit(I_horizontal_calibration_function, cal_angles, cal_rnormalized_intensity, p0=initial_guess, bounds=parameter_bounds)

    # Optional print the calibration matrices (should be close to identity) to see how well the parameters compensate
    MlCal = I_calibrated_full_mueller_polarimetry(cal_angles, lpopt[0], lpopt[1], lpopt[2], lpopt[3], lpopt[4], lpopt[5], cal_lnormalized_intensity, LPA_angle=np.pi/2) # intensities don't need to be normalized here
    MlCal = MlCal/MlCal[0,0] # Normalize by the first element
    # MlCal = MlCal/np.max(np.abs(MlCal))
    MrCal = I_calibrated_full_mueller_polarimetry(cal_angles, rpopt[0], rpopt[1], rpopt[2], rpopt[3], rpopt[4], rpopt[5], cal_rnormalized_intensity)
    MrCal = MrCal/MrCal[0,0] # Normalize by the first element
    # MrCal = MrCal/np.max(np.abs(MrCal))

    # Calculate RMS error of each calibration matrix
    lRMS_Error = RMS_calculator(MlCal)
    rRMS_Error = RMS_calculator(MrCal)

    sample_I_total_meas = sample_left_intensity + sample_right_intensity
    sample_lnormalized_intensity = sample_left_intensity/sample_I_total_meas
    sample_rnormalized_intensity = sample_right_intensity/sample_I_total_meas

    # Use the parameters found above from curve fitting to construct the actual Mueller matrix of the sample for left and right beams (Ml and Mr)
    Ml = I_calibrated_full_mueller_polarimetry(sample_angles, lpopt[0], lpopt[1], lpopt[2], lpopt[3], lpopt[4], lpopt[5], sample_lnormalized_intensity, LPA_angle=np.pi/2) # intensities don't need to be normalized
    Ml = Ml/Ml[0,0] # Normalize by the first element
    # Ml = Ml/np.max(np.abs(Ml))

    Mr = I_calibrated_full_mueller_polarimetry(sample_angles, rpopt[0], rpopt[1], rpopt[2], rpopt[3], rpopt[4], rpopt[5], sample_rnormalized_intensity)
    Mr = Mr/Mr[0,0] # Normalize by the first element
    # Mr = Mr/np.max(np.abs(Mr))

    np.set_printoptions(suppress=True)

    # Use the polar decomposition of the retarder matrix 
    l_retarder_decomposed_MSample = decompose_retarder(Ml, normalize=True)
    r_retarder_decomposed_MSample = decompose_retarder(Mr, normalize=True)

    # retardance = np.arccos(np.trace(normalized_decompose_retarder(r_decomposed_MSample))/2 - 1)/(2*np.pi) # Value in waves
    ltrace_argument = np.trace(l_retarder_decomposed_MSample)/2 - 1
    # ltrace_argument = np.clip(ltrace_argument, -1, 1) # prevent nan outputs by limiting values to the domain of arccos
    rtrace_argument = np.trace(r_retarder_decomposed_MSample)/2 - 1
    # rtrace_argument = np.clip(rtrace_argument, -1, 1)
    with np.errstate(invalid='ignore'): # Suppress the RuntimeWarning for invalid values in arccos
        lretardance = np.arccos(ltrace_argument)/(2*np.pi)  # Value in waves
        rretardance = np.arccos(rtrace_argument)/(2*np.pi)

    lRetardance_Error = propagated_error(l_retarder_decomposed_MSample, lRMS_Error)/(2*np.pi) # value in waves
    rRetardance_Error = propagated_error(r_retarder_decomposed_MSample, rRMS_Error)/(2*np.pi) # value in waves

    # if retardance and error can't be calculated with regular method, use another method (left side)
    if math.isnan(lretardance) or math.isnan(lRetardance_Error):
        lM_r_normalized = l_retarder_decomposed_MSample/np.max(np.abs(l_retarder_decomposed_MSample))
        z = lM_r_normalized[3, 3]
        sigmaz = 1-np.abs(MlCal[3, 3]) # difference between last element of calibration matrix and identity matrix (1)
        lretardance = np.arccos(lM_r_normalized[3, 3])/(2*np.pi)
        lRetardance_Error = np.sqrt((-1/np.sqrt(1-z**2))**2 * sigmaz**2)/(2*np.pi) # propagate error from arccos(z), convert radians to waves

    # same for the right side
    if math.isnan(rretardance) or math.isnan(rRetardance_Error):
        rM_r_normalized = r_retarder_decomposed_MSample/np.max(np.abs(r_retarder_decomposed_MSample))
        z = rM_r_normalized[3, 3]
        sigmaz = 1-np.abs(MrCal[3, 3])
        rretardance = np.arccos(rM_r_normalized[3, 3])/(2*np.pi)
        rRetardance_Error = np.sqrt((-1/np.sqrt(1-z**2))**2 * sigmaz**2)/(2*np.pi)

    return Ml, Mr, lretardance, rretardance, lRetardance_Error, rRetardance_Error, MlCal, MrCal, lRMS_Error, rRMS_Error, lpopt, rpopt, Cal_I_total_meas


def multi_analysis(wavelengths, cal_results, sample_results, method):
    """
    Specify data reduction method as 'I' or 'q'.
    Parameters
    ----------
    wavelengths : list of input wavelengths, like [1400, 1500, 1600]
    cal_results : dictionary with calibration data or path to a JSON file
    sample_results : dictionary with sample data or path to a JSON file
    method : string, 'I' or 'q'. Specifies which method to use for data reduction. 

    Returns 
    -------
    polarimetry_results : A dictionary with the results for each wavelength.
    """
    # Load JSON data if file paths are provided
    if isinstance(cal_results, str):
        with open(cal_results, 'r') as f:
            cal_results = json.load(f)
    
    if isinstance(sample_results, str):
        with open(sample_results, 'r') as f:
            sample_results = json.load(f)

    # Dictionary to store results for each wavelength
    polarimetry_results = {}

    # Loop over the wavelengths
    for wl in wavelengths:
        # Retrieve extracted data (ensure it has been stored in the `results`)
        Cal_theta = np.array(cal_results.get(f'Cal_theta{wl}', []))
        Cal_Il = np.array(cal_results.get(f'Cal_Il_{wl}', []))
        Cal_Ir = np.array(cal_results.get(f'Cal_Ir_{wl}', []))

        theta = np.array(sample_results.get(f'theta{wl}', []))
        Il = np.array(sample_results.get(f'Il_{wl}', []))
        Ir = np.array(sample_results.get(f'Ir_{wl}', []))
        
        # Flags for missing data
        calibration_data_missing = any(len(arr) == 0 for arr in [Cal_theta, Cal_Il, Cal_Ir])
        waveplate_data_missing = any(len(arr) == 0 for arr in [theta, Il, Ir])

        # Check missing data and print appropriate message
        if calibration_data_missing:
            print(f"Missing calibration data for wavelength {wl}, skipping...")

        if waveplate_data_missing:
            print(f"Missing waveplate sample data for wavelength {wl}, skipping...")

        # Skip processing if any data is missing
        if calibration_data_missing or waveplate_data_missing:
            continue

        # Run the `I_ultimate_polarimetry` function
        if method == 'I':
            data = I_ultimate_polarimetry(Cal_theta, Cal_Il, Cal_Ir, theta, Il, Ir)
        elif method == 'q':
            data = q_ultimate_polarimetry(Cal_theta, Cal_Il, Cal_Ir, theta, Il, Ir)
        else:
            print("Invalid method specified. Please choose 'I' or 'q'.")
            return None

        # Store the result for the current wavelength
        polarimetry_results[wl] = data

    return polarimetry_results


def coefficient_of_variation(intensities: np.ndarray) -> float:
    """
    Computes the coefficient of variation (CV) for a given array of intensity values.

    Parameters:
    intensities (np.ndarray): Array of intensity values.

    Returns:
    float: The coefficient of variation (standard deviation / mean) as a percentage.
    """
    mean_intensity = np.mean(intensities)
    std_intensity = np.std(intensities, ddof=1)  # Use ddof=1 for sample standard deviation
    CV = std_intensity / mean_intensity if mean_intensity != 0 else np.nan  # Avoid division by zero
    return CV*100 # Convert to percentage
