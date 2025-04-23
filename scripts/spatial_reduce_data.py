# Standard packages
import numpy as tnp
from time import sleep
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from tqdm import tqdm
from astropy.io import fits
from scipy.optimize import minimize
from scipy.ndimage import shift
from skimage.registration import phase_cross_correlation

# autodiff stuff
from jax import value_and_grad, jacrev

# prysm polynomials
from prysm.coordinates import make_xy_grid, cart_to_polar
from prysm.geometry import circle
from prysm.polynomials import noll_to_nm, zernike_nm_sequence, sum_of_2d_modes

# Katsu stuff
from katsu.mueller import(
    linear_retarder,
    linear_polarizer,
    retardance_parameters_from_mueller,
    retardance_from_mueller
)

from katsu.katsu_math import np, set_backend_to_jax

# derpy stuff
from derpy import Experiment, forward_calibrate, forward_simulate
from derpy.data_reduction import (
    _measure_from_experiment,
    _measure_from_experiment_polychromatic,
    mueller_from_experiment
)
from derpy.writing import read_experiment

## USER CONFIGURATION HERE
# --------------------------------
BOX_PTH = "C:/Users/Work/Box/97_Data/derp/"
CALIBRATION_ID = BOX_PTH + "20250210_GPI/GPI_HWP_50nm_offset_3_air_calibration"
EXPERIMENT_ID = BOX_PTH + "20250210_GPI/GPI_HWP_50nm_offset_3_measure_0"


# NOTE: We will use the 1550nm measurement as a starting case
# mask IDs for 02/10 GPI measurements
# 1550: WVL_ID = 4, BAD_FRAMES = [5, 12, 35], BAD_FRAMES_CAL = [11]

# TODO: Make these not redundant
WVL_ID = 4
WAVELENGTH_SELECT = 1550  # nm
NMODES = 9
PLOT_INTERMEDIATE = True
MASK_RAD = 0.5 # from 0 to 1, 1 being the full circle
MODE = "left" # "left", "right", "both"
BAD_FRAMES_CAL = [11] # [3, 13 -1], _, [8]
BAD_FRAMES = [5, 12, 35]  # [-9], _, [-11]
PLOT_IMAGES = False
SET_BACKEND_TO_JAX = True
MAX_ITERS = 250
# --------------------------------

# define functions to use in main
def jax_sum_of_2d_modes(modes, weights):
    """a clone of prysm.polynomials sum_of_2d_modes that works when using katsu's Jax backend

    Parameters
    ----------
    modes : list of ndarrays
        list of polynomials constituting the desired basis
    weights : list of floats
        coefficients that describe the presence of each mode in the modes list

    Returns
    -------
    ndarray
        2D ndarray containing the sum of weighted modes
    """
    modes = np.asarray(modes)
    weights = np.asarray(weights).astype(modes.dtype)

    # dot product of the 0th dim of modes and weights => weighted sum
    return np.tensordot(modes, weights, axes=(0, 0))


def sum_of_2d_modes_wrapper(modes, weights):
    """ Wrapper that lets us ignore which source module we want to use
    """
    if np._srcmodule == tnp:
        return sum_of_2d_modes(modes, weights)
    else:
        return jax_sum_of_2d_modes(modes, weights)


def construct_zern_basis(r, t, nmodes=11, rad=0.6):
    """constructs a Zernike polynomial basis using prysm

    Parameters
    ----------
    r : ndarray
        radial coordinate of the polynomial
    t : ndarray
        azimuthal coordinate of the polynomial
    
    Returns
    -------
    basis : list of ndarrays
        list containing the requested modes
    """
    nms = [noll_to_nm(i) for i in range(1, nmodes)]

    # Norm = False is required to have unit peak-to-valley
    basis_full = list(zernike_nm_sequence(nms, r, t, norm=False))
    A = circle(rad, r) # a circular mask to apply to the beam
    basis = [mode * A for mode in basis_full ]

    return basis


def rotation_matrix(th):
    """construct a rotation matrix given some angle 'th'
    """
    return np.array([[np.cos(th), -np.sin(th)],
                     [np.sin(th), np.cos(th)]])


def register_images(ref_pupil, images):
    """Uses a cross-correlation to center the images

    Parameters
    ----------
    ref_pupil : ndarray
        circular pupil mask centered in array
    images: ndarray
        Experiment.images attribute, should be of shape
        WVL, NMEAS, WHICH_PUPIL, NPIX, NPIX

    Returns
    -------
    images_registered: ndarray
        Array of shifted images to pass back to the Experiment
    """
    images_registered = np.zeros_like(images)

    for i in range(images.shape[0]):
        for j in range(images.shape[1]):
            for k in range(images.shape[2]):

                image_select = images[i, j, k]

                move, err, pdiff = phase_cross_correlation(ref_pupil, 
                                                            image_select)

                image_shift = shift(image_select, move) 
                
                images_registered[i, j, k] = image_shift

    return images_registered


def forward_simulate(x, experiment, NMODES=11, WVL_ID=0):
    """create an array of simulated measurements, given some parameter vector x
    """

    ROTATION_RATIO = 2.5
    END_ANGLE_PSG = 180
    NMEAS = experiment.images.shape[1]
    NPIX = experiment.images.shape[-1]

    # unpack the parameters
    theta_pg = x[0] # Starting angle of the polarizer
    theta_pa = x[1] # Starting angle of the polarizer
    psg_wvp_angle = x[2]
    psa_wvp_angle = x[3]

    # get spatially varying coefficcients
    offset_coeffs = 4
    coeffs_spatial_ret_psg = x[offset_coeffs : offset_coeffs + 1*NMODES]
    coeffs_spatial_ret_psa = x[offset_coeffs + 1*NMODES:(offset_coeffs + 2*NMODES)]

    # set up the basis with rotation
    x, y = make_xy_grid(NPIX, diameter=2)
    r, t = cart_to_polar(x, y)

    # the nominal rotations performed
    rotations_psg = tnp.radians(tnp.array(experiment.psg_positions_relative))
    rotations_psa = tnp.radians(tnp.array(experiment.psa_positions_relative))

    psg_retardances, psa_retardances = [], []
    psg_fast_axes, psa_fast_axes = [], []

    # Get power scaling measurements
    P_ref_0 = experiment.mean_power_left[WVL_ID, 0] + experiment.mean_power_right[WVL_ID, 0]
    P_ref_l = np.asarray(experiment.mean_power_left[WVL_ID]) + \
              np.asarray(experiment.mean_power_right[WVL_ID])
    P_ref = P_ref_0 / P_ref_l

    # construct rotated spatial basis
    for rot_psg, rot_psa in zip(rotations_psg, rotations_psa):

        # Get the rotated spatial basis
        basis_psg = np.asarray(construct_zern_basis(r, t + rot_psg, nmodes=NMODES+1))
        basis_psa = np.asarray(construct_zern_basis(r, t + rot_psa, nmodes=NMODES+1))

        # compute retardances
        psg_retardance = sum_of_2d_modes_wrapper(basis_psg, coeffs_spatial_ret_psg)
        psa_retardance = sum_of_2d_modes_wrapper(basis_psa, coeffs_spatial_ret_psa)

        # compute fast axes

        # store arrays in list
        psg_retardances.append(psg_retardance)
        psa_retardances.append(psa_retardance)

    # get lists as arrays
    psg_retardances = np.asarray(psg_retardances)
    psa_retardances = np.asarray(psa_retardances)
    psg_fast_axes = np.asarray(psg_wvp_angle) + rotations_psg
    psa_fast_axes = np.asarray(psa_wvp_angle) + rotations_psa

    # swap axes around
    psg_retardances = np.moveaxis(psg_retardances, 0, -1)
    psa_retardances = np.moveaxis(psa_retardances, 0, -1)

    # set up the drrp
    psg_pol = linear_polarizer(theta_pg, shape=[NMEAS])
    psg_wvp = linear_retarder(psg_fast_axes, psg_retardances, shape=[NPIX, NPIX, NMEAS])

    psa_wvp = linear_retarder(psa_fast_axes, psa_retardances, shape=[NPIX, NPIX, NMEAS])
    psa_pol = linear_polarizer(theta_pa, shape=[NMEAS])

    # Create power measurements
    power_simulated = (psa_pol @ psa_wvp @ psg_wvp @ psg_pol)[..., 0, 0]
    
    # Multiply by some max NPHOTONS
    power_simulated = power_simulated / power_simulated.max() \
            * experiment.mean_power_left[WVL_ID].max()

    return power_simulated


# simple MSE loss function
def callback(xk):
    """callback to update the tqdm progress bar
    """
    pbar.update(1)


def calibrate_experiment(x, experiment):

    experiment.psg_pol_angle = x[0]
    experiment.psg_starting_angle = np.degrees(x[1])
    experiment.psg_wvp_ret = x[2]
    experiment.psa_pol_angle = x[3]
    experiment.psa_starting_angle = np.degrees(x[4])
    experiment.psa_wvp_ret = x[5]

    return experiment


def spatial_calibrate_experiment(x, experiment, NMODES, prior_psg_motion=0, prior_psa_motion=0):

    # unpack the parameters
    theta_pg = x[0] # Starting angle of the polarizer
    theta_pa = x[1] # Starting angle of the polarizer
    psg_wvp_angle = x[2]
    psa_wvp_angle = x[3]

    # get spatially varying coefficcients
    offset_coeffs = 4
    coeffs_spatial_ret_psg = x[offset_coeffs : offset_coeffs + 1*NMODES]
    coeffs_spatial_ret_psa = x[offset_coeffs + 1*NMODES:(offset_coeffs + 2*NMODES)]
    
    # make a basis
    x, y = make_xy_grid(NPIX, diameter=2)
    r, t = cart_to_polar(x, y)
    basis = construct_zern_basis(r, t, nmodes=NMODES+1)

    experiment.psg_pol_angle = theta_pg
    experiment.psa_pol_angle = theta_pa
    experiment.psg_starting_angle = np.degrees(psg_wvp_angle) + prior_psg_motion
    experiment.psg_wvp_ret = sum_of_2d_modes_wrapper(basis, coeffs_spatial_ret_psg)
    experiment.psa_wvp_ret = sum_of_2d_modes_wrapper(basis, coeffs_spatial_ret_psa)
    experiment.psa_starting_angle = np.degrees(psa_wvp_angle) + prior_psa_motion
    
    # init a new variable called offset angle that's equal to the piston mode
    return experiment

def plot_square(x, n=4, vmin=None,vmax=None, title=None, scale_offdiagonal=1/10, mask=None):
    k = 1
    plt.figure(figsize=[11,8.5])
    if title is not None:
        plt.suptitle(title, fontsize=24)
    for i in range(n):
        for j in range(n):
            plt.subplot(n,n,k)
            if mask is not None:
                x_mean = np.mean(x[..., i, j][mask==1])
                x_meansub = x[..., i, j] - x_mean
                plt.title(f'{x_mean:.2f}'+r'$\pm$'+f'{np.std(x_meansub[mask==1]):.4f}')

                if i == j:
                    plt.imshow(x[..., i, j] * mask, vmin=vmin, vmax=vmax, cmap='RdBu_r')
                else:
                    plt.imshow(x[..., i, j] * mask, vmin=vmin * scale_offdiagonal, vmax=vmax * scale_offdiagonal, cmap='PuOr_r')

            else:
                if i == j:
                    plt.imshow(x[..., i, j], vmin=vmin, vmax=vmax, cmap='RdBu_r')
                else:
                    plt.imshow(x[..., i, j], vmin=vmin * scale_offdiagonal, vmax=vmax * scale_offdiagonal, cmap='PuOr_r')

            plt.colorbar()
            plt.xticks([],[])
            plt.yticks([],[])
            k += 1
    plt.show()



if __name__ == "__main__":

    # Load in the experiment binary
    experiment_pth = CALIBRATION_ID + ".msgpack"
    exp = read_experiment(experiment_pth)

    # grab image size
    NPIX = exp.images.shape[-1]
    NMODES = 9

    # pre-register images
    x, y = make_xy_grid(NPIX, diameter=2)
    r, t = cart_to_polar(x, y)
    center_mask = circle(0.8, r)
    
    im_before = exp.images[WVL_ID,0,0]

    exp.images = register_images(center_mask, exp.images)
    
    im_after = exp.images[WVL_ID,0,0]

    plt.figure(figsize=[10, 5])
    plt.subplot(121)
    plt.title("Before centering")
    plt.imshow(im_before * center_mask)
    plt.colorbar()
    plt.subplot(122)
    plt.title("After centering")
    plt.imshow(im_after * center_mask)
    plt.colorbar()
    plt.show()

    # Make the bad frame mask for the calibration data
    mask_bad_frames = np.ones_like(exp.psg_positions_relative, dtype=bool)
    mask_bad_frames[BAD_FRAMES_CAL] = 0

    ## perform calibration
    
    # scale measurement by input power
    # Init some guess parameters
    theta_pg = tnp.random.random()
    theta_pa = tnp.random.random()
    
    # Init retardance, PSG
    coeffs_spatial_ret_psg = tnp.zeros(NMODES)
    coeffs_spatial_ret_psg[0] = tnp.pi / 2

    # Init angle, PSG
    ang_psg = 0

    # Init retardance, PSA
    coeffs_spatial_ret_psa = tnp.zeros(NMODES)
    coeffs_spatial_ret_psa[0] = tnp.pi / 2

    # Init angle, PSA
    ang_psa = 0
    
    # pupil offset
    shift_x = 0.
    shift_y = 0.

    x0 = tnp.concatenate([tnp.array([theta_pg, theta_pa, ang_psg, ang_psa]),
                         coeffs_spatial_ret_psg,
                         coeffs_spatial_ret_psa])

    # extract power_experiment
    # experiment.images is shape WVL, NMEAS, PUPIL, NPIX, NPIX
    if MODE == "left":
        PUPIL = 0

    elif MODE == "right":
        PUPIL = 1
    
    else:
        raise NotImplementedError

    results = {}

    power_experiment = exp.images[WVL_ID, :, PUPIL]
    power_experiment = tnp.moveaxis(power_experiment, 0, -1)

    
    def loss(x, experiment, NMODES=NMODES, WVL_ID=WVL_ID):
        sim_power = forward_simulate(x, exp, NMODES=NMODES, WVL_ID=WVL_ID)
        diff = sim_power - power_experiment
        diffsq = diff ** 2
        return np.mean(diffsq)**2
    
    if SET_BACKEND_TO_JAX:
        
        # set up jax and do gradients analytically
        set_backend_to_jax()
        loss_rev = jacrev(loss)

        with tqdm(total=MAX_ITERS) as pbar:
            results[str(WAVELENGTH_SELECT)] = minimize(loss, x0=x0, callback=callback,
                                         method="L-BFGS-B", jac=loss_rev,
                                         options={"maxiter": MAX_ITERS, "disp":True},
                                         args=(exp, NMODES, WVL_ID))

    else:

        # simple MSE loss function
        with tqdm(total=MAX_ITERS) as pbar:
            results[str(WAVELENGTH_SELECT)] = minimize(loss, x0=x0, callback=callback,
                                         method="L-BFGS-B", jac=False,
                                         options={"maxiter": MAX_ITERS, "disp":True},
                                         args=(exp, NMODES))


    simulated_power_wvl = []
    #for i, wvl in enumerate(wavelengths):
    simulated_power = forward_simulate(results[str(WAVELENGTH_SELECT)].x, exp, NMODES=NMODES)
    simulated_power_wvl.append(simulated_power)
    psg_angles = exp.psg_positions_relative
    exp_calibrated = spatial_calibrate_experiment(results[str(WAVELENGTH_SELECT)].x,
                                                  exp,
                                                  NMODES=NMODES)
    M = mueller_from_experiment(exp_calibrated, channel='left',
                                frame_mask=mask_bad_frames)

    # define a mask to protect us from the big dots
    mask_data = np.zeros_like(M[WVL_ID][:,:,0,0])
    x = np.linspace(-1, 1, mask_data.shape[0])
    x, y = np.meshgrid(x, x)
    r = np.sqrt(x**2 + y**2)
    if SET_BACKEND_TO_JAX:
        mask_data = mask_data.at[r < MASK_RAD].set(1)
    else:
        mask_data[r < MASK_RAD] = 1

    if PLOT_INTERMEDIATE:
        plot_square(M[WVL_ID] / M[WVL_ID][..., 0, 0, None, None], title="Air Calibration",
                    vmin=-1.1, vmax=1.1, scale_offdiagonal=1, mask=mask_data)

    # UPDATE with calibrated parameters
    prior_psg_motion = exp_calibrated.psg_positions_relative[-1]
    prior_psa_motion = exp_calibrated.psa_positions_relative[-1]

    # load up the data
    measure = read_experiment(EXPERIMENT_ID + ".msgpack")

    if PLOT_IMAGES:
        for i, img in enumerate(measure.images[WVL_ID]):
            plt.figure(figsize=[10, 5])
            plt.suptitle(f"Image {i}")
            plt.subplot(121)
            plt.title("Left")
            plt.imshow(img[0], cmap='gray', vmax=2**14)
            plt.subplot(122)
            plt.colorbar()
            plt.title("Right")
            plt.imshow(img[1], cmap='gray', vmax=2**14)
            plt.colorbar()
        plt.show()

    # Make the bad frame mask
    mask_bad_frames = np.ones_like(np.array(measure.psg_positions_relative), dtype=bool)
    if SET_BACKEND_TO_JAX:
        for i, val in enumerate(mask_bad_frames):
            if i in BAD_FRAMES:
                mask_bad_frames = mask_bad_frames.at[i].set(0)
    else:
        mask_bad_frames[BAD_FRAMES] = 0

    if PLOT_INTERMEDIATE:
        plt.style.use("bmh")
        plt.figure(figsize=[7,4])
        plt.title("Measurement")

        P_ref_0 = measure.mean_power_left[WVL_ID, 0] + measure.mean_power_right[WVL_ID, 0]
        P_ref_l = np.asarray(measure.mean_power_left[WVL_ID]) + np.asarray(measure.mean_power_right[WVL_ID])
        P_ref = P_ref_0 / P_ref_l

        plt.plot(measure.psg_positions_relative, measure.mean_power_left[WVL_ID], label='left', marker='o')
        plt.plot(measure.psg_positions_relative, measure.mean_power_right[WVL_ID], label='right', marker='o')
        plt.plot(measure.psg_positions_relative, P_ref_l, label='total', marker='o')
        plt.legend()
        plt.ylabel('Power')
        plt.xlabel('PSG Angle')


    results = results[str(WAVELENGTH_SELECT)]

    print("Wavelengths in Experiment: ", measure.wavelengths)

    # remember to put the angles and retardation back together
    measure_calibrated = spatial_calibrate_experiment(results.x, measure, NMODES=NMODES)
    measure.psg_starting_angle += prior_psg_motion
    measure.psa_starting_angle += prior_psa_motion
    
    M_measure = mueller_from_experiment(measure, channel=MODE,
                                        frame_mask=mask_bad_frames)
   
    plt.figure()
    plt.title("Calibrated PSG Retardance")
    plt.imshow(measure.psg_wvp_ret / mask_data, cmap="PuOr_r")
    plt.colorbar()
    plt.show()

    #M_measure is a list of length 1, so if if WVL_ID is anythong other than 0 [WVL_ID] returns an IdexError

    plot_square(M_measure[WVL_ID] / M_measure[WVL_ID][..., 0, 0, None, None],
                title=f"{MODE} Inversion {WAVELENGTH_SELECT} nm", vmin=-1.1, vmax=1.1,
                scale_offdiagonal=1, mask=mask_data)
    
