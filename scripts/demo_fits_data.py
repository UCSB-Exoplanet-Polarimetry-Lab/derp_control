from numpy import exp
import derpy as derp
from pathlib import Path
import ipdb
from tqdm import tqdm
from jax import value_and_grad, config, jacrev
from warnings import warn
from time import perf_counter
from astropy.io import fits

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = 'false'

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.optimize import minimize
from scipy.ndimage import shift
from katsu.katsu_math import np, set_backend_to_jax

# our spatial calibration stuff
from derpy.calibrate import (
    create_modal_basis,
    forward_model,
    sum_of_2d_modes_wrapper,
    make_data_reduction_matrix
)
from derpy.mask import (
    create_circular_aperture,
    create_circular_obscuration
)

CHANNEL = "Left" # Right, Both

# Just measuring air
CAL_DIR = Path.home() / "Downloads/derp-selected/air_wollaston1deg_intsrphere/calibration_data_2025-07-14_17-20-06.fits"
DATA_DIR = Path.home() / "Downloads/derp-selected/air_wollaston1deg_intsrphere/measurement_data_2025-07-14_17-32-59.fits"

# Measuring air with depolarizer in PSA
# CAL_DIR = Path.home() / "Downloads/derp-selected/depolarizer_psa_wollaston1deg_intsrphere/calibration_data_2025-07-14_18-46-29.fits"
# DATA_DIR = Path.home() / "Downloads/derp-selected/depolarizer_psa_wollaston1deg_intsrphere/measurement_data_2025-07-14_18-46-38.fits"

# Get the experiment dictionaries
loaded_data = derp.load_fits_data(measurement_pth=DATA_DIR,
                                  calibration_pth=CAL_DIR,
                                  use_encoder=True)

# Reduce the data
out = loaded_data["Calibration"]
out_exp = loaded_data["Measurement"]
reduced_cal, circle_params = derp.reduce_data(out, centering=None, bin=4)
reduced_exp, circle_params_exp = derp.reduce_data(out_exp, centering=None, bin=4)

# Extract which channel we are operating on
if CHANNEL == 'Left':
    # true_frames = reduced_cal[:, 0]
    # exp_frames = reduced_exp[:, 0]
    true_frames = reduced_cal[0][:24]
    exp_frames = reduced_exp[0][24:]

    # get in orientation spatial calibration likes
    # true_frames = np.moveaxis(true_frames,0, -1)
    # exp_frames = np.moveaxis(exp_frames, 0, -1)

elif CHANNEL == 'Right':
    # true_frames = reduced_cal[:, 1]
    # exp_frames = reduced_exp[:, 1]
    true_frames = reduced_cal[1]
    exp_frames = reduced_exp[1]

    # get in orientation spatial calibration likes
    # true_frames = np.moveaxis(true_frames,0, -1)
    # exp_frames = np.moveaxis(exp_frames, 0, -1)

elif CHANNEL == 'Both':
    # Concatenate the left, then right frames
    left_frames = reduced_cal[:, 0]
    right_frames = reduced_cal[:, 1]
    true_frames = np.concatenate([left_frames, right_frames])

    left_frames = reduced_exp[:, 0]
    right_frames = reduced_exp[:, 1]
    exp_frames = np.concatenate([left_frames, right_frames])
    warn("Channel 'Both' is untested, be wary of results")


# Generate polynomials
NMODES = 32
NPIX = true_frames.shape[-1]
basis = create_modal_basis(NMODES, NPIX)

# Create a mask from the circle parameters
# TODO: I think these are from circle fitting BEFORE binning, binning should happen first!
mask = np.zeros((NPIX, NPIX), dtype=int)
y0, x0 = circle_params['center']
radius = circle_params['radius'] / 4 # divide by bin amount

x = np.arange(-NPIX//2, NPIX//2, dtype=np.float64)
y, x = np.meshgrid(x, x)
r = np.sqrt((y)**2 + (x)**2)

# Using 90% of the radius to account for misregistration at the edges
mask[r <= radius * 0.95] = 1

# # Mask the artifact from the collimator
# dot = create_circular_obscuration(mask.shape[0], radius=7, center=(130, 102))

# mask *= dot
# mask = mask.astype(bool) # need float to use nans
mask[mask < 1e-10] = 0
mask_extend = [mask for i in range(true_frames.shape[0])]
mask_extend = np.asarray(mask_extend)
mask_extend = np.moveaxis(mask_extend, 0, -1)

# Apply the mask to the true frames
true_frames_masked = [i * mask for i in true_frames]
true_frames = np.asarray(true_frames_masked)
true_frames = np.moveaxis(true_frames, 0, -1)

exp_frames_masked = [i * mask for i in exp_frames]
exp_frames = np.asarray(exp_frames_masked)
exp_frames = np.moveaxis(exp_frames, 0, -1)
basis_masked = [i * mask for i in basis]


# Init the starting guesses for calibrated values
np.random.seed(32123)
x0 = np.random.random(4 + 2*len(basis)) / 10

# ensures the piston term is quarter-wave to start / also need the second
x0[4] = np.pi / 2
x0[4 + len(basis)] = np.pi / 2
psg_angles = np.radians(out['psg_angles'].data[:24])
psa_angles = np.radians(out['psa_angles'].data[:24])

set_backend_to_jax()
mask_extend = mask_extend.astype(np.float64)
mask_extend[mask_extend==0] = np.nan

def loss(x):

    if CHANNEL.lower() == "both":
        sim_frames = forward_model(x, basis_masked, psg_angles,
                                    rotation_ratio=2.5,
                                    dual_I=True)
    else:
        sim_frames = forward_model(x, basis_masked, psg_angles,
                                    rotation_ratio=2.5,
                                    dual_I=False,
                                    psa_angles=psa_angles)


    sim_array = np.asarray(sim_frames)#[mask_extend]
    true_array = np.asarray(true_frames)#[mask_extend]
    diff_array = (sim_array - true_array)**2
    # mean_array = np.nanmean(diff_array[mask, :], axis=(0))
    diff_array = np.abs(sim_array - true_array)**2

    # nanmean is important for masked values
    #MSE = np.sum(mean_array)
    MSE = np.mean(diff_array[mask_extend==1])  # / len(mask[mask==1])

    return MSE

from time import perf_counter
t1 = perf_counter()
_ = loss(x0)
print(f"Time taken for forward model: {perf_counter() - t1:.2f} seconds")

loss_rev = jacrev(loss)

t1 = perf_counter()
_ = loss_rev(x0)
print(f"Time taken for reverse model: {perf_counter() - t1:.2f} seconds")

# Callback at every function initialization
pbar = None # Initialize pbar globally or pass it as an argument
def callback_function(xk):
    global pbar
    if pbar is None:
        pbar = tqdm(desc="Optimization Progress") # Example total
    pbar.update(1) # Increment the progress bar

results = minimize(loss, x0=x0, method="L-BFGS-B", jac=loss_rev,
                    callback=callback_function,
                    options={"maxiter":1e10, "ftol":1e-10, "gtol":1e-10})

if pbar is not None:
    pbar.close()

# This is to debug error in the model
# loss_stop(results.x)


# extract the retarder coeffs
psg_ret_coeffs = results.x[4 : 4+len(basis)]
psg_retarder_estimate = sum_of_2d_modes_wrapper(basis_masked, psg_ret_coeffs)

psa_ret_coeffs = results.x[4+len(basis) : 4 + 2*len(basis)]
psa_retarder_estimate = sum_of_2d_modes_wrapper(basis_masked, psa_ret_coeffs)

plt.figure()
plt.subplot(131)
plt.title("Estimated PSG Retarder")
plt.imshow(psg_retarder_estimate / mask, cmap="RdBu_r")
plt.colorbar()
plt.xticks([], [])
plt.yticks([], [])
plt.subplot(132)
plt.title("Estimated PSA Retarder")
plt.imshow(psa_retarder_estimate / mask, cmap="RdBu_r")
plt.colorbar()
plt.xticks([], [])
plt.yticks([], [])
plt.subplot(133)
plt.plot(psg_ret_coeffs, label="PSG coefficients", marker="x")
plt.plot(psa_ret_coeffs, label="PSA coefficients", marker="x")
plt.legend()

# Running out of GPU memory oops
del psg_retarder_estimate, psa_retarder_estimate

# create simulated power
if not CHANNEL.lower() == "both":
    sim_frames = forward_model(results.x, basis_masked, psg_angles, psa_angles=psa_angles)
else:
    sim_frames = forward_model(results.x, basis_masked, psg_angles,
                                rotation_ratio=2.5,
                                dual_I=True)


# perform a comparison via mean power
# NOTE I've commited a heinous crime with the following lines of code, please
# forgive me. To help explain, I wanted to do list comprehension over the last
# axis of the `sim_frames` and `true_frames` arrays. This was the most concise
# way I could think of doing so
mean_simulated = [np.mean(i[mask==1]) for i in np.moveaxis(sim_frames, -1, 0)]
mean_observed = [np.mean(i[mask==1]) for i in np.moveaxis(true_frames, -1, 0)]

if CHANNEL == "Both":
    psg_angles_plot = np.concatenate([psg_angles, psg_angles])
else:
    psg_angles_plot = psg_angles

plt.figure()
plt.plot(np.degrees(psg_angles_plot), mean_simulated, label="Fit Power", marker="x")
plt.plot(np.degrees(psg_angles_plot), mean_observed, label="Measured Power",
                                                marker="o", linestyle=None)
plt.ylabel("power")
plt.xlabel("PSG Angle, deg")
plt.legend()

# Perform polarimetric data reduction before and after calibration
# Update results.x with the global rotation
spatial_cal_results = results.x.copy()

# spatial_cal_results[2] += psg_angles[-1]
# spatial_cal_results[3] += psa_angles[-1]

# experiment PSG angles
psg_angles_exp = np.radians(out_exp['psg_angles'].data.astype(np.float64))[24:]
psa_angles_exp = np.radians(out_exp['psa_angles'].data.astype(np.float64))[24:]


if CHANNEL.lower() == "both":
    Winv = make_data_reduction_matrix(spatial_cal_results,
                                        basis_masked,
                                        psg_angles_exp,
                                        rotation_ratio=2.5,
                                        dual_I=True)
else:
    Winv = make_data_reduction_matrix(spatial_cal_results,
                                        basis_masked,
                                        psg_angles_exp,
                                        rotation_ratio=2.5,
                                        dual_I=False,
                                        psa_angles=psa_angles_exp)


true_array = np.asarray(exp_frames)
true_array = true_array[..., np.newaxis]

# The polarimetric data reduction step
M_meas = Winv @ true_array

M_meas = M_meas[...,0] # cut off the last axis, which was there for matrix multiplication
M_meas = M_meas.reshape([*Winv.shape[:-2], 4, 4])
M_meas /= M_meas[..., 0, 0, None, None]

# Get the RMS of data within mask
I = np.eye(4)

med_M = np.median(M_meas[mask], axis=0)
var = (med_M - I)**2
rms = np.sqrt(np.sum(var))

derp.plot_4x4_grid(M_meas, title="Measured Mueller Matrix, "+fr"$\sigma={rms:.5f}$", vmin=-1, vmax=1, cmap="RdBu_r")
plt.show()

WAVELENGTH_SELECT = 1500

hdu = fits.PrimaryHDU(M_meas)
hdu.header["NMODES"] = (NMODES, "Number of Spatial Modes used to calibrate")
hdu.header["WAVELENGTH"] = (WAVELENGTH_SELECT, "Measured Wavelength")
hdu.writeto(f"spatial_cal_gpi_hwp_{NMODES}modes_1e-40ftol.fits", overwrite=True)
