"""
This is a script that runs the pre-generation of rotated modal bases before calibrating
using data from Dan Shanks' VIS-DERP at JPL's MDL
"""

from numpy import exp
import derpy as derp
from pathlib import Path
import ipdb
from tqdm import tqdm
from jax import value_and_grad, config, jacrev, debug, jit
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
from katsu.mueller import linear_retarder

# our spatial calibration stuff
from derpy.calibrate import (
    create_modal_basis,
    sum_of_2d_modes_wrapper,
    make_data_reduction_matrix
)
from derpy.mask import (
    create_circular_aperture,
    create_circular_obscuration
)

# Options for cost functions
from prysm.x.optym.cost import bias_and_gain_invariant_error


"""
USER INPUTS
----------------------------------------------------------
"""
CHANNEL = "Left" # Right, Both

NMODES = 1
TOL = 1e-10 # adjusts both function and gradient tolerance, exits when EITHER are below this value

# Just measuring air
CAL_DIR = Path.home() / "Data/dans_data" \
/ "Capture_DRRP_Photodiode_251103_163635_UNCORRECTED.fits"

DATA_DIR = Path.home() / "Data/dans_data" \
/ "Capture_DRRP_Photodiode_251104_091851_UNCORRECTED.fits"

"""
----------------------------------------------------------
"""

hdu_cal = fits.open(CAL_DIR)
hdu_data = fits.open(DATA_DIR)

# Get the experiment dictionaries
loaded_data = derp.load_fits_data(measurement_pth=DATA_DIR,
                                  calibration_pth=CAL_DIR,
                                  use_encoder=False,
                                  centering_ref_img=10,
                                  use_photodiode=True)

# Reduce the data
binsize = 12
out = loaded_data["Calibration"]
out_exp = loaded_data["Measurement"]

# make a mask
print(out["images"].shape)
before_bin_mask = np.zeros_like(out["images"][0])
x = np.linspace(-1, 1, before_bin_mask.shape[0])
x, y = np.meshgrid(x, x)
r = np.hypot(x, y)
before_bin_mask[r < 1.] = 1

reduced_cal, circle_params = derp.reduce_data(out,
                                              centering=None,
                                              bin=binsize,
                                              mask=before_bin_mask)

reduced_exp, circle_params_exp = derp.reduce_data(out_exp,
                                                  centering=None,
                                                  bin=binsize,
                                                  mask=before_bin_mask)
true_frames = reduced_cal
exp_frames = reduced_exp

# Generate polynomials
NPIX = true_frames.shape[-1]

# Create a mask from the circle parameters
mask = np.ones_like(true_frames[0])
mask[true_frames[0] < 1] = 0
# mask = np.zeros((NPIX, NPIX), dtype=int)
y0, x0 = circle_params['center']
radius = circle_params['radius'] / binsize # divide by bin amount

# x = np.arange(-NPIX//2, NPIX//2, dtype=np.float64)
# y, x = np.meshgrid(x, x)
# r = np.sqrt((y)**2 + (x)**2)
# 
# # Using 90% of the radius to account for misregistration at the edges
# mask[r <= radius * .6] = 1

# Apply the mask to the true frames
print(f"true frames shape = {true_frames.shape}")
true_frames_masked = [i * mask for i in true_frames]
true_frames = np.asarray(true_frames_masked)
true_frames = np.moveaxis(true_frames, 0, -1)

exp_frames_masked = [i * mask for i in exp_frames]
exp_frames = np.asarray(exp_frames_masked)
exp_frames = np.moveaxis(exp_frames, 0, -1)

# Init the starting guesses for calibrated values
np.random.seed(32123)
x0 = np.random.random(2 + 4*NMODES) / 100

# ensures the piston term is quarter-wave to start / also need the second
x0[2] = np.pi / 2
x0[2 + 1*NMODES] = np.pi / 2

# x0[2 + 4*NMODES] = 0 # PSA is a polarizer
# x0[2 + 4*NMODES+1:] = 0
psg_angles = np.radians(out['psg_angles'].data.astype(np.float64))
psa_angles = np.radians(out['psa_angles'].data.astype(np.float64))

# experiment PSG angles
psg_angles_exp = np.radians(out_exp['psg_angles'].data.astype(np.float64))
psa_angles_exp = np.radians(out_exp['psa_angles'].data.astype(np.float64))

from derpy.calibrate import forward_model

basis_withrotations_psg = []
basis_withrotations_psa = []
basis_withrotations_psg_exp = []
basis_withrotations_psa_exp = []

plt.figure()
plt.title("Mask before basis creation")
plt.imshow(mask)
plt.colorbar()

# Construct Calibration Basis
for offset_psg, offset_psa in zip(psg_angles, psa_angles):
    
    # offset is in radians to be compatible with prysm angles
    basis = create_modal_basis(NMODES, NPIX, angle_offset=offset_psg)
    basis_masked = [i * mask for i in basis]
    basis_masked = np.asarray(basis_masked)
    basis_withrotations_psg.append(basis_masked)
    
    basis = create_modal_basis(NMODES, NPIX, angle_offset=offset_psa)
    basis_masked = [i * mask for i in basis]
    basis_masked = np.asarray(basis_masked)
    basis_withrotations_psa.append(basis_masked)

# Construct Experiment Basis
for offset_psg_exp, offset_psa_exp in zip(psg_angles_exp, psa_angles_exp):
    
    # offset is in radians to be compatible with prysm angles
    basis = create_modal_basis(NMODES, NPIX, angle_offset=offset_psg_exp)
    basis_masked = [i * mask for i in basis]
    basis_masked = np.asarray(basis_masked)
    basis_withrotations_psg_exp.append(basis_masked)
    
    basis = create_modal_basis(NMODES, NPIX, angle_offset=offset_psa_exp)
    basis_masked = [i * mask for i in basis]
    basis_masked = np.asarray(basis_masked)
    basis_withrotations_psa_exp.append(basis_masked)

# override the prior basis
basis_masked_psg = np.asarray(basis_withrotations_psg)
basis_masked_psa = np.asarray(basis_withrotations_psa)
basis_masked_psg_exp = np.asarray(basis_withrotations_psg_exp)
basis_masked_psa_exp = np.asarray(basis_withrotations_psa_exp)

mode_to_show = NMODES-1
angle_to_show = 4

plt.figure(figsize=[16,4])
plt.suptitle("Checking Modes")
plt.subplot(141)
plt.imshow(basis_masked_psg[angle_to_show, mode_to_show])
plt.colorbar()
plt.subplot(142)
plt.imshow(basis_masked_psa[angle_to_show, mode_to_show])
plt.colorbar()
plt.subplot(143)
plt.imshow(basis_masked_psg_exp[angle_to_show, mode_to_show])
plt.colorbar()
plt.subplot(144)
plt.imshow(basis_masked_psa_exp[angle_to_show, mode_to_show])
plt.colorbar()

plt.figure()
plt.title("Checking power frames masking")
print("True frames shape =  ",true_frames.shape)
plt.imshow(true_frames[...,0])
plt.colorbar()
# Clear memory
del basis_withrotations_psg, basis_withrotations_psa
del basis_withrotations_psg_exp, basis_withrotations_psa_exp

set_backend_to_jax()

def GIE(I, D):
    t1 = np.sum(I * D) ** 2
    t2 = np.sum(D ** 2) 
    t3 = np.sum(I ** 2)
    return 1 - t1 / (t2 * t3)


def MSE(I, D):
    squared_error = (I - D) ** 2
    return np.mean(squared_error)


"""
Cases
- 1) Fit to only polarizer angles, retarder fast axis and retardance
- 2) Fit to polarizer angles and diattenuation, retarder fast axis and retardance
- 3) Fit to polarizer angles and diattenuation and retardance, retarder fast axis and retardance
"""

# def loss(x):
# 
#     sim_frames = forward_model(x, basis_masked_psg, basis_masked_psa,
#                                 psg_angles,
#                                 dual_I=False,
#                                 psa_angles=psa_angles)
# 
# 
#     sim_array = np.asarray(sim_frames)
#     true_array = np.asarray(true_frames)
#     true_array = true_array / true_array.max()
#     return MSE(sim_array[mask_extend==1], true_array[mask_extend==1])


# Try a different loss where we normalize by the identity matrix

def loss(x):

    true_array = np.asarray(true_frames)
    true_array = true_array[..., np.newaxis]
    
    # make the data reduction matrix
    Winv = make_data_reduction_matrix(x,
                                      basis_masked_psg,
                                      basis_masked_psa,
                                      psg_angles=psg_angles,
                                      psa_angles=psa_angles)
    
    M_meas = Winv @ true_array
    M_meas = M_meas[...,0] # cut off the last axis, which was there for matrix multiplication
    M_meas = M_meas.reshape([*Winv.shape[:-2], 4, 4]) # make a Mueller matrix again
    
    # Normalize using mask
    M_meas = M_meas.at[mask==1.].set(M_meas[mask==1] / M_meas[mask==1, 0, 0, None, None]) # Normalize by the transmission element
    
    return MSE(np.eye(4), M_meas[mask==1])


from time import perf_counter
_ = loss(x0)
t1 = perf_counter()
loss_fg = value_and_grad(loss)
f, g = loss_fg(x0)

print(f"Time taken to compile and run the fg(x): {perf_counter() - t1:.2f} seconds")
print(f"function val = {f}")
print(f"gradient val = {g}")

# Callback at every function initialization
pbar = None # Initialize pbar globally or pass it as an argument
def callback_function(xk):
    global pbar
    if pbar is None:
        pbar = tqdm(desc="Optimization Progress") # Example total
    pbar.update(1) # Increment the progress bar

results = minimize(loss_fg, x0=x0, method="L-BFGS-B", jac=True,
                    callback=callback_function,
                   options={"maxiter":100_000, "ftol":TOL, "gtol":TOL,
                            "maxfun":100_000})

if pbar is not None:
    pbar.close()

print(results)

# extract the retarder coeffs
psg_ret_coeffs = results.x[2 : 2+len(basis)]
psg_retarder_estimate = sum_of_2d_modes_wrapper(basis_masked_psg, psg_ret_coeffs)[0]

psa_ret_coeffs = results.x[2 + len(basis) : 2 + 2*len(basis)]
psa_retarder_estimate = sum_of_2d_modes_wrapper(basis_masked_psa, psa_ret_coeffs)[0]

psg_ang_coeffs = results.x[2 + 2 * len(basis) : 2 + 3 * len(basis)]
psg_angle_estimate = sum_of_2d_modes_wrapper(basis_masked_psg, psg_ang_coeffs)[0]

psa_ang_coeffs = results.x[2 + 3 * len(basis) : 2 + 4 * len(basis)]
psa_angle_estimate = sum_of_2d_modes_wrapper(basis_masked_psa, psa_ang_coeffs)[0]

# psa_dia_coeffs = results.x[2 + 4 * len(basis) : 2 + 5 * len(basis)]
# psa_dia_estimate = sum_of_2d_modes_wrapper(basis_masked_psa, psa_dia_coeffs)[0]
# 
# psa_dia_coeffs_ret = results.x[2 + 5 * len(basis) : 2 + 6 * len(basis)]
# psa_dia_estimate_ret = sum_of_2d_modes_wrapper(basis_masked_psa, psa_dia_coeffs_ret)[0]
# 
# psa_dia_coeffs_ang = results.x[2 + 6 * len(basis) : 2 + 7 * len(basis)]
# psa_dia_estimate_ang = sum_of_2d_modes_wrapper(basis_masked_psa, psa_dia_coeffs_ang)[0]

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

plt.figure()
plt.subplot(131)
plt.title("Estimated PSG Retarder Angle")
plt.imshow(psg_angle_estimate / mask, cmap="RdBu_r")
plt.colorbar()
plt.xticks([], [])
plt.yticks([], [])
plt.subplot(132)
plt.title("Estimated PSA Retarder Angle")
plt.imshow(psa_angle_estimate / mask, cmap="RdBu_r")
plt.colorbar()
plt.xticks([], [])
plt.yticks([], [])
plt.subplot(133)
plt.plot(psg_ang_coeffs, label="PSG coefficients", marker="x")
plt.plot(psa_ang_coeffs, label="PSA coefficients", marker="x")
plt.legend()

# plt.figure()
# plt.subplot(121)
# plt.title("Estimated PSA Diattenutation")
# plt.imshow(psa_dia_estimate / mask)
# plt.colorbar()
# plt.xticks([], [])
# plt.yticks([], [])
# plt.subplot(122)
# plt.plot(psa_dia_coeffs, label="Wollaston Diattenuation coefficients", marker="x")
# plt.legend()
# 
# plt.figure()
# plt.subplot(131)
# plt.title("Estimated Wollaston Retardance")
# plt.imshow(psa_dia_estimate_ret / mask, cmap="RdBu_r")
# plt.colorbar()
# plt.xticks([], [])
# plt.yticks([], [])
# plt.subplot(132)
# plt.title("Estimated Wollaston Retarder Angle")
# plt.imshow(psa_dia_estimate_ang / mask, cmap="RdBu_r")
# plt.colorbar()
# plt.xticks([], [])
# plt.yticks([], [])
# plt.subplot(133)
# plt.plot(psa_dia_coeffs_ret, label="PSG coefficients", marker="x")
# plt.plot(psa_dia_coeffs_ang, label="PSA coefficients", marker="x")
# plt.legend()

# Running out of GPU memory oops
del psg_retarder_estimate, psa_retarder_estimate

# create simulated power
sim_frames = forward_model(results.x, basis_masked_psg, basis_masked_psa,
                            psg_angles,
                            rotation_ratio=4.91,
                            dual_I=False,
                            psa_angles=psa_angles)


# perform a comparison via mean power
# NOTE I've commited a heinous crime with the following lines of code, please
# forgive me. To help explain, I wanted to do list comprehension over the last
# axis of the `sim_frames` and `true_frames` arrays. This was the most concise
# way I could think of doing so
mean_simulated = [np.mean(i[mask==1]) for i in np.moveaxis(sim_frames, -1, 0)]
mean_observed = [np.mean(i[mask==1]) for i in np.moveaxis(true_frames, -1, 0)]
mean_simulated = np.asarray(mean_simulated)
mean_observed = np.asarray(mean_observed)
psg_angles_plot = psg_angles

plt.figure()
plt.title("max-normalized power observed")
plt.plot(np.degrees(psg_angles_plot), mean_simulated / mean_simulated.max(), label="Fit Power", marker="x")
plt.plot(np.degrees(psg_angles_plot), mean_observed / mean_observed.max(), label="Measured Power",
                                                marker="o", linestyle=None)
plt.ylabel("power")
plt.xlabel("PSG Angle, deg")
plt.legend()

# Perform polarimetric data reduction before and after calibration
spatial_cal_results = results.x.copy()

plt.figure(figsize=[16,4])
plt.suptitle("Checking Modes Experiment")
plt.subplot(141)
plt.imshow(basis_masked_psg[angle_to_show, mode_to_show])
plt.colorbar()
plt.subplot(142)
plt.imshow(basis_masked_psa[angle_to_show, mode_to_show])
plt.colorbar()
plt.subplot(143)
plt.imshow(basis_masked_psg_exp[angle_to_show, mode_to_show])
plt.colorbar()
plt.subplot(144)
plt.imshow(basis_masked_psa_exp[angle_to_show, mode_to_show])
plt.colorbar()

Winv = make_data_reduction_matrix(results.x,
                                    basis_masked_psg_exp,
                                    basis_masked_psa_exp,
                                    psg_angles_exp,
                                    rotation_ratio=4.91,
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

med_M = np.nanmedian(M_meas[mask.astype(int)], axis=0)
var = (med_M - I)**2
rms = np.sqrt(np.sum(var))

derp.plot_4x4_grid(M_meas, title="Measured Mueller Matrix, "+f"{np.nanmean(med_M):.5f}" + r"$\pm$ " + f"{rms:.5f}", vmin=-1, vmax=1, cmap="RdBu_r")


# Plot the retarder
from katsu.mueller import decompose_depolarizer, retardance_from_mueller
M_dia = np.zeros_like(M_meas)
M_ret = np.zeros_like(M_meas)
M_dep = np.zeros_like(M_meas)

# It would be nice to have a faster way of doing this that avoids NaNs
from scipy.linalg import logm

# Need to store the ellipse parameters
A = np.zeros(M_meas.shape[:-2])
B = np.zeros(M_meas.shape[:-2])
theta = np.zeros(M_meas.shape[:-2])
handedness = np.zeros(M_meas.shape[:-2])
qwp = linear_retarder(0, np.pi / 4) 

for i in range(M_meas.shape[0]):
    for j in range(M_meas.shape[1]):

        mdep, mret, mdia = decompose_depolarizer(M_meas[i, j], return_all=True)
        M_dep = M_dep.at[i, j].set(mdep)
        M_ret = M_ret.at[i, j].set(mret)
        M_dia = M_dia.at[i, j].set(mdia)
        
        # Let's get the eigenpolarization map from the retarder
        # mret = qwp @ mret 
        tracem = np.trace(mret, axis1=-2, axis2=-1)
        phi = np.arccos(tracem/2 - 1)
        front = phi / (2 * np.sin(phi))

        # These are the given quantities, but it looks flipped to me
        phi_h = front * (mret[..., 2, 3] - mret[..., 3, 2])
        phi_45 = front * (mret[..., 3, 1] - mret[..., 1, 3])
        phi_L = front * (mret[..., 2, 1] - mret[..., 1, 2])

        # Determine the stokes vector
        stokes_fast = np.asarray([1, phi_h, phi_45, phi_L])
        # stokes_slow = np.asarray([1, -phi_h, -phi_45, -phi_L])

        absL = np.sqrt(phi_h**2 + phi_45**2)
        Ip = np.sqrt(absL**2 + phi_L**2)

        # Ellipse parameters
        theta = theta.at[i, j].set(0.5 * np.arctan2(phi_45, phi_h))
        A = A.at[i, j].set(np.sqrt(0.5 * (Ip + absL)))
        B = B.at[i, j].set(np.sqrt(0.5 * (Ip - absL)))
        handedness = handedness.at[i, j].set(np.sign(phi_L))

retardance_pupil = retardance_from_mueller(M_ret)

plt.figure()
plt.title(f"Retardance Pupil, NMODES={NMODES}, "+fr"${np.nanmean(np.degrees(retardance_pupil))} \pm {np.nanstd(np.degrees(retardance_pupil)):.2f}^\circ$")
plt.imshow(np.degrees(retardance_pupil), cmap="RdBu_r", vmin=1, vmax=4)
plt.colorbar(label="Retardance, degrees")

plt.figure(figsize=[12, 3])
plt.subplot(141)
plt.imshow(A)
plt.title("Semi-major axis")
plt.colorbar()
plt.subplot(142)
plt.imshow(B)
plt.title("Semi-minor axis")
plt.colorbar()
plt.subplot(143)
plt.imshow(theta)
plt.title("AoLP")
plt.colorbar()
plt.subplot(144)
plt.imshow(handedness)
plt.title("Handedness")
plt.colorbar()


# Try this ellipse plotting code
from matplotlib.patches import Ellipse

# Example data - replace with your actual arrays

# Create background image (e.g., intensity)
background = np.degrees(retardance_pupil)
size = background.shape[0]

# Create figure
fig, ax = plt.subplots(figsize=(10, 10))

# Display background
im = ax.imshow(background, cmap='Spectral', origin='lower', extent=[0, size, 0, size])
plt.colorbar(im, ax=ax, label='Retardance [rad]')

# Downsample for clearer visualization (plot every nth ellipse)
step = 5  # Adjust this to control density of ellipses
scale = 1  # Scale factor for ellipse size

for i in range(0, size, step):
    for j in range(0, size, step):
        # Get ellipse parameters at this position
        a = A[i, j] * scale
        b = B[i, j] * scale
        angle = theta[i, j]
        h = handedness[i, j]
        
        # Create ellipse
        ellipse = Ellipse(
            xy=(j + 0.5, i + 0.5),  # Center position
            width=2*a,              # Full width
            height=2*b,             # Full height
            angle=angle,            # Rotation angle
            facecolor='none',
            edgecolor='red' if h > 0 else 'blue',  # Color by handedness
            linewidth=1.5,
            alpha=0.7
        )
        ax.add_patch(ellipse)

# Set labels and title
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_title('Elliptical Polarization Visualization\n(Red: Right-handed, Blue: Left-handed)')
ax.set_xlim(0, size)
ax.set_ylim(0, size)

plt.tight_layout()
plt.show()

WAVELENGTH_SELECT = 595

hdu = fits.PrimaryHDU(M_meas)
hdu.header["NMODES"] = (NMODES, "Number of Spatial Modes used to calibrate")
hdu.header["WAVELENGTH"] = (WAVELENGTH_SELECT, "Measured Wavelength")
hdu.writeto(f"spatial_cal_gpi_hwp_{NMODES}modes_1e-40ftol.fits", overwrite=True)
