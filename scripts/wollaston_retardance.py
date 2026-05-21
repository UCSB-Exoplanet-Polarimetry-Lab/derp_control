import derpy as derp
import ipdb
from tqdm import tqdm
from jax import value_and_grad
import numpy as tnp
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from katsu.katsu_math import np, set_backend_to_jax

from derpy.calibrate import (
    create_modal_basis,
    sum_of_2d_modes_wrapper,
    make_data_reduction_matrix,
    forward_model,
    make_data_reduction_matrix
)
from derpy.mask import (
    create_circular_aperture,
    create_circular_obscuration
)

NMEAS = 24
NMODES = 12
NPIX = 64
psg_angles = np.linspace(0, 2 * np.pi, NMEAS)
psa_angles = 2.5 * psg_angles

x = np.linspace(-1, 1, NPIX)
x, y = np.meshgrid(x, x)
r = np.hypot(x, y)
mask = np.zeros([NPIX, NPIX])
mask[r < 1] = 1



# Construct Calibration Basis
basis_withrotations_psg = []
basis_withrotations_psa = []
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

basis_withrotations_psg = np.asarray(basis_withrotations_psg)
basis_withrotations_psa = np.asarray(basis_withrotations_psa)

# Init the starting guesses for calibrated values
np.random.seed(32123)
x0 = np.random.random(2 + 7*NMODES) / 1e10

# ensures the piston term is quarter-wave to start / also need the second
x0[2] = np.pi / 2
x0[2 + 1*NMODES] = np.pi / 2

# Assign random spatial variation to Wollaston
x0[2 + 5 * NMODES : 2 + 7 * NMODES] = np.random.random(2 * NMODES) / 10

set_backend_to_jax()

sim_frames = forward_model(x0, basis_psg=basis_withrotations_psg, basis_psa=basis_withrotations_psa,
                           psg_angles=psg_angles, psa_angles=psa_angles)


# MAKE DATA REDUCTION MATRIX
NMODES = 1

# Construct Calibration Basis
basis_withrotations_psg = []
basis_withrotations_psa = []
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

basis_withrotations_psg = np.asarray(basis_withrotations_psg)
basis_withrotations_psa = np.asarray(basis_withrotations_psa)

# Init the starting guesses for calibrated values
x0 = tnp.random.random(2 + 7*NMODES) / 1e10

# ensures the piston term is quarter-wave to start / also need the second
x0[2] = np.pi / 2
x0[2 + 1*NMODES] = np.pi / 2

Winv = make_data_reduction_matrix(x0, basis_psg=basis_withrotations_psg, basis_psa=basis_withrotations_psa, psg_angles=psg_angles, psa_angles=psa_angles)

M_meas = Winv @ sim_frames[..., None]
M_meas = M_meas[...,0] # cut off the last axis, which was there for matrix multiplication
M_meas = M_meas.reshape([*Winv.shape[:-2], 4, 4])
M_meas /= M_meas[..., 0, 0, None, None]


derp.plot_4x4_grid(M_meas, title="Measured Mueller Matrix", vmin=-1, vmax=1, cmap="RdBu_r")

# Plot the retarder
from katsu.mueller import decompose_depolarizer, retardance_from_mueller
M_dia = np.zeros_like(M_meas)
M_ret = np.zeros_like(M_meas)
M_dep = np.zeros_like(M_meas)

for i in range(M_meas.shape[0]):
    for j in range(M_meas.shape[1]):

        mdep, mret, mdia = decompose_depolarizer(M_meas[i, j], return_all=True)
        M_dep = M_dep.at[i, j].set(mdep)
        M_ret = M_ret.at[i, j].set(mret)
        M_dia = M_dia.at[i, j].set(mdia)

retardance_pupil = retardance_from_mueller(M_ret)

plt.figure()
plt.title(f"Retardance Pupil, NMODES={NMODES}, "+fr"${np.nanmean(np.degrees(retardance_pupil))} \pm {np.nanstd(np.degrees(retardance_pupil)):.2f}^\circ$")
plt.imshow(np.degrees(retardance_pupil), cmap="RdBu_r")
plt.colorbar(label="Retardance, degrees")
plt.show()

