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
PARENT_DIR = Path.home() / "Downloads/derp-selected/depolarizer_psa_wollaston1deg_intsrphere"
CAL_DIR = PARENT_DIR / "calibration_data_2025-07-14_18-33-29.fits"
DATA_DIR = PARENT_DIR / "measurement_data_2025-07-14_18-46-38.fits"

# Get the experiment dictionaries
loaded_data = derp.load_fits_data(measurement_pth=DATA_DIR,
                                  calibration_pth=CAL_DIR,
                                  use_encoder=True)

# Reduce the data
out = loaded_data["Calibration"]
out_exp = loaded_data["Measurement"]
reduced_cal, circle_params = derp.reduce_data(out, centering=None, bin=2)
reduced_exp, circle_params_exp = derp.reduce_data(out_exp, centering=None, bin=2)

fig, ax = plt.subplots()
fig.suptitle("Check Measurement")

print(f"Reduced cal shape = {reduced_cal.shape}")
im = ax.imshow(reduced_cal[0, 0])
ax.set_xticks([])
ax.set_yticks([])

def animate(frame):
    im.set_array(reduced_cal[0, frame])
    return im

anim = FuncAnimation(
    fig,
    animate,
    frames=24,
    repeat=True,
    interval=200,
)

plt.show()

