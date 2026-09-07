"""Running the standard data reduction, generates plots
"""

# Standard imports
from pathlib import Path
from time import perf_counter
from astropy.io import fits
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# polarimetry stuff
from katsu.katsu_math import np, set_backend_to_jax
from katsu.models import Model, LinearRetarder, LinearDiattenuator
import derpy

# fitting stuff
import jax
import optax
import zodiax as zdx
import equinox as eqx
from tqdm import tqdm


"""
-------------------------
--- Put in data paths ---
-------------------------
"""

# Let's try and load up Dan's data
CAL_DIR = Path.home() / "Data/microscope_objective" \
/ "calibration_data_2025-09-18_15-26-15.fits"

DATA_DIR = Path.home() / "Data/microscope_objective" \
/ "measurement_data_2025-09-18_16-24-34.fits"

# Let's try and load up Dan's data
CAL_DIR = Path.home() / "Data/dans_data" \
/ "Capture_DRRP_Photodiode_251103_163635_UNCORRECTED.fits"

DATA_DIR = Path.home() / "Data/dans_data" \
/ "Capture_DRRP_Photodiode_251104_091851_UNCORRECTED.fits"

# Define a bin-down
BINSIZE = 4

# Get the experiment dictionaries
out = derpy.load_fits_data(measurement_pth=CAL_DIR,
                                  use_encoder=False,
                                  centering_ref_img=0,
                                  use_photodiode=True,
                                  label="Dan_1103",
                                  mask_frames=None)

out_exp = derpy.load_fits_data(measurement_pth=DATA_DIR,
                                  use_encoder=False,
                                  centering_ref_img=0,
                                  use_photodiode=True,
                                  label="Dan_1103")

# Set up a data mask
out["images"] = out["images"] / out["images"][0]
before_bin_mask = np.zeros_like(out["images"][0])
x, y = np.meshgrid(np.linspace(-1, 1, out['images'].shape[2]),
                   np.linspace(-1, 1, out['images'].shape[1]))
r = np.hypot(x, y)
before_bin_mask[r < 0.9] = 1

set_backend_to_jax()  # must precede construction of any Mueller object

# Pull angles out of data reduction
psg_angles = np.radians(out['psg_angles'].astype(np.float64))
psa_angles = np.radians(out['psa_angles'].astype(np.float64))

# Set up the DRRP layers
layers = [
    ('psg_polarizer', LinearDiattenuator(
        # DRRPs are invariant under rotation, so this angle is fixed
        transmission_axis=0., 
        Tmin=0.,
        shape=psg_angles.shape)
    ),
    ('psg_retarder', LinearRetarder(
        fast_axis=psg_angles,
        retardance=np.pi / 2,
        offset=0.,
        shape=psg_angles.shape)
    ),
    ('psa_retarder', LinearRetarder(
        fast_axis=psa_angles,
        retardance=np.pi / 2,
        offset=0.,
        shape=psa_angles.shape)
    ),
    ('psa_polarizer', LinearDiattenuator(
        transmission_axis=np.pi / 2,
        Tmin=0.,
        shape=psa_angles.shape)
    ),
]

"""
-------------------------------------
--- Calibrate the DRRP with zodiax ---
-------------------------------------
"""
# The calibration measurement is air (no sample), so the model is just the
# four layers above. Power measured by the photodiode.
power_measured = np.asarray(out["images"], dtype=np.float64)
power_measured = np.mean(power_measured, axis=(-1, -2))
stokes_in = np.array([1., 0., 0., 0.])

system = Model(layers)

# Both polarizers share the attribute name `transmission_axis`, and both
# retarders share `retardance`. `Model.__getattr__` returns the *first* layer
# that has an attribute, so bare names like "retardance" would silently grab
# the PSG one. Address the leaves by their full `layer.attribute` path instead.
CAL_PARAMS = [
    "psg_retarder.retardance",
    "psa_retarder.retardance",
    "psg_retarder.offset",
    "psa_retarder.offset",
    "psa_polarizer.transmission_axis",
]

# `as_dict=True` casts the leaves to jax arrays; raw python floats are static
# under eqx.filter_value_and_grad and would come back with `None` gradients.
params = system.get(CAL_PARAMS, as_dict=True)

# The retardances are deliberately left unbounded. Projecting them back into a
# box after each step is invisible to the L-BFGS line search: the step actually
# taken no longer matches the one the curvature pair was recorded for, and runs
# terminated with a retardance pinned exactly on a wall. The branch ambiguity
# the box was meant to remove is handled at reporting time instead, by
# `canonicalise_retarder` below, which has no boundary to stick to.


def loss_fn(params, model, stokes, data):
    """Sum-square error between modeled and measured power.

    The model returns power normalized to the input Stokes vector, while the
    photodiode reads in its own units. Rather than fit a gain parameter (which
    is degenerate with the rest of the model), solve for the least-squares
    optimal gain analytically at each step.
    """
    model = model.set(params)
    modeled = model.forward(stokes)[..., 0]
    gain = np.sum(modeled * data) / np.sum(modeled ** 2)
    return np.sum((gain * modeled - data) ** 2)


def objective(params):
    """The loss as a function of the parameters alone.

    L-BFGS drives a line search that re-evaluates the objective internally, so
    it needs a callable that takes nothing but `params`.
    """
    return loss_fn(params, system, stokes_in, power_measured)


# --- L-BFGS local solver --------------------------------------------------
solver = optax.lbfgs()

# Reuses the value/gradient the line search has already computed rather than
# recomputing them at the top of each iteration.
value_and_grad = optax.value_and_grad_from_state(objective)

N_LOCAL_STEPS = 300   # L-BFGS iterations per local solve
N_STARTS = 8        # random restarts, run in parallel under vmap
N_HOPS = 60           # basin-hopping proposals
HOP_SIZE = 0.2        # rad, std dev of the basin-hopping perturbation


@eqx.filter_jit
def minimise(params):
    """One complete L-BFGS local solve, compiled as a single `scan`.

    Running the whole descent inside `lax.scan` rather than stepping it from
    Python turns a local solve into one compiled call instead of N_LOCAL_STEPS
    dispatches. That is what makes hundreds of restarts and hops affordable,
    and it lets the solve be `vmap`ped.
    """
    state = solver.init(params)

    def body(carry, _):
        p, st = carry
        value, grad = value_and_grad(p, state=st)
        updates, st = solver.update(
            grad, st, p, value=value, grad=grad, value_fn=objective
        )
        return (optax.apply_updates(p, updates), st), value

    (p, _), history = jax.lax.scan(body, (params, state), None,
                                   length=N_LOCAL_STEPS)
    return p, objective(p), history


# Independent restarts are embarrassingly parallel, so vmap beats a sequential
# chain for finding the global basin. Basin hopping then refines from the best.
minimise_batch = eqx.filter_jit(jax.vmap(minimise))


def basin_hop(params, key, n_hops=N_HOPS, step=HOP_SIZE):
    """Perturb the incumbent, re-minimise, Metropolis-accept on local minima.

    optax has no basin-hopping transformation: it supplies the local solver and
    this outer accept/reject loop wraps it.
    """
    params, value, _ = minimise(params)
    best, best_value = params, value
    temperature = float(value)   # accept scale, set by the starting depth
    accepted = 0

    for i in range(n_hops):
        key, hop_key, acc_key = jax.random.split(key, 3)
        subkeys = jax.random.split(hop_key, len(CAL_PARAMS))
        trial = {k: params[k] + step * jax.random.normal(subkeys[j])
                 for j, k in enumerate(CAL_PARAMS)}
        trial, trial_value, _ = minimise(trial)

        if trial_value < best_value:
            best, best_value = trial, trial_value

        # Metropolis criterion applied to the *minima*, not to raw samples
        downhill = trial_value < value
        accept_p = np.exp(-(trial_value - value) / max(temperature, 1e-30))
        if downhill or float(jax.random.uniform(acc_key)) < float(accept_p):
            params, value = trial, trial_value
            accepted += 1

    return best, best_value, accepted


power_guess = system.forward(stokes_in)[..., 0]

# Stage 1: parallel random restarts to map the basins.
key = jax.random.PRNGKey(0)
start_keys = jax.random.split(key, len(CAL_PARAMS))
starts = {
    "psg_retarder.retardance": jax.random.uniform(
        start_keys[0], (N_STARTS,), minval=0.05, maxval=np.pi),
    "psa_retarder.retardance": jax.random.uniform(
        start_keys[1], (N_STARTS,), minval=0.05, maxval=np.pi),
    "psg_retarder.offset": jax.random.uniform(
        start_keys[2], (N_STARTS,), minval=-np.pi / 2, maxval=np.pi / 2),
    "psa_retarder.offset": jax.random.uniform(
        start_keys[3], (N_STARTS,), minval=-np.pi / 2, maxval=np.pi / 2),
    "psa_polarizer.transmission_axis": jax.random.uniform(
        start_keys[4], (N_STARTS,), minval=-np.pi / 2, maxval=np.pi / 2),
}
print(f"Running {N_STARTS} parallel L-BFGS restarts...")
batch_params, batch_values, _ = minimise_batch(starts)
finite = np.isfinite(batch_values)
best_start = int(np.argmin(np.where(finite, batch_values, np.inf)))
params = {k: v[best_start] for k, v in batch_params.items()}
print(f"  best restart loss: {float(batch_values[best_start]):.6e}")

# Stage 2: basin hopping from the best restart.
print(f"Basin hopping ({N_HOPS} hops)...")
params, best_value, accepted = basin_hop(params, jax.random.PRNGKey(1))
print(f"  best loss {float(best_value):.6e}   accepted {accepted}/{N_HOPS}")

# Loss history of the final local solve, for the training-history plot
_, _, losses = minimise(params)

# Fold the calibrated parameters back into the model
system = system.set(params)
power_after = system.forward(stokes_in)[..., 0]


def canonicalise_retarder(offset, retardance):
    """Fold a fitted (offset, retardance) pair onto its canonical branch.

    A linear retarder's Mueller matrix is invariant under three
    transformations, so the optimizer is free to land on any of them:

      * retardance -> retardance + 2*pi      (it enters only as sin/cos)
      * offset     -> offset + pi            (the axis enters as 2*theta)
      * retardance -> -retardance  TOGETHER WITH  offset -> offset + pi/2

    Note that retardance is *not* pi-periodic on its own; only the axis is.
    The third symmetry is the one that makes a quarter-wave plate report as
    -pi/2 or 3*pi/2. Canonicalising puts retardance in [0, pi], where a QWP
    should read ~pi/2. Without this the fit prints physically meaningless
    negative retardances even when it has converged perfectly well.

    One degeneracy survives this and cannot be removed here: rotating BOTH
    retarder offsets by pi/2 together leaves the measured intensities exactly
    unchanged, so (offset_psg, offset_psa) is only pinned up to a common pi/2.
    Breaking that needs an external reference. It does not affect retardance.
    """
    retardance = (retardance + np.pi) % (2 * np.pi) - np.pi   # -> [-pi, pi)
    if retardance < 0:
        retardance, offset = -retardance, offset + np.pi / 2
    offset = (offset + np.pi / 2) % np.pi - np.pi / 2         # -> [-pi/2, pi/2)
    return float(offset), float(retardance)


def show(name, value):
    print(f"  {name:<38s} {value:.6f} rad ({np.degrees(value):.4f} deg)")


print("Calibrated DRRP parameters (canonical branch):")
for stage in ("psg_retarder", "psa_retarder"):
    offset, retardance = canonicalise_retarder(
        float(params[f"{stage}.offset"]),
        float(params[f"{stage}.retardance"]),
    )
    show(f"{stage}.retardance", retardance)
    show(f"{stage}.offset", offset)

for name in CAL_PARAMS:
    if name.endswith("transmission_axis"):
        # A polarizer axis is pi-periodic on its own.
        show(name, (float(params[name]) + np.pi / 2) % np.pi - np.pi / 2)

# Rescale the model onto the photodiode units for plotting
gain = np.sum(power_after * power_measured) / np.sum(power_after ** 2)

plt.figure(figsize=[10, 4])
plt.subplot(121)
plt.title("Training History")
plt.plot(losses)
plt.xlabel("L-BFGS Iteration")
plt.ylabel("Loss Function")
plt.yscale("log")
plt.subplot(122)
plt.title("Power Calibration")
plt.plot(psg_angles, power_measured, marker="o", linestyle="None", label="Measured")
plt.plot(psg_angles, gain * power_after, linestyle="dashed", label="Calibrated")
plt.xlabel("PSG angle, rad")
plt.ylabel("power observed, a.u.")
plt.legend()
plt.show()
