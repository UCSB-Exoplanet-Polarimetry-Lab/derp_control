import numpy as tnp
import jax.numpy as jnp
from jax import jit
import ipdb
from katsu.katsu_math import np, broadcast_kron
from katsu.mueller import linear_retarder, linear_polarizer, linear_diattenuator
from katsu.polarimetry import drrp_data_reduction_matrix

from prysm.coordinates import make_xy_grid, cart_to_polar
from prysm.polynomials import noll_to_nm, sum_of_2d_modes
# from .zernike import zernike_nm_seq
# Handle different prysm naming
try:
    from prysm.polynomials import zernike_nm_seq
except ImportError:
    from prysm.polynomials import zernike_nm_sequence as zernike_nm_seq

def jax_sum_of_2d_modes(modes, weights):
    """a clone of prysm.polynomials sum_of_2d_modes that works when using
    katsu's Jax backend

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
        # do some dimensional handling
        if modes.ndim == 4:

            # Need to re-shape array to make it tensordot friendly
            modes = np.asarray(modes)
            modes = np.swapaxes(modes, 0, 1)

            return jax_sum_of_2d_modes(modes, weights)
        else:
            return jax_sum_of_2d_modes(modes, weights)


def create_modal_basis(num_modes, num_pix, radial_offset=0, angle_offset=0):
    """Generates a zernike polynomial basis

    Parameters
    ----------
    num_modes : int
        number of zernike modes to use in the decomposition.
        Includes the piston term at element zero, and goes up
        to noll index = num_modes + 1
    num_pix : int
        number of samples to use across the array
    radial_offset : float or ndarray
        radial offset for the angular coordinate of the basis
    angle_offset : float or ndarray
        angle offset for the angular coordinate of the basis

    Returns
    -------
    list
        zernike polynomials ordered by noll index evaluated on
        an array of `num_pix` samples

    """
    # logic to parse which is bigger and broadcast
    if np.isscalar(radial_offset) and np.isscalar(angle_offset):
        pass
    elif np.isscalar(radial_offset) and not np.isscalar(angle_offset):
        radial_offset = np.full_like(angle_offset, radial_offset)
    elif not np.isscalar(radial_offset) and np.isscalar(angle_offset):
        angle_offset = np.full_like(radial_offset, angle_offset)
    else:
        assert radial_offset.shape == angle_offset.shape, "If both radial and angle offsets are arrays, they must have the same shape"

    # assume a unit disk
    x, y = make_xy_grid(num_pix, diameter=2)
    r, t = cart_to_polar(x, y)
    t = t + angle_offset
    r = r + radial_offset

    # build the polynomials
    # NOTE: num_modes is total number of modes, since we start the Zernike
    # index at 1, we add 1 to the end
    nms = [noll_to_nm(i) for i in range(1, num_modes + 1)]
    basis = list(zernike_nm_seq(nms, r, t))

    return basis

@jit
def psg_psa_states_broadcast(x0, basis_psg, basis_psa, psg_angles, rotation_ratio=2.5, psa_angles=None, psa_offset=0):
    """
    Constructs the Mueller states for the given parameters, broadcast for more efficient computation.

    Parameters
    ----------
    x0 : ndarray
        initial coefficients for the forward model
    basis : list of ndarrays
        modal basis used in the forward model
    psg_angles : ndarray
        angles of the PSG waveplate
    rotation_ratio : float, optional
        ratio of PSA to PSG angles, by default 2.5
    psa_angles : ndarray, optional
        angles of the PSA polarizers, by default None. Setting this kwarg
        overrides the rotation_ratio.
    psa_offset : float, optional
        Offset to apply to the analyzer in radians. Useful for Dual-I-inversion.

    Returns
    -------
    ndarray
        list of Mueller matrices constructed from the given parameters evaluated
        at the given psg and psa angles.
    """

    # extract the front elements that contain the polarizer angles
    dc_power_term = x0[0]
    psg_pol_angle = x0[1]
    psa_pol_angle = x0[2] + psa_offset

    npix = basis_psg[0].shape[0]
    nmodes = basis_psg.shape[1]

    # Generate a new basis for each of the angular offsets
    offset = 3
    psg_wvp_coeffs = x0[offset + 0 * nmodes : offset + 1 * nmodes]
    psa_wvp_coeffs = x0[offset + 1 * nmodes : offset + 2 * nmodes]
    psg_ang_coeffs = x0[offset + 2 * nmodes : offset + 3 * nmodes]
    psa_ang_coeffs = x0[offset + 3 * nmodes : offset + 4 * nmodes]

    # Re-generate the basis at each iteration - pre-allocate large arrays to store the basis
    basis_withrotations_psa = [ Construct Calibration Basis
    for offset_psg, offset_psa in zip(psg_angles, psa_angles):

        # offset is in radians to be compatible with prysm angles
        basis = create_modal_basis(nmodes, npix, radial_offset=drg, angle_offset=offset_psg + dtg)
        basis_masked = np.asarray(basis)
        basis_withrotations_psg.append(basis_masked)

        basis = create_modal_basis(nmodes, npix, radial_offset=dra, angle_offset=offset_psa + dta)
        basis_masked = np.asarray(basis)
        basis_withrotations_psa.append(basis_masked)


    # Adding support for PSA diattenuation which DO NOT ROTATE
    # psa_dia_coeffs = x0[offset + 4 * nmodes : offset + 5 * nmodes]
    # psa_dia_coeffs_ret = x0[offset + 5 * nmodes : offset + 6 * nmodes]
    # psa_dia_coeffs_ang = x0[offset + 6 * nmodes : offset + 7 * nmodes]

    # Good to make sure we are splitting the list correctly
    assert len(psg_wvp_coeffs) == nmodes
    assert len(psa_wvp_coeffs) == nmodes

    # Computes from a rotation ratio if PSA angles not supplied
    if psa_angles is None:
        psa_angles = rotation_ratio * psg_angles

    # Begin the construction of power frames
    PSAs = []
    PSGs = []

    # Construct the retardance estimation
    # TODO: Implement more user-friendly way of doing this
    # Current basis shape
    #    0: Angle position
    #    1: Mode index
    #    2: NPIX
    #    3: NPIX
    basis_npix = basis_psg.shape[-1] # grab last element
    basis_nmode = nmodes

    # Assemble retarders at various rotations
    psg_ret = sum_of_2d_modes_wrapper(basis_psg, psg_wvp_coeffs)
    psa_ret = sum_of_2d_modes_wrapper(basis_psa, psa_wvp_coeffs)

    psg_ang = sum_of_2d_modes_wrapper(basis_psg, psg_ang_coeffs)
    psa_ang = sum_of_2d_modes_wrapper(basis_psa, psa_ang_coeffs)

    # grab first element of the basis
    # psa_dia = sum_of_2d_modes_wrapper(basis_psa[0], psa_dia_coeffs)

    # Npix x Npix x Nangle
    # NOTE: psg/psa ang here are an artifact from before the basis pre-computation for each angle
    psg_angles = psg_ang + psg_angles[..., None, None] #+ psg_ang[..., None]
    psa_angles = psa_ang + psa_angles[..., None, None] #+ psa_ang[..., None]

    # Fixed quantity
    # wollaston_ret = sum_of_2d_modes_wrapper(basis_psa, psa_dia_coeffs_ret)[0]
    # wollaston_ang = sum_of_2d_modes_wrapper(basis_psa, psa_dia_coeffs_ang)[0]
    psg_pol = linear_polarizer(psg_pol_angle)
    psa_pol = linear_polarizer(psa_pol_angle)
    # psa_pol = linear_polarizer(psa_pol_angle) @ linear_retarder(wollaston_ang, wollaston_ret, shape=[*psg_angles.shape])
    # psa_pol = linear_diattenuator(psa_dia, psa_pol_angle, shape=[*psg_angles.shape])

    # I believe this rotates
    psg_wvp = linear_retarder(psg_angles, psg_ret, shape=[*psg_angles.shape])
    psa_wvp = linear_retarder(psa_angles, psa_ret, shape=[*psa_angles.shape])

    PSGs = psg_wvp @ psg_pol
    PSAs = psa_pol @ psa_wvp
    PSGs = np.moveaxis(PSGs, 0, 2)
    PSAs = np.moveaxis(PSAs, 0, 2)
    return PSGs, PSAs, dc_power_term


def mueller_state(x0, basis_psg, basis_psa, psg_angles, rotation_ratio=2.5, psa_angles=None):
    """
    Constructs the Mueller states for the given parameters

    Parameters
    ----------
    x0 : ndarray
        initial coefficients for the forward model
    basis : list of ndarrays
        modal basis used in the forward model
    psg_angles : ndarray
        angles of the PSG waveplate
    rotation_ratio : float, optional
        ratio of PSA to PSG angles, by default 2.5
    psa_angles : ndarray, optional
        angles of the PSA polarizers, by default None. Setting this kwarg
        overrides the rotation_ratio.

    Returns
    -------
    ndarray
        list of Mueller matrices constructed from the given parameters evaluated
        at the given psg and psa angles.
    """

    # get the PSG and PSA states
    PSGs, PSAs, dc_power = psg_psa_states_broadcast(x0, basis_psg, basis_psa, psg_angles,
                                rotation_ratio=rotation_ratio,
                                psa_angles=psa_angles)

    mueller_states = PSAs @ PSGs

    return mueller_states, dc_power


def dual_I_mueller_state(x0, basis_psg, basis_psa, psg_angles, rotation_ratio=2.5, psa_angles=None):

    # Left Channel, pass DC power to void register
    PSGs_L, PSAs_L, _ = psg_psa_states_broadcast(x0, basis_psg, basis_psa, psg_angles,
                                rotation_ratio=rotation_ratio,
                                psa_angles=psa_angles)

    # Right Channel analyzer is rotated by 90 deg
    PSGs_R, PSAs_R, _ = psg_psa_states_broadcast(x0, basis_psg, basis_psa, psg_angles,
                                rotation_ratio=rotation_ratio,
                                psa_angles=psa_angles,
                                psa_offset=np.radians(90))

    # Concatenate, first dimension before the Mueller matrix dimensions
    PSGs = np.concatenate([PSGs_L, PSGs_R], axis=-3)
    PSAs = np.concatenate([PSAs_L, PSAs_R], axis=-3)
    mueller_states = PSAs @ PSGs
    return mueller_states

def forward_model(x0, basis_psg, basis_psa, psg_angles, rotation_ratio=2.5, psa_angles=None, dual_I=False):
    """Forward model for simulating the power frames

    Parameters
    ----------
    x0 : ndarray
        initial coefficients for the forward model
    basis : list of ndarrays
        modal basis used in the forward model
    psg_angles : ndarray
        angles of the PSG waveplate
    rotation_ratio : float, optional
        ratio of PSA to PSG angles, by default 2.5
    psa_angles : ndarray, optional
        angles of the PSA polarizers, by default None. Setting this kwarg
        overrides the rotation_ratio.

    Returns
    -------
    ndarray
        simulated power frames from the forward model
    """

    if not dual_I:
        mueller_states, dc_power = mueller_state(x0,
                                       basis_psg,
                                       basis_psa,
                                       psg_angles,
                                       rotation_ratio,
                                       psa_angles=psa_angles)
    else:
        mueller_states, dc_power = dual_I_mueller_state(x0,
                                       basis_psg,
                                       basis_psa,
                                       psg_angles,
                                       rotation_ratio,
                                       psa_angles)

    simulated_frames = mueller_states[..., 0, 0] * dc_power

    return simulated_frames


def make_data_reduction_matrix(x0, basis_psg, basis_psa, psg_angles,
                               rotation_ratio=2.5, psa_angles=None, dual_I=False):
    """Creates a data reduction matrix for the given parameters

    Parameters
    ----------
    x0 : ndarray
        initial coefficients for the forward model
    basis : list of ndarrays
        modal basis used in the forward model
    psg_angles : ndarray
        angles of the PSG polarizers
    rotation_ratio : float, optional
        ratio of PSA to PSG angles, by default 2.5
    psa_angles : ndarray, optional
        angles of the PSA polarizers, by default None

    Returns
    -------
    ndarray
        data reduction matrix to compute the system Mueller matrix from power
        measurements
    """
    # get the PSG and PSA states, DC power goes to void register
    if not dual_I:
        PSGs, PSAs, _ = psg_psa_states_broadcast(x0, basis_psg, basis_psa, psg_angles,
                                    rotation_ratio=rotation_ratio,
                                    psa_angles=psa_angles)
    else:
        # Left Channel
        PSGs_L, PSAs_L, _ = psg_psa_states_broadcast(x0, basis_psg, basis_psa, psg_angles,
                                    rotation_ratio=rotation_ratio,
                                    psa_angles=psa_angles)

        # Right Channel analyzer is rotated by 90 deg
        PSGs_R, PSAs_R, _ = psg_psa_states_broadcast(x0, basis_psg, basis_psa, psg_angles,
                                    rotation_ratio=rotation_ratio,
                                    psa_angles=psa_angles,
                                    psa_offset=np.radians(90))

        # Concatenate, first dimension before the Mueller matrix dimensions
        PSGs = np.concatenate([PSGs_L, PSGs_R], axis=-3)
        PSAs = np.concatenate([PSAs_L, PSAs_R], axis=-3)

    data_reduction_matrix = drrp_data_reduction_matrix(PSGs, PSAs, invert=True)

    return data_reduction_matrix
