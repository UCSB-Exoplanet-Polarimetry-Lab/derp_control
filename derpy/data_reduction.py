from astropy.io import fits
from katsu.katsu_math import broadcast_kron, np
from katsu.mueller import linear_polarizer, linear_retarder, linear_diattenuator
from skimage.registration import phase_cross_correlation
from scipy.ndimage import center_of_mass, shift
import ipdb
import os
import json
import matplotlib.pyplot as plt

from .gui import launch_image_selector
from .centering import robust_circle_fit
from .binning import bin_array_2d

def _measure_from_experiment(experiment, channel="both", frame_mask=None,
                            wavelength=None):
    """Measure a Mueller Matrix from an Experiment

    Parameters
    ----------
    experiment : Experiment
        _description_
    channel : str, optional
        Which subaperture of Derp to use, accepts "right", "left", or "both",
        by default "both"
    frame_mask : list or array of bool, optional
        indices to ignore in the data reduction. Values of 1 are kept,
        values of 0 are removed. by default None, which doesn't mask anything
    wavelength : int, optional
        wavelength to perform the data reduction, by default None

    Returns
    -------
    ndarray
        Mueller Matrix
    """

    # Get the cuts
    cxl = experiment.cxl
    cyl = experiment.cyl
    cxr = experiment.cxr
    cyr = experiment.cyr
    cut = experiment.cut

    # Where it currently is
    starting_angle_psg_pol = experiment.psg_pol_angle
    starting_angle_psg_wvp = np.radians(experiment.psg_starting_angle)
    retardance_psg_wvp = experiment.psg_wvp_ret

    starting_angle_psa_pol = experiment.psa_pol_angle
    starting_angle_psa_wvp = np.radians(experiment.psa_starting_angle)
    retardance_psa_wvp = experiment.psa_wvp_ret

    # Construct the position arrays
    unmasked_psg_angles = np.radians(np.array(experiment.psg_positions_relative)) + starting_angle_psg_wvp
    unmasked_psa_angles = np.radians(np.array(experiment.psa_positions_relative)) + starting_angle_psa_wvp

    # Mask bad images
    if frame_mask is not None:

        # update images
        if wavelength is None:
            images = []
        else:
            good_indices = [i for i in range(len(experiment.psg_positions_relative)) if i not in frame_mask]
            images = []

        unmasked_images = experiment.images
        images = []
        psg_angles = []
        psa_angles = []
        for i, (dont_mask , psg, psa) in enumerate(zip(frame_mask , unmasked_psg_angles, unmasked_psa_angles)):
            if dont_mask==1:

                psg_angles.append(psg)
                psa_angles.append(psa)

                # get the image
                if wavelength is None:
                    im = unmasked_images[i]
                    images.append(im)
                else:
                    im = unmasked_images[:, i]
                    images.append(im)

        psg_angles = np.array(psg_angles)
        psa_angles = np.array(psa_angles)
        images = np.array(images)
        images = np.swapaxes(images, 0, 1)

    else:
        images = experiment.images
        psg_angles = unmasked_psg_angles
        psa_angles = unmasked_psa_angles

    if wavelength is None:

        # preallocate power
        power = []

        if channel == "left":
            power = []

            for img in images:
                cut_power = img[cxl-cut:cxl+cut, cyl-cut:cyl+cut]
                power.append(cut_power)

        elif channel == "right":
            for img in images:
                cut_power = img[cxr-cut:cxr+cut, cyr-cut:cyr+cut]
                power.append(cut_power)

            starting_angle_psa_pol += np.pi/2

        # Honest to goodness dual_channel
        elif channel == "both":

            for p in range(2):
                for img in images:
                    if p == 0:
                        cut_power = img[cxl-cut:cxl+cut, cyl-cut:cyl+cut]
                    elif p == 1:
                        cut_power = img[cxr-cut:cxr+cut, cyr-cut:cyr+cut]

                    power.append(cut_power)

            # update the psg/psa angles
            psg_angles = np.concatenate([psg_angles, psg_angles])
            psa_angles = np.concatenate([psa_angles, psa_angles])

        # Try out bright-normalized on the left image
        else:

            for i, img in enumerate(images):
                cut_left = img[cxl-cut:cxl+cut, cyl-cut:cyl+cut]

                if i == 0:
                    power_ref = experiment.mean_power_left[i] + experiment.mean_power_right[i]

                # try with left pupil
                p_ref_l = experiment.mean_power_left[i] + experiment.mean_power_right[i]
                p_ref = power_ref / p_ref_l
                cut_power = cut_left * p_ref

                power.append(cut_power)

    else:
        power = []

        if channel == "left":
            for i in range(len(psg_angles)):
                cut_left = images[wavelength, i, 0]

                if i == 0:
                    power_ref = experiment.mean_power_left[wavelength, i] + \
                        experiment.mean_power_right[wavelength, i]

                # try with left pupil
                p_ref_l = experiment.mean_power_left[wavelength, i] + \
                        experiment.mean_power_right[wavelength, i]
                p_ref = power_ref / p_ref_l
                cut_power = cut_left * p_ref

                power.append(cut_power)

        if channel == "right":
            for i in range(len(psg_angles)):
                cut_right = images[wavelength, i, 1]

                if i == 0:
                    power_ref = experiment.mean_power_left[wavelength, i] + \
                        experiment.mean_power_right[wavelength, i]

                # try with left pupil
                p_ref_l = experiment.mean_power_left[wavelength, i] + \
                        experiment.mean_power_right[wavelength, i]
                p_ref = power_ref / p_ref_l
                cut_power = cut_right * p_ref

                power.append(cut_power)

        elif channel == "both":
            power_left = []
            power_right = []
            for p in range(2):

                # start with the left pupil
                if p == 0:
                    for i in range(len(psg_angles)):
                        cut_left = images[wavelength, i, 0]

                        if i == 0:
                            power_ref = experiment.mean_power_left[wavelength, i] + \
                                experiment.mean_power_right[wavelength, i]

                        # try with left pupil
                        p_ref_l = experiment.mean_power_left[wavelength, i] + \
                                experiment.mean_power_right[wavelength, i]
                        p_ref = power_ref / p_ref_l
                        cut_power = cut_left * p_ref

                        power_left.append(cut_power)

                elif p == 1:
                    for i in range(len(psg_angles)):
                        cut_right = images[wavelength, i, 1]

                        if i == 0:
                            power_ref = experiment.mean_power_left[wavelength, i] + \
                                experiment.mean_power_right[wavelength, i]

                        # try with left pupil
                        p_ref_l = experiment.mean_power_left[wavelength, i] + \
                                experiment.mean_power_right[wavelength, i]
                        p_ref = power_ref / p_ref_l
                        cut_power = cut_right * p_ref

                        power_right.append(cut_power)

            power = np.concatenate([power_left, power_right])

            # update the angles
            psg_angles = np.concatenate([psg_angles, psg_angles])
            psa_angles = np.concatenate([psa_angles, psa_angles])


    power = np.asarray(power)

    shapes = [*power.shape[-2:], psa_angles.shape[0]]
    shapes_half = [*power.shape[-2:], psa_angles.shape[0]//2]
    power = np.moveaxis(power,0,-1)

    psg_pol = linear_polarizer(starting_angle_psg_pol)
    psg_wvp = linear_retarder(psg_angles, retardance_psg_wvp, shape=shapes)

    psa_wvp = linear_retarder(psa_angles, retardance_psa_wvp, shape=shapes)

    if channel == "both":
        psa_pol_l = linear_polarizer(starting_angle_psa_pol, shape=shapes_half)
        psa_pol_r = linear_polarizer(starting_angle_psa_pol + np.pi/2, shape=shapes_half)
        psa_pol = np.concatenate([psa_pol_l, psa_pol_r], axis=-3)

    elif channel == "left":
        psa_pol = linear_polarizer(starting_angle_psa_pol)
    elif channel == "right":
        psa_pol = linear_polarizer(starting_angle_psa_pol + np.pi / 2)

    Mg = psg_wvp @ psg_pol
    Ma = psa_pol @ psa_wvp

    PSA = Ma[..., 0, :]
    PSG = Mg[..., :, 0]

    # polarimetric data reduction matrix, flatten Mueller matrix dimension
    Wmat = broadcast_kron(PSA[..., np.newaxis], PSG[..., np.newaxis])
    Wmat = Wmat.reshape([*Wmat.shape[:-2], 16])
    Winv = np.linalg.pinv(Wmat)
    power_expand = power[..., np.newaxis]

    # Do the data reduction
    M_meas = Winv @ power_expand
    M_meas = M_meas[..., 0]

    return M_meas.reshape([*M_meas.shape[:-1], 4, 4])


def _measure_from_experiment_old(experiment, channel="both", frame_mask=None):

    # Get the cuts
    cxl = experiment.cxl
    cyl = experiment.cyl
    cxr = experiment.cxr
    cyr = experiment.cyr
    cut = experiment.cut

    # Where it currently is
    starting_angle_psg_pol = experiment.psg_pol_angle
    starting_angle_psg_wvp = np.radians(experiment.psg_starting_angle)
    retardance_psg_wvp = experiment.psg_wvp_ret

    starting_angle_psa_pol = experiment.psa_pol_angle
    starting_angle_psa_wvp = np.radians(experiment.psa_starting_angle)
    retardance_psa_wvp = experiment.psa_wvp_ret

    # Construct the position arrays
    unmasked_psg_angles = np.radians(np.array(experiment.psg_positions_relative)) + starting_angle_psg_wvp
    unmasked_psa_angles = np.radians(np.array(experiment.psa_positions_relative)) + starting_angle_psa_wvp

    # Mask bad images
    if frame_mask is not None:

        # update images
        unmasked_images = experiment.images
        images = []
        psg_angles = []
        psa_angles = []

        for dont_mask, im, psg, psa in zip(frame_mask, unmasked_images, unmasked_psg_angles, unmasked_psa_angles):
            if dont_mask:
                images.append(im)
                psg_angles.append(psg)
                psa_angles.append(psa)
        psg_angles = np.array(psg_angles)
        psa_angles = np.array(psa_angles)

    else:
        images = experiment.images
        psg_angles = unmasked_psg_angles
        psa_angles = unmasked_psa_angles

    # preallocate power
    power = []


    if channel == "left":
        power = []
        for img in images:
            cut_power = img[0]
            power.append(cut_power)

    elif channel == "right":
        for img in images:
            cut_power = img[1]
            power.append(cut_power)

        starting_angle_psa_pol += np.pi/2

    # Honest to goodness dual_channel
    elif channel == "both":

        for p in range(2):
            for img in images:
                if p == 0:
                    cut_power = img[0]

                elif p == 1:
                    cut_power = img[1]

                power.append(cut_power)

        # update the psg/psa angles
        psg_angles = np.concatenate([psg_angles, psg_angles])
        psa_angles = np.concatenate([psa_angles, psa_angles])

    # Try out bright-normalized on the left image
    else:


        for i, img in enumerate(images):
            cut_left = img[0]

            if i == 0:
                power_ref = experiment.mean_power_left[i] + experiment.mean_power_right[i]

            # try with left pupil
            p_ref_l = experiment.mean_power_left[i] + experiment.mean_power_right[i]
            p_ref = power_ref / p_ref_l
            cut_power = cut_left * p_ref

            power.append(cut_power)


    power = np.asarray(power)
    shapes = [*power.shape[-2:], psa_angles.shape[0]]
    shapes_half = [*power.shape[-2:], psa_angles.shape[0]//2]
    power = np.moveaxis(power,0,-1)

    psg_pol = linear_polarizer(starting_angle_psg_pol)
    psg_wvp = linear_retarder(psg_angles, retardance_psg_wvp, shape=shapes)

    psa_wvp = linear_retarder(psa_angles, retardance_psa_wvp, shape=shapes)

    if channel == "both":
        psa_pol_l = linear_polarizer(starting_angle_psa_pol, shape=shapes_half)
        psa_pol_r = linear_polarizer(starting_angle_psa_pol + np.pi/2, shape=shapes_half)
        psa_pol = np.concatenate([psa_pol_l, psa_pol_r], axis=-3)

    else:
        psa_pol = linear_polarizer(starting_angle_psa_pol)

    Mg = psg_wvp @ psg_pol
    Ma = psa_pol @ psa_wvp

    PSA = Ma[..., 0, :]
    PSG = Mg[..., :, 0]

    # polarimetric data reduction matrix, flatten Mueller matrix dimension
    Wmat = broadcast_kron(PSA[..., np.newaxis], PSG[..., np.newaxis])
    Wmat = Wmat.reshape([*Wmat.shape[:-2], 16])
    Winv = np.linalg.pinv(Wmat)
    power_expand = power[..., np.newaxis]

    # Do the data reduction
    M_meas = Winv @ power_expand
    M_meas = M_meas[..., 0]

    return M_meas.reshape([*M_meas.shape[:-1], 4, 4])


def _measure_from_experiment_polychromatic(experiment, channel="both",
                                          frame_mask=None, wavelength=0):

    # in-line replace images
    images = experiment.images[wavelength]
    experiment.images = images

    M = measure_from_experiment_old(experiment, channel=channel, frame_mask=frame_mask)

    return M


def measure_from_images(data, psg_angles, psa_angles,
                        pol_angle_g, pol_angle_a,
                        ret_angle_g, ret_angle_a):

    power = np.asarray(data)
    shapes = [*power.shape[-2:], psa_angles.shape[0]]
    shapes_half = [*power.shape[-2:], psa_angles.shape[0]//2]
    power = np.moveaxis(power,0,-1)

    psg_pol = linear_polarizer(pol_angle_g)
    psg_wvp = linear_retarder(psg_angles, ret_angle_g, shape=shapes)

    psa_wvp = linear_retarder(psa_angles, ret_angle_a, shape=shapes)

    # this means its a list
    if hasattr(pol_angle_a, "append"):
        starting_angle_psa_pol_l = pol_angle_a[0]
        starting_angle_psa_pol_r = pol_angle_a[1]
        psa_pol_l = linear_polarizer(starting_angle_psa_pol_l, shape=shapes_half)
        psa_pol_r = linear_polarizer(starting_angle_psa_pol_r, shape=shapes_half)
        psa_pol = np.concatenate([psa_pol_l, psa_pol_r], axis=-3)

    else:
        psa_pol = linear_polarizer(pol_angle_a)

    Mg = psg_wvp @ psg_pol
    Ma = psa_pol @ psa_wvp

    PSA = Ma[..., 0, :]
    PSG = Mg[..., :, 0]

    # polarimetric data reduction matrix, flatten Mueller matrix dimension
    Wmat = broadcast_kron(PSA[..., np.newaxis], PSG[..., np.newaxis])
    Wmat = Wmat.reshape([*Wmat.shape[:-2], 16])
    Winv = np.linalg.pinv(Wmat)
    power_expand = power[..., np.newaxis]

    # Do the data reduction
    M_meas = Winv @ power_expand
    M_meas = M_meas[..., 0]

    return M_meas.reshape([*M_meas.shape[:-1], 4, 4])


def mask_bad_data(experiment, frame_mask, wavelength_index, channel):

    # update images
    if channel == "both":
        unmasked_images_l = experiment.images[wavelength_index, :, 0]
        unmasked_images_r = experiment.images[wavelength_index, :, 1]
        unmasked_images = np.concatenate([unmasked_images_l, unmasked_images_r])

        starting_angle_psg_wvp = np.radians(experiment.psg_starting_angle)
        starting_angle_psa_wvp = np.radians(experiment.psa_starting_angle)

        unmasked_psg_angles = np.radians(np.array(experiment.psg_positions_relative)) + starting_angle_psg_wvp
        unmasked_psa_angles = np.radians(np.array(experiment.psa_positions_relative)) + starting_angle_psa_wvp

        unmasked_psg_angles = np.concatenate([unmasked_psg_angles, unmasked_psg_angles])
        unmasked_psa_angles = np.concatenate([unmasked_psa_angles, unmasked_psa_angles])

        frame_mask = np.concatenate([frame_mask, frame_mask])

        mean_power_left = experiment.mean_power_left[wavelength_index]
        mean_power_right = experiment.mean_power_right[wavelength_index]
        mean_power_left = np.concatenate([mean_power_left, mean_power_left])
        mean_power_right = np.concatenate([mean_power_right, mean_power_right])

    else:
        if channel == "left":
            i = 0
        elif channel == "right":
            i = 1

        unmasked_images = experiment.images[wavelength_index, :, i]

        # Where it currently is
        starting_angle_psg_wvp = np.radians(experiment.psg_starting_angle)
        starting_angle_psa_wvp = np.radians(experiment.psa_starting_angle)
        unmasked_psg_angles = np.radians(np.array(experiment.psg_positions_relative)) + starting_angle_psg_wvp
        unmasked_psa_angles = np.radians(np.array(experiment.psa_positions_relative)) + starting_angle_psa_wvp

        mean_power_left = experiment.mean_power_left[wavelength_index]
        mean_power_right = experiment.mean_power_right[wavelength_index]

    images = []
    psg_angles = []
    psa_angles = []
    good_frame_counter = 0
    for i, (im, dont_mask, psg, psa) in enumerate(zip(unmasked_images, frame_mask , unmasked_psg_angles, unmasked_psa_angles)):

        if dont_mask==1:

            # save the first good frame as the reference
            if good_frame_counter == 0:

                # get the reference power
                power_ref = mean_power_left[i] + mean_power_right[i]

            # perform the intensity correction
            p_ref_l = mean_power_left[i] + mean_power_right[i]
            p_ref = power_ref / p_ref_l

            im_corrected = im * p_ref

            psg_angles.append(psg)
            psa_angles.append(psa)
            images.append(im_corrected)

            good_frame_counter += 1

    psg_angles = np.array(psg_angles)
    psa_angles = np.array(psa_angles)
    images = np.array(images)

    return images, psg_angles, psa_angles


def mueller_from_experiment(experiment, channel="left", frame_mask=None):

    # do a mueller matrix measurement for each wavelength
    mueller_matrices = []
    for i, wvl in enumerate(experiment.wavelengths):

        # remove bad frames + associated angles from the image
        data, psg_angles, psa_angles = mask_bad_data(experiment, frame_mask, i, channel)

        pol_angle_g = experiment.psg_pol_angle
        pol_angle_a = experiment.psa_pol_angle

        # subtract off the analyzer angle to align polarimeter into the laboratory frame
        # NOTE: This assumes that the left channel analyzes the horizontal
        pol_angle_g -= pol_angle_a
        psg_angles -= pol_angle_a
        psa_angles -= pol_angle_a
        pol_angle_a -= pol_angle_a

        if channel == "right":
            pol_angle_a += np.pi/2

        elif channel == "both":
            pol_angle_a = [pol_angle_a, pol_angle_a + np.pi/2]

        ret_angle_g = experiment.psg_wvp_ret
        ret_angle_a = experiment.psa_wvp_ret

        M = measure_from_images(data, psg_angles, psa_angles,
                                pol_angle_g, pol_angle_a,
                                ret_angle_g, ret_angle_a)

        mueller_matrices.append(M)

    return mueller_matrices


def q_continuum_from_experiment(experiment, channel="dual"):

    # Get the cuts
    cxl = experiment.cxl
    cyl = experiment.cyl
    cxr = experiment.cxr
    cyr = experiment.cyr
    cut = experiment.cut

    # Where it currently is
    offset = experiment.psa_pol_angle
    starting_angle_psg_pol = experiment.psg_pol_angle - offset
    starting_angle_psg_wvp = np.radians(experiment.psg_starting_angle) - offset
    retardance_psg_wvp = experiment.psg_wvp_ret

    starting_angle_psa_pol = experiment.psa_pol_angle - offset
    starting_angle_psa_wvp = np.radians(experiment.psa_starting_angle) - offset
    retardance_psa_wvp = experiment.psa_wvp_ret

    # Construct the position arrays
    psg_angles = np.radians(np.array(experiment.psg_positions_relative)) + starting_angle_psg_wvp
    psa_angles = np.radians(np.array(experiment.psa_positions_relative)) + starting_angle_psa_wvp


    # Honest to goodness dual_channel
    images = experiment.images
    power_left = []
    power_right = []
    Q = []

    # extract Q next
    for img in images:
        cut_power_left = img[cxl-cut:cxl+cut, cyl-cut:cyl+cut]
        cut_power_right = img[cxr-cut:cxr+cut, cyr-cut:cyr+cut]

        # 3 data points per measurement
        I = cut_power_left + cut_power_right
        power_left.append(cut_power_left / I)
        power_right.append(cut_power_right / I)
        Q.append((cut_power_left - cut_power_right) / I)

    ## SET UP INDIVIDUAL I-Inversions
    # update the psg/psa angles, we will need a boatload of these for independent experiments
    power_left = np.asarray(power_left)
    power_right = np.asarray(power_right)

    shapes = [*power_left.shape[-2:], psa_angles.shape[0]]
    power_left = np.moveaxis(power_left,0,-1)
    power_right = np.moveaxis(power_right,0,-1)

    psg_pol = linear_polarizer(starting_angle_psg_pol)
    psg_wvp = linear_retarder(psg_angles, retardance_psg_wvp, shape=shapes)

    psa_wvp = linear_retarder(psa_angles, retardance_psa_wvp, shape=shapes)

    psa_pol_l = linear_polarizer(starting_angle_psa_pol, shape=shapes)
    psa_pol_r = linear_polarizer(starting_angle_psa_pol + np.pi/2, shape=shapes)

    Mg = psg_wvp @ psg_pol
    Ma_l = psa_pol_l @ psa_wvp
    Ma_r = psa_pol_r @ psa_wvp

    PSA_L = Ma_l[..., 0, :]
    PSA_R = Ma_r[..., 0, :]
    PSG = Mg[..., :, 0]

    # polarimetric data reduction matrix, flatten Mueller matrix dimension
    Wmat_L = broadcast_kron(PSA_L[..., np.newaxis], PSG[..., np.newaxis])
    Wmat_R = broadcast_kron(PSA_R[..., np.newaxis], PSG[..., np.newaxis])
    Wmat_L = Wmat_L.reshape([*Wmat_L.shape[:-2], 16])
    Wmat_R = Wmat_R.reshape([*Wmat_R.shape[:-2], 16])

    # The matrices we want to concatenate pre-inversion
    Winv_L = np.linalg.pinv(Wmat_L)
    Winv_R = np.linalg.pinv(Wmat_R)

    # delete these big matrices to clear up memory
    del Wmat_L, Wmat_R, PSA_L, PSA_R, Ma_l, Ma_r

    ## SET UP Q INVERSION, in the back I guess?
    Q = np.asarray(Q)
    shapes = [*power_left.shape[-2:], psa_angles.shape[0]]
    Q = np.moveaxis(Q,0,-1)

    Ma = psa_wvp

    PSA = Ma[..., 1, :]

    # polarimetric data reduction matrix, flatten Mueller matrix dimension
    Wmat_q = broadcast_kron(PSA[..., np.newaxis], PSG[..., np.newaxis])
    Wmat_q = Wmat_q.reshape([*Wmat_q.shape[:-2], 16])
    Winv_Q = np.linalg.pinv(Wmat_q)

    # pack it all together
    Q_expand = Q[..., np.newaxis]
    power_left_expand = power_left[..., np.newaxis]
    power_right_expand = power_right[..., np.newaxis]

    Winv = np.concatenate([Winv_L, Winv_R, Winv_Q], axis=-1)
    power_expand = np.concatenate([power_left_expand, power_right_expand, Q_expand], axis=-2)

    # Do the data reduction - BIG MATRIX
    M_meas = Winv @ power_expand
    M_meas = M_meas[..., 0]

    return M_meas.reshape([*M_meas.shape[:-1], 4, 4])


# Inherits from github.com/Jashcraf/Spatial_Calibration
def reduce_data(data, centering='circle', mask=None, bin=None, reference_frame=0):
    """ Reduces the data by compensating for source fluctuations and aligning images
    using phase cross-correlation.

    Parameters
    ----------
    data : dict
        A dictionary containi1the following
        keys:
        - "images": numpy array of images.
        - "angles": numpy array of angles corresponding to the images.
        - "powers_left": numpy array of mean powers in the left channel.
        - "powers_right": numpy array of mean powers in the right channel.
        - "powers_total": numpy array of total mean powers (left + right).
        - "reference_channel": index of the reference channel (0 for left, 1 for right).
        - "other_channel": index of the other channel (1 for left, 0 for right).
    centering : str, optional
        Method to use for centering the beam. Either 'circle' or 'com' are supported.
    mask : ndarray, optional
        NaN mask to apply to all images at the end of the reduction.

    Returns
    -------
    images : ndarray
        A numpy array of images after compensation for source fluctuations and alignment.
    """

    assert centering in ['circle', 'com', None], "centering must be either 'circle' or 'com'"

    if "angles" in data.keys():
        angles = data["angles"]
    else:
        angles = data["psg_angles"]

    images = data["images"]
    powers_left = data["powers_left"]
    powers_right = data["powers_right"]
    powers_total = data["powers_total"]
    reference_channel = data["reference_channel"]
    other_channel = data["other_channel"]
    use_photodiode = data["use_photodiode"]

    # Digest the images
    if  images.dtype != np.float32 or images.dtype != np.float64:
        images = images.astype(np.float32)

    # ensure there aren't negative numbers
    images[images <= 0] = 1

    # Compute the circle fit for the reference frame
    if not use_photodiode:
        ref_image = images[reference_frame, reference_channel]
    else:
        ref_image = images[reference_frame]


    # Phase cross-correlation to align the images
    for i, img in enumerate(images):

        if centering == 'circle':

            # Center the reference image using circle fitting
            ref_image, shift_px, circle_params = robust_circle_fit(ref_image)

        elif centering == 'com':
            # Center the reference image using center of mass
            ref_image = center_beam(ref_image)
            circle_params = 0

        # If centering is not supported, don't do anything
        else:
            # Center the reference image using circle fitting
            # Get circle params but pass centered img to void register
            _, shift_px, circle_params = robust_circle_fit(ref_image)
            pass

        # have to do for the left and right channels separately
        if centering is not None:
            if not use_photodiode:
                for channel in range(2):

                    # Calculate the shift between the current image and the reference image
                    shift_y, shift_x = phase_cross_correlation(ref_image,
                                                            images[i, channel],
                                                            upsample_factor=10)[0]

                    # Apply the shift to the current image
                    images[i, channel] = shift(images[i, channel], shift=(shift_y, shift_x), mode='wrap')
            else:
                shift_y, shift_x = phase_cross_correlation(ref_image,
                                                        images[i],
                                                        upsample_factor=10)[0]

                # apply shift to current image
                images[i] = shift(images[i], shift=(shift_y, shift_x), mode='wrap')
    
    # Perform power normalization now that frames are co-registered
    if not use_photodiode:
        images = np.swapaxes(images, 0, 1)
        for i, img in enumerate(images):

            p_ref = img[0] + img[1]  # total power in the frame
            zero_mask = np.ones_like(p_ref, dtype=bool)
            zero_mask[p_ref <= 1e-5] = False

            # Apply the mask to the image
            if mask is not None:
                img = img * mask

            # 1/2 comes from polarizer transmission
            # NOTE: It is CRITICAL that the divide by 2 is an integer
            # Otherwise, this returns a zero if img is dtype="uint16"
            set = img / p_ref / 2
            images[i, 0] = set[0] # [zero_mask]
            images[i, 1] = set[1] # [zero_mask]
    else:
        p_ref_0 = powers_total[0]

        # Find frame where power is maximized
        max_idx = len(powers_total) - int(np.where(powers_total==np.max(powers_total))[0]) - 1
        print(max_idx)

        for i, img in enumerate(images):

            p_ref = powers_total[i]
            zero_mask = np.ones_like(p_ref, dtype=bool)
            zero_mask[p_ref <= 1e-5] = False

            # apply image mask
            if mask is not None:
                img = img * mask
            
            # In the non-photodiode case this has a 1/2, but because we are normalizing to
            # A single frame (instead of a sum) this is /4
            # print(images.shape)
            # print(max_idx)
            # plt.figure()
            # plt.subplot(121)
            # plt.imshow(img)
            # plt.colorbar()
            # plt.subplot(122)
            # plt.title("The Brightest Frame")
            # plt.imshow(images[max_idx] * mask)
            # plt.colorbar()
            # plt.show()
            # np.mean(images[max_idx][mask==1]) /
            set = img * p_ref / p_ref_0 /  4
            images[i] = set

    # Bin the image if binning is specified
    if not use_photodiode:
        if bin is not None:
            binned_images_left = []
            binned_images_right = []
            for i, img in enumerate(images):

                binned_left = bin_array_2d(img[0], bin, method='mean')
                binned_right = bin_array_2d(img[1], bin, method='mean')
                binned_images_left.append(binned_left)
                binned_images_right.append(binned_right)

            images = np.stack([binned_images_left, binned_images_right], axis=0)

        else:
            images = np.swapaxes(images, 0, 1)
    
    # only need to bin left
    else:
        if bin is not None:
            binned_images_left = []
            for i, img in enumerate(images):

                binned_left = bin_array_2d(img, bin, method='mean')
                binned_images_left.append(binned_left)

            images = np.asarray(binned_images_left)

    # returns centered images
    return images, circle_params


def load_fits_data(measurement_pth, calibration_pth,
                   dark_pth=None, use_encoder=False, reference_channel="Left",
                   centering_ref_img=0, use_photodiode=False):
    """load data from .fits file experiments

    Parameters
    ----------
    measurement_pth: str or PosixPath
        Path to the .fits file containing the measured data
    calibration_pth: str or PosixPath
        Path to the .fits file containing the calibration data
    dark_pth: str or PosixPath
        Path to the .fits file containing the dark frame, optional.
        Defaults to None. Currently not supported
    use_encoder: bool
        Whether to use the encoder-read angles instead of those commanded during
        data acquisition, optional. Defaults to False. If True, the measurement
        and calibration .fits files need to have the "PSG_ENCODER_ANGLES" and
        "PSA_ENCODER_ANGLES" ImageHDU.

    Returns
    -------
    dict
        Dictionary keyed by experiment (calibration, measurement) containing the experimental
        data for later data reduction.

    """

    # TODO
    if dark_pth is not None:
        raise ValueError("Optional dark subtraction not yet implemented")

    if isinstance(reference_channel, str):
        assert reference_channel.lower() in ["left", "right"]
        if reference_channel.lower() == "left":
            reference_channel = 0
            other_channel = 1
        else:
            reference_channel = 1
            other_channel = 0

    elif isinstance(reference_channel, int):
        assert reference_channel in [0, 1]
        if reference_channel == 0:
            other_channel = 1
        else:
            other_channel = 0
    else:
        raise ValueError(f"Channel {reference_channel} is not in 'Left'/'Right' or 0/1")

    drrp_raw_data = {}
    pths = [calibration_pth, measurement_pth]
    experiment_keys = ["Calibration", "Measurement"]

    for pth, key in zip(pths, experiment_keys):

        # Load the data
        measurement = fits.open(pth)
        power_measurement = measurement["PSA_IMAGES"].data

        # Subaperture based on the Calibration file
        if key == "Calibration":

            # check to see if there's a path called "image_selection.json"
            if os.path.exists("image_selection.json"):
                with open("image_selection.json", "r") as f:
                    selected_coordinates = json.load(f)
            else:
                selected_areas, selected_coordinates = launch_image_selector(power_measurement[centering_ref_img], use_photodiode)
                # Save the selected areas
                with open("image_selection.json", "w") as f:
                    json.dump(selected_coordinates, f)

        # Use Wollaston for power tracking, requires both frames
        if not use_photodiode:
            x1, y1, x2, y2 = selected_coordinates[0]
            images_left = power_measurement[..., y1:y2, x1:x2]
            powers_left = np.median(images_left, axis=(1, 2))

            x1, y1, x2, y2 = selected_coordinates[1]
            images_right = power_measurement[..., y1:y2, x1:x2]
            powers_right = np.median(images_right, axis=(1, 2))

            # There should be some frame filtering here
            good_powers_right = powers_right
            good_powers_left = powers_left
            good_powers_total = powers_right + powers_left
            print(power_measurement.shape)
            print(images_left.shape)
            print(images_right.shape)
            good_images = np.array([images_left, images_right])
        
        # Use photodiode for power tracking, only pulls frame on left
        else:
            x1, y1, x2, y2 = selected_coordinates[0]
            good_images = power_measurement[..., y1:y2, x1:x2]
            good_powers_left = np.median(good_images, axis=(1, 2))
            good_powers_right = None
            
            # Load the photodiode, median first dimension
            if measurement["PSA_POWER_METER"].data.ndim > 1:
                good_powers_total = np.median(measurement["PSA_POWER_METER"].data, axis=1)
            else:
                good_powers_total = measurement["PSA_POWER_METER"].data

        if not use_encoder:
            psg_angles = measurement["PSG_COMMAND_ANGLES"]
            psa_angles = measurement["PSA_COMMAND_ANGLES"]
        else:
            psg_angles = measurement["PSG_ENCODER_ANGLES"]
            psa_angles = measurement["PSA_ENCODER_ANGLES"]
        

        experiment_data = {
            "images": good_images,
            "psg_angles": psg_angles,
            "psa_angles": psa_angles,
            "powers_left": good_powers_left,
            "powers_right": good_powers_right,
            "powers_total": good_powers_total,
            "reference_channel": reference_channel,
            "other_channel": other_channel,
            "use_photodiode": use_photodiode
        }

        drrp_raw_data[key] = experiment_data

    return drrp_raw_data


# def subaperture_fits_data(drrp_raw_data):
