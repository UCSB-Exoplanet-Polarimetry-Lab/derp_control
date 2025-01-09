
from katsu.katsu_math import broadcast_kron, np
from katsu.mueller import linear_polarizer, linear_retarder, linear_diattenuator


def measure_from_experiment(experiment, channel="both", frame_mask=None):

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


def q_measure_from_experiment(experiment):

    # Get the cuts
    cxl = experiment.cxl
    cyl = experiment.cyl
    cxr = experiment.cxr
    cyr = experiment.cyr
    cut = experiment.cut

    # Where it currently is
    derotate = experiment.psa_pol_angle
    starting_angle_psg_pol = experiment.psg_pol_angle - derotate
    starting_angle_psg_wvp = np.radians(experiment.psg_starting_angle) - derotate
    retardance_psg_wvp = experiment.psg_wvp_ret

    # starting_angle_psa_pol = experiment.psa_pol_angle
    starting_angle_psa_wvp = np.radians(experiment.psa_starting_angle) - derotate
    retardance_psa_wvp = experiment.psa_wvp_ret

    # Construct the position arrays
    psg_angles = np.radians(np.array(experiment.psg_positions_relative)) + starting_angle_psg_wvp
    psa_angles = np.radians(np.array(experiment.psa_positions_relative)) + starting_angle_psa_wvp

    # grab the sub-pupils - TODO: Will need to work on image registration
    images = experiment.images
    power_left = []
    power_right = []
    for img in images:
        cut_power_left = img[cxl-cut:cxl+cut, cyl-cut:cyl+cut]
        cut_power_right = img[cxr-cut:cxr+cut, cyr-cut:cyr+cut]
        power_left.append(cut_power_left)
        power_right.append(cut_power_right)

    power_left = np.asarray(power_left)
    power_right = np.asarray(power_right)

    shapes = [*power_left.shape[-2:], psa_angles.shape[0]]
    power_left = np.moveaxis(power_left, 0, -1)
    power_right = np.moveaxis(power_right, 0, -1)
    Q = (power_left - power_right) / (power_left + power_right)

    psg_pol = linear_polarizer(starting_angle_psg_pol)
    psg_wvp = linear_retarder(psg_angles, retardance_psg_wvp, shape=shapes)

    psa_wvp = linear_retarder(psa_angles, retardance_psa_wvp, shape=shapes)
    # psa_pol_left = linear_polarizer(starting_angle_psa_pol)
    # psa_pol_right = linear_polarizer(starting_angle_psa_pol + np.pi/2)

    Mg = psg_wvp @ psg_pol
    Ma = psa_wvp
    # Ma_left = psa_pol_left @ psa_wvp
    # Ma_right = psa_pol_right @ psa_wvp

    PSA = Ma[..., 1, :]
    # PSA_Left = Ma_left[..., 0, :]
    PSG = Mg[..., :, 0]

    # polarimetric data reduction matrix, flatten Mueller matrix dimension
    Wmat = broadcast_kron(PSA[..., np.newaxis], PSG[..., np.newaxis])
    Wmat = Wmat.reshape([*Wmat.shape[:-2], 16])

    # Wmat_Left = broadcast_kron(PSA_Left[..., np.newaxis], PSG[..., np.newaxis])
    # Wmat_Left = Wmat_Left.reshape([*Wmat_Left.shape[:-2], 16])

    # Winv_Left = np.linalg.pinv(Wmat_Left)
    Winv = np.linalg.pinv(Wmat)
    # power_left_expand = power_left[..., np.newaxis] / (power_left + power_right)[..., np.newaxis]
    Q_expand = Q[..., np.newaxis]
    # power_expand = np.concatenate([power_left_expand, Q_expand], axis=-2)
    # Winv = np.concatenate([Winv_Left, Winv], axis=-1)

    # Do the data reduction
    M_meas = Winv @ Q_expand
    M_meas = M_meas[..., 0]

    return M_meas.reshape([*M_meas.shape[:-1], 4, 4])


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