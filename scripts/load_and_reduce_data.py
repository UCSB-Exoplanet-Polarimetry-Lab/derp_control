import numpy as np
import derpy
from time import sleep
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from tqdm import tqdm
from astropy.io import fits
from scipy.optimize import minimize

from katsu.mueller import(
    linear_retarder,
    linear_polarizer,
    retardance_parameters_from_mueller,
    retardance_from_mueller
)
from derpy import Experiment, forward_calibrate, forward_simulate
from derpy.data_reduction import (
    _measure_from_experiment,
    _measure_from_experiment_polychromatic,
    mueller_from_experiment
)
from derpy.writing import read_experiment

## USER CONFIGURATION HERE
# --------------------------------
# CALIBRATION_ID = "data/20250205_Pre_GPI_Test/air_test_1_air_calibration"
# EXPERIMENT_ID = "data/20250205_Pre_GPI_Test/air_test_1_measure_0"
CALIBRATION_ID = "data/20250210_GPI/GPI_HWP_50nm_offset_3_air_calibration"
EXPERIMENT_ID = "data/20250210_GPI/GPI_HWP_50nm_offset_3_measure_0"

# mask IDs for 02/10 GPI measurements
# 1550: WVL_ID = 4, BAD_FRAMES = [5, 12, 35], BAD_FRAMES_CAL = [11]

# TODO: Make these not redundant
WVL_ID = 4
WAVELENGTH_SELECT = 1550  # nm

PLOT_INTERMEDIATE = True
MASK_RAD = 0.5 # from 0 to 1, 1 being the full circle
MODE = "both" # "left", "right", "both"
BAD_FRAMES_CAL = [11] # [3, 13 -1], _, [8]
BAD_FRAMES = [5, 12, 35]  # [-9], _, [-11]
PLOT_IMAGES = False
# --------------------------------

def calibrate_experiment(x, experiment):

    experiment.psg_pol_angle = x[0]
    experiment.psg_starting_angle = np.degrees(x[1])
    experiment.psg_wvp_ret = x[2]
    experiment.psa_pol_angle = x[3]
    experiment.psa_starting_angle = np.degrees(x[4])
    experiment.psa_wvp_ret = x[5]

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

    # Make the bad frame mask for the calibration data
    mask_bad_frames = np.ones_like(exp.psg_positions_relative, dtype=bool)
    mask_bad_frames[BAD_FRAMES_CAL] = 0

    # perform calibration
    wavelengths = exp.wavelengths

    # Plotting option for looking at the calibration power
    if PLOT_INTERMEDIATE:
        for i, wvl in enumerate(wavelengths):
            plt.figure()
            plt.title("Initial calibration run "+rf"$\lambda={wvl}nm$")
            P_ref_0 = exp.mean_power_left[i, 0] + exp.mean_power_right[i, 0]
            P_ref_l = np.asarray(exp.mean_power_left[i]) + np.asarray(exp.mean_power_right[i])
            P_ref = P_ref_0 / P_ref_l

            # I'm wondering if the mean power computation is bugged
            plt.plot(exp.psg_positions_relative, exp.mean_power_left[i] * P_ref, label='left', marker='o')
            plt.plot(exp.psg_positions_relative, exp.mean_power_right[i] * P_ref, label='right', marker='o')
            plt.plot(exp.psg_positions_relative, P_ref_l, label='total', marker='o')
            plt.legend()
            plt.ylabel('Power')
            plt.xlabel('PSG Angle')
        # plt.show()

    # doing the calibration
    # Polychromatic calibration
    results_wvl = {}
    for i, wvl in enumerate(wavelengths):

        # init a starting guess
        x_model =[
            0, # starting_angle_psg_pol
            0, # starting_angle_psg_wvp
            np.pi/2, # retardance_psg_wvp
            0, # starting_angle_psa_pol
            0, # starting_angle_psa_wvp
            np.pi/2 # retardance_psa_wvp
        ]

        results = minimize(derpy.forward_calibrate, x0=x_model,
                           args=(exp, 'left', i, mask_bad_frames))
        results_wvl[str(wvl)] = results

    simulated_power_wvl = []
    for i, wvl in enumerate(wavelengths):
        simulated_power = forward_simulate(results_wvl[str(wvl)].x, 
                                        exp,
                                        'left')
        
        simulated_power_wvl.append(simulated_power)

    psg_angles = exp.psg_positions_relative

    if PLOT_INTERMEDIATE:
        for i, simulated_power in enumerate(simulated_power_wvl):
            plt.figure()
            plt.title(rf"Calibration run $\lambda={wavelengths[i]}nm$")
            plt.plot(psg_angles, simulated_power / np.max(simulated_power), label='simulated')
            plt.plot(psg_angles, exp.mean_power_left[i] / np.max(exp.mean_power_left[i]), label='measured', linestyle="None", marker='o')
            # plt.plot(psg_angles, exp.mean_power_right[i] / np.max(exp.mean_power_right[i]), label='measured', linestyle="None", marker='o')
            plt.legend()
            plt.xlabel('PSG Angle, deg')
            plt.ylabel('Normalized Power')
        # plt.show()

    exp_calibrated = calibrate_experiment(results_wvl[str(WAVELENGTH_SELECT)].x, exp)
    M = mueller_from_experiment(exp_calibrated, channel='left',
                                frame_mask=mask_bad_frames)

    # define a mask to protect us from the big dots
    mask_data = np.zeros_like(M[WVL_ID][:,:,0,0])
    x = np.linspace(-1, 1, mask_data.shape[0])
    x, y = np.meshgrid(x, x)
    r = np.sqrt(x**2 + y**2)
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
    mask_bad_frames = np.ones_like(measure.psg_positions_relative, dtype=bool)
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


    results = results_wvl[str(WAVELENGTH_SELECT)]

    # Update experiment position
    # TODO: THIS WILL BREAK EVERYTHING IF YOU CHANGE ANYTHING IN THE MEASURE_POLYCHROMATIC.PY FILE
    # prior_psg_motion += 180 * WVL_ID
    # prior_psa_motion += 450 * WVL_ID

    print("Wavelengths in Experiment: ", measure.wavelengths)

    measure.psg_pol_angle = results.x[0]
    measure.psg_starting_angle = np.degrees(results.x[1]) + prior_psg_motion
    measure.psg_wvp_ret = results.x[2]
    measure.psa_pol_angle = results.x[3]
    measure.psa_starting_angle = np.degrees(results.x[4]) + prior_psa_motion
    measure.psa_wvp_ret = results.x[5]
    
    M_measure = mueller_from_experiment(measure, channel=MODE,
                                        frame_mask=mask_bad_frames)
    

    #M_measure is a list of length 1, so if if WVL_ID is anythong other than 0 [WVL_ID] returns an IdexError

    plot_square(M_measure[WVL_ID] / M_measure[WVL_ID][..., 0, 0, None, None],
                title=f"{MODE} Inversion {WAVELENGTH_SELECT} nm", vmin=-1.1, vmax=1.1,
                scale_offdiagonal=1, mask=mask_data)
    
    # plot_square(M_measure[0] / M_measure[0][..., 0, 0, None, None],
    #             title=f"{MODE} Inversion {WAVELENGTH_SELECT} nm", vmin=-1.1, vmax=1.1,
    #             scale_offdiagonal=1, mask=mask_data)
    
    #I have no idea what i'm doing
    # square_duim = M_measure[WVL_ID].shape[0]
    # retardance = np.zeros([square_duim, square_duim])
    # for i in range(square_duim):
    #     for j in range(square_duim):
    #         retardance[i, j] = retardance_parameters_from_mueller(M_measure[WVL_ID][i, j])
    # retardance = retardance_from_mueller(M_measure[WVL_ID])

    # plt.figure()
    # plt.title(f"{WAVELENGTH_SELECT}nm Retardance")
    # plt.imshow(retardance, cmap="RdBu_r")
    # plt.colorbar()
    # plt.show()

    #plot_square(retardance, title=f"{MODE} Retardance", vmin=-1.1, vmax=1.1, scale_offdiagonal=1, mask=mask_data)



