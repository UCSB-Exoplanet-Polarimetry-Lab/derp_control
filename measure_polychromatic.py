import numpy as np
import derpy
from time import sleep
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from tqdm import tqdm
from scipy.optimize import minimize

# Added to have access to the lab's functions for NKT laser control
# TODO: We want to make this installable later
import sys
sys.path.append("C:\\Users\\EPL User\\Documents\\Github\\NKT_laser_control\\")
from NKTcontrols.controls import compact, select, driver, get_status, SuperK

from katsu.mueller import(
    linear_retarder,
    linear_polarizer
)

from derpy import Experiment, forward_calibrate, forward_simulate
from derpy.writing import write_experiment

## USER CONFIGURATION HERE
# --------------------------------
# 1.1 to 2 microns
#wavelengths =        [1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]  # Ensure unique wavelengths
#channel_power_list = [100,  100,  50,   30,   30,   30,   30,   30,   50,   50]
wavelengths =        [1400, 1500]  # Ensure unique wavelengths
channel_power_list = [10, 10]
overall_power_list = len(channel_power_list) * [100,] # NOTE: This is not currently used
N_EXPERIMENTS = 1
ANGULAR_STEP = 3.6 # deg
ANGULAR_RATIO = 2.5 # deg
N_CAL_MEASUREMENTS = 24
N_MEASUREMENTS = 50
DATA_PATH = "data/20250206_Pre_GPI_Test/air_test_0_"
# --------------------------------

# define subaperture centers
cxr, cyr = 249, 326 # pixel index
cxl, cyl = 250, 135 # pixel index
cut = 80 # crop radius

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
    
    # Initialize the laser
    superk = SuperK()

    # Initialize the camera
    cam = derpy.CRED2(set_temperature=-40,
                      fps=100, tint=1, temp_tolerance=0.5) # TODO: add units to docstring (also add docstring)
    
    # Initialize the rotation stages
    psg_stg = derpy.PSGRotationStage()
    psa_stg = derpy.PSARotationStage()

    # Take a dark image
    superk.set_channel(1, wavelengths[0], 0)
    dark = cam.take_median_image(100)
    superk.set_channel(1, wavelengths[0], 100)
    superk.compact.overall_power(power=100) # this is not altered

    bright = cam.take_many_images(10)
    bright_avg = np.median(bright, axis=0) - dark
    bright_avg[bright_avg < 0] = 0

    # define subaperture centers
    cxr, cyr = 250, 320 # pixel index
    cxl, cyl = 250, 120 # pixel index
    cut = 100 # crop radius

    # defined mask
    x = np.linspace(-1, 1, 2*cut)
    x, y = np.meshgrid(x, x)
    r = np.sqrt(x**2 + y**2)
    mask = np.zeros_like(r)
    mask[r < 1] = 1

    left_channel = bright_avg[cxl-cut:cxl+cut,cyl-cut:cyl+cut]
    right_channel = bright_avg[cxr-cut:cxr+cut,cyr-cut:cyr+cut]

    # Calibration step
    air_cal = Experiment(cam, psg_stg, psa_stg, superk,
                      dark=dark, cut=cut,
                      cxl=cxl, cxr=cxr, cyl=cyl, cyr=cyr,
                      wavelengths=wavelengths)

    cal_measure_ratio = N_MEASUREMENTS / N_CAL_MEASUREMENTS
    psg_step = ANGULAR_STEP * cal_measure_ratio
    psa_step = psg_step * ANGULAR_RATIO

    
    air_cal.measurement(psg_step, psa_step, N_CAL_MEASUREMENTS,
                         power=channel_power_list)
    
    print(20*"-")
    print("AIR CALIBRATION MEASUREMENT COMPLETE")
    print("Please place sample to test in Derp")
    print(20*"-")
    val = input("Press the ENTER key: ")

    print("proceeding with measurement ...")

    write_experiment(air_cal, DATA_PATH + "air_calibration")
    
    for i in range(N_EXPERIMENTS):
        # Set up a measurement
        measure = Experiment(cam, psg_stg, psa_stg, superk,
                            dark=dark, cut=cut,
                            cxl=cxl, cxr=cxr, cyl=cyl, cyr=cyr,
                            wavelengths=wavelengths)
        
        psg_step = ANGULAR_STEP
        psa_step = ANGULAR_STEP * ANGULAR_RATIO

        measure.measurement(psg_step, psa_step, N_MEASUREMENTS,
                            power=channel_power_list)
        

        # Save the experiments
        write_experiment(measure, DATA_PATH + f"measure_{i}")

    # close up
    cam.close()
    superk.set_emission("OFF")
    psg_stg.close()
    psa_stg.close()


    
        
    

    