import numpy as np
from time import sleep, perf_counter
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from warnings import warn
from astropy.io import fits
from datetime import datetime
import ipdb

# Set up derpy
from derpy.motion import BaseZaberStage, ZaberConnection
from derpy.camera import CRED2, display_all_temps, ZWOASI
from derpy.derpy_conf import ZABER_PORT
from derpy.photodiode_class import OPM


"""
EXPERIMENT PARAMETERS DEFINED BY USER
--------------------------------
"""

ANGULAR_STEP = 3.6  # degrees
ANGULAR_RATIO = 2.5  # degrees
N_CAL_MEASUREMENTS = 24
N_MEASUREMENTS = 50
DATA_PATH = Path.home() / "Data/Derpy/01-13-2026/J_band_VVC"
TINT = 50 # milliseconds
FPS = 10
SET_TEMPERATURE = None  # degrees Celsius
WAVELENGTHS = [635]  # nm
WVL_POWERS = [1.41]
N_MEDIANS_DARK = 10
N_MEDIANS = 10
DARK_SUBTRACT = True # Subtract dark frames from images
SAVE_PSG_IMGS = False # Save PSG images
EXPERIMENT_NAME = "J_band_VVC_VRRP"
"""
---------------------------------
"""

if __name__ == "__main__":

    t1 = perf_counter()

    # Create experiment directory to save data
    DATA_PATH.mkdir(parents=True, exist_ok=True)

    # Init camera connection with default settings
    cam = ZWOASI(tint=TINT, fps=FPS)

    # Take a dark
    _ = input("Turn off laser and press ENTER to take a dark image: ")
    dark, dark_power = cam.take_median_image(N_MEDIANS_DARK)

    # Initialize the Zaber stage
    from derpy import (
        VRRP_PSG_ROTATION_STAGE_ID,
        VRRP_PSA_ROTATION_STAGE_ID,
    )
    zaber_connection = ZaberConnection(ZABER_PORT)
    psg = BaseZaberStage(zaber_connection, VRRP_PSG_ROTATION_STAGE_ID)
    psa = BaseZaberStage(zaber_connection, VRRP_PSA_ROTATION_STAGE_ID)

    # Home the stages
    print("Homing the stages...")
    psg.home()
    psa.home()

    nums = [N_CAL_MEASUREMENTS, N_MEASUREMENTS]
    labels = ["Calibration", "Measurement"]


    # Begin calibration
    for experiment, N in zip(labels, nums):
    
        # images after stage moves,
        # NOTE that we do polarimetric data reduction on the PSA images
        # TODO: Need to put definitions inside experiment loop
        psg_images = []
        psg_command_angles = []
        psg_encoder_angles = []
        psg_img_temps = [] # CRED2 sensor temperatures

        psa_images = []
        psa_command_angles = []
        psa_encoder_angles = []
        psa_img_temps = [] # CRED2 sensor temperatures
        psa_power_meter = []


        if experiment == "Calibration":
            _ = input("Turn on laser and press ENTER to begin calibration...")
            msg = "Running calibration..."

        else:
            _ = input("Place sample and press ENTER to begin measurement...")
            msg = "Running measurement..."

        psg_angles = np.linspace(0, N * ANGULAR_STEP, N)

        for i, angle in enumerate(tqdm(psg_angles, desc=msg)):

            # Move the PSG stage
            if i != 0:
                psg.step(ANGULAR_STEP)

            if SAVE_PSG_IMGS:

                imstack, _ = cam.take_many_images(N_MEDIANS)

                if DARK_SUBTRACT:
                    imstack_darksub = [im - dark for im in imstack]
                else:
                    imstack_darksub = imstack

                # Save the data
                psg_img_temps.append(display_all_temps(cam.context, verbose=False))
                psg_images.append(np.median(imstack_darksub, axis=0))
                
            # Save the command and encoder angles
            psg_command_angles.append(angle)
            psg_encoder_angles.append(psg.get_current_position())

            # Move the PSA stage
            if i != 0:
                psa.step(ANGULAR_STEP * ANGULAR_RATIO)

            imstack, psa_power = cam.take_many_images(N_MEDIANS)

            if DARK_SUBTRACT:
                imstack_darksub = [im - dark for im in imstack]
            else:
                imstack_darksub = imstack

            psa_images.append(np.median(imstack_darksub, axis=0))
            psa_power_meter.append(np.median(psa_power))
            psa_command_angles.append(angle * ANGULAR_RATIO)
            psa_encoder_angles.append(psa.get_current_position())

        # Save the images
        psg_cube = np.asarray(psg_images)
        psa_cube = np.asarray(psa_images)
        psa_meter_cube = np.asarray(psa_power_meter)

        todaysdate = datetime.now().strftime("%Y-%m-%d")
        todaystime = datetime.now().strftime("%H-%M-%S")

        # Save experiment parameters to primary hdu
        primary_hdu = fits.PrimaryHDU()
        primary_hdu.header['SET_TEMP'] = SET_TEMPERATURE
        primary_hdu.header['FPS'] = FPS
        primary_hdu.header['TINT'] = TINT
        primary_hdu.header['N_PER_MEDIAN'] = N_MEDIANS
        primary_hdu.header['DARK_SUBTRACT'] = DARK_SUBTRACT
        primary_hdu.header['SAVE_PSG_IMGS'] = SAVE_PSG_IMGS
        primary_hdu.header['WAVELENGTHS'] = str(WAVELENGTHS)
        primary_hdu.header['WVL_POWERS'] = str(WVL_POWERS)
        primary_hdu.header['EXPERIMENT'] = experiment
        primary_hdu.header['NUM_MEASUREMENTS'] = N
        primary_hdu.header['PSG_STEP'] = ANGULAR_STEP
        primary_hdu.header['PSA_STEP'] = ANGULAR_RATIO * ANGULAR_STEP
        primary_hdu.header['EXPERIMENT_NAME'] = EXPERIMENT_NAME
        primary_hdu.header['N_MEDIANS_DARK'] = N_MEDIANS_DARK
        primary_hdu.header['DATE'] = todaysdate
        primary_hdu.header['TIME'] = todaystime

        # Add axis labels to the primary hdu to keep track of things
        primary_hdu.header['AXIS0'] = 'Experiment Parmeters'
        primary_hdu.header['AXIS1'] = 'PSG_IMAGES'
        primary_hdu.header['AXIS2'] = 'PSG_TEMPERATURES'
        primary_hdu.header['AXIS3'] = 'PSG_COMMAND_ANGLES'
        primary_hdu.header['AXIS4'] = 'PSG_ENCODER_ANGLES'
        primary_hdu.header['AXIS5'] = 'PSA_IMAGES'
        primary_hdu.header['AXIS6'] = 'PSA_TEMPERATURES'
        primary_hdu.header['AXIS7'] = 'PSA_COMMAND_ANGLES'
        primary_hdu.header['AXIS8'] = 'PSA_ENCODER_ANGLES'

        # The PSG data
        psg_hdu = fits.ImageHDU(data=psg_cube, name='PSG_IMAGES')

        # NOTE
        # for the love of god please don't ask me why this is necessary,
        # I tried PrimaryHDU and it didn't let me name it. And I tried BinTableHDU and it
        # vomited out the numpy array and threw a big error.
        psg_temp_hdu = fits.ImageHDU(data=np.asarray(psg_img_temps), name='PSG_TEMPERATURES')
        psg_temp_hdu.header['UNITS'] = 'CELSIUS'

        psg_command_angles_hdu = fits.ImageHDU(data=np.asarray(psg_command_angles), name='PSG_COMMAND_ANGLES')
        psg_command_angles_hdu.header['UNITS'] = 'DEGREES'

        psg_encoder_angles_hdu = fits.ImageHDU(data=np.asarray(psg_encoder_angles), name='PSG_ENCODER_ANGLES')
        psg_encoder_angles_hdu.header['UNITS'] = 'DEGREES'

        # The PSA data
        psa_hdu = fits.ImageHDU(data=psa_cube, name='PSA_IMAGES')

        # NOTE
        # for the love of god please don't ask me why this is necessary,
        # I tried PrimaryHDU and it didn't let me name it. And I tried BinTableHDU and it
        # vomited out the numpy array and threw a big error.
        psa_temp_hdu = fits.ImageHDU(data=np.asarray(psa_img_temps), name='PSA_TEMPERATURES')
        psa_temp_hdu.header['UNITS'] = 'CELSIUS'

        psa_command_angles_hdu = fits.ImageHDU(data=np.asarray(psa_command_angles), name='PSA_COMMAND_ANGLES')
        psa_command_angles_hdu.header['UNITS'] = 'DEGREES'

        psa_encoder_angles_hdu = fits.ImageHDU(data=np.asarray(psa_encoder_angles), name='PSA_ENCODER_ANGLES')
        psa_encoder_angles_hdu.header['UNITS'] = 'DEGREES'


        # NOTE
        # for the love of god please don't ask me why this is necessary,
        # I tried PrimaryHDU and it didn't let me name it. And I tried BinTableHDU and it
        # vomited out the numpy array and threw a big error.
        psa_power_hdu = fits.ImageHDU(data=psa_meter_cube, name='PSA_POWER_METER')
        psa_power_hdu.header['METER'] = "THORLABS P100D"

        hdul = fits.HDUList([primary_hdu,
                             psg_hdu, psg_temp_hdu, psg_command_angles_hdu, psg_encoder_angles_hdu,
                             psa_hdu, psa_temp_hdu, psa_command_angles_hdu, psa_encoder_angles_hdu,
                             psa_power_hdu])
        hdul.writeto(DATA_PATH / f"{experiment.lower()}_data_{todaysdate}_{todaystime}.fits", overwrite=False)

    # Save the dark
    dark_hdu = fits.PrimaryHDU(data=dark)
    dark_hdu.header['N_PER_MEDIAN'] = N_MEDIANS_DARK
    dark_hdu.header['DARK_SUBTRACT'] = DARK_SUBTRACT
    dark_hdu.header['FPS'] = FPS
    dark_hdu.header['TINT'] = TINT
    dark_hdu.header['SET_TEMP'] = SET_TEMPERATURE

    # Close the Zaber connection
    zaber_connection.close()
    cam.close()

    t2 = perf_counter()
    print(f"Experiment completed in {t2 - t1:.2f} seconds.")
