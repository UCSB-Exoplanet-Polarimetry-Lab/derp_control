import time
import numpy as np
from astropy.io import fits
import os

class BaseCamera:
    def __init__(self, fps=None, tint=None, bit_depth=None):
        self._fps = fps
        self._tint = tint
        self.bit_depth = bit_depth

    @property
    def fps(self):
        return self._fps

    @fps.setter
    def fps(self, value):
        raise NotImplementedError("fps setter must be implemented in subclass")

    @property
    def tint(self):
        return self._tint

    @tint.setter
    def tint(self, value):
        raise NotImplementedError("tint setter must be implemented in subclass")

    def get_temperature(self):
        """Return current sensor temperature"""
        raise NotImplementedError

    def set_temperature(self, target_temp):
        """Set and wait for temperature"""
        raise NotImplementedError

    def _capture_raw_frame(self):
        """Return a single raw frame as numpy array"""
        raise NotImplementedError

    def take_many_images(self, num_frames, save_path=None, verbose=False):
        frames = []
        for i in range(num_frames):
            frame = self._capture_raw_frame()
            frames.append(frame)

            if verbose:
                print(f"Captured frame {i+1}/{num_frames}")

            # pacing by fps if known
            if self.fps:
                time.sleep(1.0 / self.fps)

        frames = np.array(frames)

        if save_path:
            fits.PrimaryHDU(frames).writeto(save_path, overwrite=True)

        return frames

    def take_median_image(self, n_frames, save_path=None, verbose=False):
        frames = self.take_many_images(n_frames, verbose=verbose)
        med = np.median(frames, axis=0)

        if save_path:
            fits.PrimaryHDU(med).writeto(f"{save_path}_median.fits", overwrite=True)

        return med

    def take_mean_image(self, n_frames, save_path=None, verbose=False):
        frames = self.take_many_images(n_frames, verbose=verbose)
        mean = np.mean(frames, axis=0)

        if save_path:
            fits.PrimaryHDU(mean).writeto(f"{save_path}_mean.fits", overwrite=True)

        return mean

    def take_std_image(self, n_frames, save_path=None, verbose=False):
        frames = self.take_many_images(n_frames, verbose=verbose)
        std = np.std(frames, axis=0)

        if save_path:
            fits.PrimaryHDU(std).writeto(f"{save_path}_std.fits", overwrite=True)

        return std

    def take_image(self, save_path=None):
        frame = self._capture_raw_frame()

        if save_path:
            fits.PrimaryHDU(frame).writeto(save_path, overwrite=True)

        return frame

    def close(self):
        raise NotImplementedError("close() must be implemented in subclass")

####################### ZWOASI required imports and functions #######################
import zwoasi as asi

class ZWOASI(BaseCamera):
    def __init__(self, camera_index=0, fps=200, tint=10, conversion_gain=150,
                 set_temperature=0, temp_tolerance=0.5, bit_depth=16):

        super().__init__(fps=fps, tint=tint, bit_depth=bit_depth)

        self.camera_index = camera_index
        self.conversion_gain = conversion_gain
        self.target_temperature = set_temperature
        self.temp_tolerance = temp_tolerance

        env_filename = os.getenv('ZWO_ASI_LIB')
        if not env_filename:
            raise RuntimeError("Set the environment variable ZWO_ASI_LIB")

        asi.init(env_filename)

        if asi.get_num_cameras() == 0:
            raise RuntimeError("No ASI cameras found")

        self.camera = asi.Camera(self.camera_index)

        # initialize camera
        self.camera.disable_dark_subtract()
        self.camera.stop_video_capture()
        self.camera.stop_exposure()

        self.camera.set_image_type(asi.ASI_IMG_RAW16 if bit_depth == 16 else asi.ASI_IMG_RAW8)
        self.camera.set_control_value(asi.ASI_GAIN, conversion_gain)

        # temperature + exposure
        self.set_temperature(self.target_temperature)
        self.tint = tint

        self.camera.start_video_capture()

    def get_temperature(self):
        return self.camera.get_control_value(asi.ASI_TEMPERATURE)[0]

    def set_temperature(self, target_temp):
        self.camera.set_control_value(asi.ASI_COOLER_ON, 1)
        self.camera.set_control_value(asi.ASI_TARGET_TEMP, float(target_temp))

        while True:
            current = self.get_temperature()
            if abs(current - target_temp) <= self.temp_tolerance:
                break
            print(f"Cooling… Current {current:.2f}C  Target {target_temp}C")
            time.sleep(1)

        print(f"Temperature stabilized at {self.get_temperature():.2f}C")

    @BaseCamera.tint.setter
    def tint(self, value):
        self._tint = float(value)
        self.camera.set_control_value(asi.ASI_EXPOSURE, int(self._tint * 1000))

    def _capture_raw_frame(self):
        return np.array(self.camera.capture_video_frame(), dtype=np.float32)

    def close(self):
        self.camera.stop_video_capture()
        self.camera.close()

####################### CRED2 required imports and functions #######################

from .derpy_conf import (
    np,
    CRED2_CAMERA_INDEX,
    CAMERA_TEMP_READOUT_DELAY,
    VERBOSE,
    FLI_SDK_PTH
)
from warnings import warn
import sys

from  .photodiode_class import OPM

try:
    sys.path.append(FLI_SDK_PTH)
    import FliSdk_V2 as sdk

except ImportError:
    warn(f"FliSdk_V2 not found at {FLI_SDK_PTH}. \n Make sure the SDK is installed and in your PYTHONPATH.")

def display_all_temps(context,verbose = True):
    res, mb, fe, pw, sensor, peltier, heatsink = sdk.FliCredTwo.GetAllTemp(context)
    if res:
        if verbose:
            print("Sensor Temperature: " + str(sensor) + "C")
            print("Motherboard Temperature: " + str(mb) + "C")
            print("Frontend Temperature: " + str(fe) + "C")
            print("Powerboard Temperature: " + str(pw) + "C")
            print("Peltier Temperature: " + str(peltier) + "C")
            print("Heatsink Temperature: " + str(heatsink) + "C")
            print("***********************")
    else:
        print("Could not read temperature")

    return sensor

def update_context(context):
    print("Updating...")
    ok = sdk.Update(context)

    if not ok:
        print("Error while updating.")
        exit()

class CRED2(BaseCamera):

    def __init__(self, set_temperature, fps, tint,
                 temp_tolerance=0.5, conversion_gain="low"):

        super().__init__(fps=fps, tint=tint, bit_depth=2**14)

        self.context = sdk.Init()
        self.conversion_gain = conversion_gain
        self.temp_tolerance = temp_tolerance
        self.target_temperature = float(set_temperature)

        # detect & initialize hardware
        self.grabbers = sdk.DetectGrabbers(self.context)
        assert len(self.grabbers) > 0, "No grabbers found"

        self.cameras = sdk.DetectCameras(self.context)
        assert len(self.cameras) > 0, "No cameras found"

        ok = sdk.SetCamera(self.context, self.cameras[CRED2_CAMERA_INDEX])
        assert ok, "Error setting camera"

        update_context(self.context)

        # set + wait for temperature
        self.set_temperature(set_temperature)

        # set fps, tint, gain via setters
        self.fps = fps
        self.tint = tint

    def get_temperature(self):
        res, mb, fe, pw, sensor, peltier, heatsink = sdk.FliCredTwo.GetAllTemp(self.context)
        if not res:
            raise RuntimeError("Temperature read failed")
        return sensor

    def set_temperature(self, target_temp):
        sdk.FliCredTwo.SetSensorTemp(self.context, float(target_temp))
        while True:
            sensor = self.get_temperature()
            if abs(sensor - target_temp) <= self.temp_tolerance:
                break
            print(f"Cooling… Current {sensor:.2f}C Target {target_temp}")
            time.sleep(CAMERA_TEMP_READOUT_DELAY)
        print(f"Temperature stabilized at {self.get_temperature():.2f}C")

    @BaseCamera.fps.setter
    def fps(self, value):
        self._fps = float(value)
        if sdk.IsSerialCamera(self.context):
            sdk.FliSerialCamera.SetFps(self.context, self._fps)
        elif sdk.IsCblueSfnc(self.context):
            sdk.FliCblueSfnc.SetAcquisitionFrameRate(self.context, self._fps)

    @BaseCamera.tint.setter
    def tint(self, value):
        self._tint = float(value)
        sdk.FliCredTwo.SetTint(self.context, self._tint / 1000.0)
        sdk.Update(self.context)

    def _capture_raw_frame(self):
        frame = sdk.GetRawImageAsNumpyArray(self.context, 0)
        return frame.astype(np.float32)

    def close(self):
        sdk.Stop(self.context)
        sdk.Exit(self.context)
