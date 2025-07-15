import time
from astropy.io import fits
from .derpy_conf import (
    np,
    CRED2_CAMERA_INDEX,
    CAMERA_TEMP_READOUT_DELAY,
    VERBOSE,
    FLI_SDK_PTH
)
from warnings import warn
import sys

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


class CRED2:

    def __init__(self, set_temperature, fps, tint, temp_tolerance=0.5, conversion_gain='low'):

        self.context = sdk.Init()
        self.grabbers = sdk.DetectGrabbers(self.context)
        self.temperature_change = []
        self.set_temperature = set_temperature
        self.bit_depth = 2**14

        assert (set_temperature > -55) and (set_temperature < 20), f"{set_temperature}C is not a valid temperature"
        self.set_temp = np.float64(set_temperature)

        self._fps = fps
        self._tint = tint

        if len(self.grabbers) == 0:

            assert len(self.grabbers) > 0, "No grabbers found"

        self.cameras = sdk.DetectCameras(self.context)

        if len(self.cameras) == 0:

            assert len(self.cameras) > 0, "No camera found"

        ok = sdk.SetCamera(self.context, self.cameras[CRED2_CAMERA_INDEX])

        assert ok, "Error while setting camera"

        update_context(self.context)
        
        initial_temp = display_all_temps(self.context)

        ok = sdk.FliCredTwo.SetSensorTemp(self.context, self.set_temp)
        assert ok, "Error while setting camera temperature"

        sensor_temp = display_all_temps(self.context)
        

        # Use np.isclose() to check if the temperature is within tolerance
        tol = temp_tolerance # absolute tolerance
        rtol = 0 # relative tolerance (optional)

        if sensor_temp != self.set_temp:
            while not np.isclose(self.set_temp, sensor_temp, rtol=rtol, atol=tol):
                sensor_temp = display_all_temps(self.context)
                self.temperature_change.append(sensor_temp)
                time.sleep(CAMERA_TEMP_READOUT_DELAY)           


        self.sensor_temp = display_all_temps(self.context, verbose=False)
        print(f'Final Sensor Temperature {self.sensor_temp:.2f}C')

        # set the camera fps, tint, and conversion gain using the setters
        self.fps = fps
        self.tint = tint
        self.conversion_gain = conversion_gain

    @property
    def fps(self):
        return self._fps
    
    @fps.setter
    def fps(self, value):
        self._fps = float(value)

        if sdk.IsSerialCamera(self.context):
            sdk.FliSerialCamera.SetFps(self.context, self._fps)

        elif sdk.IsCblueSfnc(self.context):
            sdk.FliCblueSfnc.SetAcquisitionFrameRate(self.context, self._fps)

    @property
    def tint(self):
        return self._tint
    
    @tint.setter
    def tint(self, value):
        """set the integration time of the camera in ms

        Parameters
        ----------
        value : float
            integration time in ms
        """

        if sdk.IsCredTwo(self.context) or sdk.IsCredThree(self.context):
            res, response = sdk.FliSerialCamera.SendCommand(self.context, "mintint raw")
            self.min_tint = float(response) * 1000

            res, response = sdk.FliSerialCamera.SendCommand(self.context, "maxtint raw")
            self.max_tint = float(response) * 1000

            res, response = sdk.FliSerialCamera.SendCommand(self.context, "tint raw")
            tint = response * 1000
            #if VERBOSE:
                #print(f"Prior camera tint: {tint}ms")

            mintint = self.min_tint
            maxtint = self.max_tint

            assert (value > mintint) and (value < maxtint), f"tint value {value}ms must be between {mintint} and {maxtint}"
           
            sdk.FliCredTwo.SetTint(self.context, float(float(value)/1000))
            
            ok = sdk.Update(self.context)
            assert ok, "Error while setting tint"

            res, response = sdk.FliCredTwo.GetTint(self.context)
            tint = response * 1000
            self._tint = tint

            if VERBOSE:
                print(f"Current camera tint: {tint}ms")

        else:
            raise Exception("Camera is not a Cred2 or Cred3")
        
    @property
    def conversion_gain(self):
        return self._conversion_gain
    
    @conversion_gain.setter
    def conversion_gain(self, value):
        self._conversion_gain = value
        res = sdk.FliCredTwo.SetConversionGain(self.context, value)

    def take_many_images(self, num_images, save_path=None, verbose=False):
        """take many images and save them to a directory

        Parameters
        ----------
        num_images : int
            number of images to take
        save_path : str
            path to save the images, if None, images are not saved
        verbose : bool, optional
            whether to print the current image number, by default False
        """
        sdk.Update(self.context)
        sdk.Start(self.context)

        frame_list = []

        for i in range(num_images):
            frame = sdk.GetRawImageAsNumpyArray(self.context, 0).astype(np.float64)
            frame_list.append(frame)

            if verbose:
                print(f"Image {i} taken")

            # TODO: I think this is the wrong way to do this, should be fps not tint
            time.sleep(10 * self.tint / 1e3)

        frame_list = np.array(frame_list)

        if save_path is not None:
            hdu = fits.PrimaryHDU(frame_list)
            hdul = fits.HDUList([hdu])
            hdul.writeto(save_path, overwrite=True)

        return frame_list

    def take_median_image(self, n_frames, save_path=None, verbose=False):
        frame_list = self.take_many_images(n_frames, save_path=save_path, verbose=verbose) 
        frame_list_median = np.median(frame_list, axis=0)

        if save_path is not None:
            hdu = fits.PrimaryHDU(frame_list_median)
            hdul = fits.HDUList([hdu])
            hdul.writeto(f'{save_path}_median', overwrite=True) # overwrites original, non-median-combined image

        return frame_list_median 
    
    def take_mean_image(self, n_frames, save_path=None, verbose=False):
        frame_list = self.take_many_images(n_frames, save_path=save_path, verbose=verbose) 
        frame_list_mean = np.mean(frame_list, axis=0)

        if save_path is not None:
            hdu = fits.PrimaryHDU(frame_list_mean)
            hdul = fits.HDUList([hdu])
            hdul.writeto(f'{save_path}_mean', overwrite=True) # overwrites original, non-mean-combined image

        return frame_list_mean 
    
    def take_std_image(self, n_frames, save_path=None, verbose=False):
        frame_list = self.take_many_images(n_frames, save_path=save_path, verbose=verbose) 
        frame_list_std = np.std(frame_list, axis=0)

        if save_path is not None:
            hdu = fits.PrimaryHDU(frame_list_std)
            hdul = fits.HDUList([hdu])
            hdul.writeto(f'{save_path}_std', overwrite=True) # overwrites original, non-std-combined image

        return frame_list_std 

    def take_image(self, save_path=None, verbose=False):

        frame_list = self.take_many_images(1, save_path=save_path, verbose=verbose)
        return frame_list

    def close(self):
        sdk.Stop(self.context)
        sdk.Exit(self.context)