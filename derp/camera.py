import FliSdk_V2 as sdk
import time
from astropy.io import fits
from .derp_conf import (
    np,
    CRED2_CAMERA_INDEX,
    CAMERA_TEMP_READOUT_DELAY
)

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

    def __init__(self, set_temperature, fps, tint, conversion_gain='low'):

        self.context = sdk.Init()
        self.grabbers = sdk.DetectGrabbers(self.context)
        self.temperature_change = []
        self.set_temperature = set_temperature

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
        
        if self.set_temp < initial_temp:
            while sensor_temp > self.set_temp:
                sensor_temp = display_all_temps(self.context)
                self.temperature_change.append(sensor_temp)
                time.sleep(CAMERA_TEMP_READOUT_DELAY)

        elif self.set_temp > initial_temp:
            while sensor_temp < self.set_temp:
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
    def tint(self, value, verbose=False):
        """set the integration time of the camera in ms

        Parameters
        ----------
        value : float
            integration time in ms
        verbose : bool, optional
            whether to print the current and prior tint values, by default False
        """

        if sdk.IsCredTwo(self.context) or sdk.IsCredThree(self.context):
            res, response = sdk.FliSerialCamera.SendCommand(self.context, "mintint raw")
            self.min_tint = float(response)

            res, response = sdk.FliSerialCamera.SendCommand(self.context, "maxtint raw")
            self.max_tint = float(response)

            res, response = sdk.FliSerialCamera.SendCommand(self.context, "tint raw")
            tint = response * 1000
            if verbose:
                print(f"Prior camera tint: {tint}ms")

            mintint = self.min_tint*1000
            maxtint = self.max_tint*1000

            assert (value > mintint) and (value < maxtint), f"tint value {value}ms must be between {mintint} and {maxtint}"
           
            sdk.FliCredTwo.SetTint(self.context, float(float(value)/1000))
            
            ok = sdk.Update(self.context)
            assert ok, "Error while setting tint"

            res, response = sdk.FliCredTwo.GetTint(self.context)
            tint = response * 1000

            if verbose:
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

    def take_image(self, save_path=None, verbose=False):

        frame_list = self.take_many_images(1, save_path=save_path, verbose=verbose)
        return frame_list

    def close(self):
        sdk.Stop(self.context)
        sdk.Exit(self.context)