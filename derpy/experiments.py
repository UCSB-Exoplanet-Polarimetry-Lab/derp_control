import numpy as np
from time import sleep
from katsu.mueller import linear_polarizer, linear_retarder, linear_diattenuator
from katsu.katsu_math import broadcast_kron
from tqdm import tqdm

def autofocus_stage(camera, linear_stage, dark=None,
                    repetition_depth=10, step_size=1e6, maxiters=100,
                    verbose=True):

    # record current max brightness
    brightvals = []
    i = 0

    # take an initial image
    def _combine():
        
        im = camera.take_many_images(10)

        if dark is not None:
            for i, _ in enumerate(im):
                im[i] -= dark

        im_med = np.mean(im, axis=0)
        profile_median = np.mean(im_med, axis=0)

        return profile_median

    med_start = _combine()
    max_start = np.max(med_start)

    # move the stage by one step
    while (repetition_depth > 0) and (i < maxiters):

        i += 1
        if verbose:
            print(f"iteration {i} and repetition depth {repetition_depth}")

        linear_stage.step(step_size)
        mean_end = _combine()
        max_end = np.max(mean_end)

        if max_start > max_end:
            if verbose:
                print("DIRECTION CHANGE")
            step_size /= -2
            repetition_depth -= 1

        max_start = max_end
        brightvals.append(max_end)

        sleep(0.1)

    return brightvals


class Experiment:

    def __init__(self, cam, psg, psa,
                 psg_pol_angle=0, psg_wvp_angle=0, psg_wvp_ret=np.pi/2,
                 psa_pol_angle=0, psa_wvp_angle=0, psa_wvp_ret=np.pi/2,
                 dark=None, cxr=200, cyr=387, cxl=200, cyl=195, cut=120):
        
        # initialize the hardware
        self.cam = cam
        self.psg = psg
        self.psa = psa

        # initialize the parameters to calibrate
        self.psg_pol_angle = psg_pol_angle
        self.psg_wvp_ret = psg_wvp_ret
        self.psa_pol_angle = psa_pol_angle
        self.psa_wvp_ret = psa_wvp_ret

        # track the relative waveplate motion
        self.psg_positions_relative = [] # the position history
        self.psg_starting_angle = psg_wvp_angle # where the waveplate started
        self.psa_positions_relative = [] # the position history
        self.psa_starting_angle = psa_wvp_angle # where the waveplate started

        # track the image acquisition history
        self.images = []
        self.mean_power_left = []
        self.mean_power_right = []

        if dark is not None:
            self.dark = dark

        # set up aperture masks
        x = np.linspace(-1, 1, 2*cut)
        x, y = np.meshgrid(x, x)
        r = np.sqrt(x**2 + y**2)
        self.mask = np.zeros_like(r)
        self.mask[r < 0.5] = 1

        # Add the crop parameters
        self.cxl = cxl
        self.cyl = cyl
        self.cxr = cxr
        self.cyr = cyr
        self.cut = cut

    def measurement(self, psg_angular_step, psa_angular_step, n_steps, n_imgs=5):

        # grab cuts
        cxl = self.cxl
        cyl = self.cyl
        cxr = self.cxr
        cyr = self.cyr
        cut = self.cut

        for i in tqdm(range(n_steps)):

            # move the psg
            self.psg.step(psg_angular_step)
            self.psg_positions_relative.append(psg_angular_step * (i+1))
            sleep(0.1) # beccause of paranoia

            # move the psa
            self.psa.step(psa_angular_step)
            self.psa_positions_relative.append(psa_angular_step * (i+1))
            sleep(0.1)

            # take a measurement
            img = self.cam.take_median_image(n_imgs)
            if hasattr(self, 'dark'):
                img -= self.dark
                img[img < 0] = 1e-10

            # capture the full frame, no sub-pupiling
            self.images.append(img)

            # Get the average power in the sub-pupils
            self.mean_power_left.append(np.mean(img[cxl-cut:cxl+cut, cyl-cut:cyl+cut][self.mask==1]))
            self.mean_power_right.append(np.mean(img[cxr-cut:cxr+cut, cyr-cut:cyr+cut][self.mask==1]))
            
    
    @property
    def psg_wvp_angle(self):
        return self.psg_starting_angle + np.array(self.psg_positions_relative)
    
    @psg_wvp_angle.setter
    def psg_wvp_angle(self, value):
        self.psg_starting_angle = value 
    
    
    @property
    def psa_wvp_angle(self):
        return self.psa_starting_angle + np.array(self.psa_positions_relative)
    
    @psa_wvp_angle.setter
    def psg_wvp_angle(self, value):
        self.psa_starting_angle = value 


def forward_simulate(x, experiment, channel="left"):

    starting_angle_psg_pol = x[0]
    starting_angle_psg_wvp = x[1]
    retardance_psg_wvp = x[2]

    starting_angle_psa_pol = x[3]
    starting_angle_psa_wvp = x[4]
    retardance_psa_wvp = x[5]

    # parse experiment
    psg_angles = np.radians(np.array(experiment.psg_positions_relative))
    psa_angles = np.radians(np.array(experiment.psa_positions_relative))

    shapes = [psa_angles.shape[0]]

    psg_pol = linear_polarizer(starting_angle_psg_pol)
    psg_wvp = linear_retarder(psg_angles + starting_angle_psg_wvp, retardance_psg_wvp, shape=shapes)

    psa_wvp = linear_retarder(psa_angles + starting_angle_psa_wvp, retardance_psa_wvp, shape=shapes)
    psa_pol = linear_polarizer(starting_angle_psa_pol)

    M_total = psa_pol @ psa_wvp @ psg_wvp @ psg_pol
    power_simulated = M_total[..., 0, 0]

    return power_simulated


def forward_calibrate(x, experiment, channel="left"):

    if channel == "left":
        power = experiment.mean_power_left

    elif channel == "right":
        power = experiment.mean_power_right

    else:
        raise ValueError(f"channel '{channel}' must be either 'left' or 'right'")

    power_simulated = forward_simulate(x, experiment, channel=channel)

    # normalize the power
    power_simulated = power_simulated / np.max(power_simulated)
    power = power / np.max(power)

    # compute error to calibrate
    error = np.sum((power - power_simulated)**2)

    return error



