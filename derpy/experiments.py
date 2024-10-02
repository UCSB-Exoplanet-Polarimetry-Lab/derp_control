import numpy as np
from time import sleep

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





