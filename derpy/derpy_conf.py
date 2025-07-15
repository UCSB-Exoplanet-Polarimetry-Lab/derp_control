import numpy as np

FOCUS_STAGE_ID = 45452684
PSA_ROTATION_STAGE_ID = 55344714
PSG_ROTATION_STAGE_ID = 55346494
CRED2_CAMERA_INDEX = 0
CAMERA_TEMP_READOUT_DELAY = 8 # s
ZABER_PORT = "COM4"

VERBOSE = True

FLI_SDK_PTH = "C:\\Program Files\\FirstLightImaging\\FliSdk\\Python\\lib"


class BackendShim:
    """A shim that allows a backend to be swapped at runtime.
    Taken from prysm.mathops with permission from Brandon Dube
    """

    def __init__(self, src):
        self._srcmodule = src
 
    def __getattr__(self, key):
        if key == "_srcmodule":
            return self._srcmodule

        return getattr(self._srcmodule, key)


_np = np
np = BackendShim(_np)


def set_backend_to_numpy():
    """Convenience method to automatically configure katsu's backend to numpy."""
    import numpy

    np._srcmodule = numpy

    return