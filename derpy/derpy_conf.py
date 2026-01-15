import numpy as np
import os

# TODO: Writing a conf class that gets called everywhere

# The Thorlabs configurations - outdated
FOCUS_STAGE_ID = 45452684
KINESIS_PSA_ROTATION_STAGE_ID = 55344714
KINESIS_PSG_ROTATION_STAGE_ID = 55346494

# NRRP
NRRP_PSG_ROTATION_STAGE_ID = 0
NRRP_PSA_ROTATION_STAGE_ID = 1

# VRRP - plugged in in reverse order from NRRP
VRRP_PSG_ROTATION_STAGE_ID = 3
VRRP_PSA_ROTATION_STAGE_ID = 2

# CRED Camera config
CRED2_CAMERA_INDEX = 0
CAMERA_TEMP_READOUT_DELAY = 8 # s

# Both VRRP and NRRP
# Check `Device Manager` to see what USB port Zaber is using
ZABER_PORT = "COM4"
VERBOSE = True

# File paths
FLI_SDK_PTH = "C:\\Program Files\\FirstLightImaging\\FliSdk\\Python\\lib"
os.environ['ZWO_ASI_LIB'] = r'C:\\Program Files\\ASIStudio\\ASICamera2.dll'
        

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
