from pylablib.devices import Thorlabs
from .derpy_conf import (
    FOCUS_STAGE_ID,
    PSG_ROTATION_STAGE_ID,
    PSA_ROTATION_STAGE_ID,
    np
)

def print_connected_devices():
    print(Thorlabs.list_kinesis_devices())


class BaseStage:

    def __init__(self, ID):
        """A base class for all stages that utilize pylablib. Goal
        is for this to be a front-end tailored to the DRRP in the Exopol
        lab. The pylablib interface is accessed through self.device.

        Parameters
        ----------
        ID : int
            device ID, use print_connected_devices() to find this 
        """
        self.ID = ID

    def get_status(self, channel=None):
        """Get the status of the stage

        Returns
        -------
        str
            device status
        """
        return self.device.get_status(channel=channel)

    def home(self, sync=True):
        """Home the stage

        Parameters
        ----------
        sync : bool, optional
            whether to wait until homing is done, by default True

        Returns
        -------
        _type_
            _description_
        """
        self.device._setup_homing()
        self.home = self.device.home(sync=sync)
        return self.home

    def step(self, steps):
        """step the stage in one direction

        Parameters
        ----------
        steps : int
            number of steps to move the stage
        """
        self.device.move_by(steps)
        self.device.wait_move()

    def close(self):
        """Close the connection to the stage device
        """
        self.device.close()


class FocusStage(BaseStage):

    def __init__(self):
        super().__init__(FOCUS_STAGE_ID)
        self.device = Thorlabs.KinesisMotor(self.ID, is_rack_system=True)


class PSGRotationStage(BaseStage):

    def __init__(self):
        super().__init__(PSG_ROTATION_STAGE_ID)
        self.device = Thorlabs.KinesisMotor(self.ID, scale='stage')


class PSARotationStage(BaseStage):
    
    def __init__(self):
        super().__init__(PSA_ROTATION_STAGE_ID)
        self.device = Thorlabs.KinesisMotor(self.ID, scale='stage')


# class RotationStage:

#     def __init__(self):


