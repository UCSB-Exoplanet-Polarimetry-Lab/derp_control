import numpy as np
import msgpack
import msgpack_numpy as m
from astropy.io import fits
from astropy.table import Table
from .experiments import Experiment

"""taken from poke.writing with permission from author Jaren Ashcraft"""

m.patch()


def serialize(T):
    """serializes a class using msgpack
    written by Brandon Dube, docstring by Jaren Ashcraft

    Parameters
    ----------
    T : class
        class to convert to hex code. Used for rayfronts

    Returns
    -------
    serdat
        serial data corresponding to class T
    """
    glb = globals()
    Tname = T.__class__.__name__
    # assert Tname in glb, 'class must exist in globals in order to be re-hydrateable, with the same constraint'

    # now we make our storage format.  It will be:
    # 1) a header with the class name
    # 2) the content of vars(T)
    serdat = (Tname, vars(T))
    return msgpack.packb(serdat)


class MsgpackTrickerEmpty:
    """dummy class to trick msgpack
    """
    pass


def deserialize(buf):
    """deserializes a class using msgpack
    written by Brandon Dube, docstring by Jaren Ashcraft

    Parameters
    ----------
    buf : serdat
        serial data coorresponding to class

    Returns
    -------
    class
        deserialized class, typically a rayfront
    """
    e = MsgpackTrickerEmpty()
    Tname, varzzz = msgpack.unpackb(buf, use_list=True)
    for k, v in varzzz.items():
        setattr(e, k, v)

    e.__class__ = globals()[Tname]
    return e


def write_experiment(experiment, filename):
    """writes Experiment object to serial file using msgpack

    Parameters
    ----------
    rayfront : derpy.Experiment
        Rayfront object to serialize
    filename : str
        name of the file to save serial data to. The .msgpack extension will be added to this string
    """

    # clear of hardware connection
    experiment.cam = None
    experiment.psa = None
    experiment.psg = None
    experiment.laser = None

    serdata = serialize(experiment)

    with open(filename + ".msgpack", "wb") as outfile:
        outfile.write(serdata)


def read_experiment(filename):
    """reads serial data containing Experiment into an Experiment object

    Parameters
    ----------
    filename : str
        name of the file to read serial data from

    Returns
    -------
    derpy.Experiment
        the saved derpy.Experiment object
    """

    with open(filename, "rb") as infile:
        serdata = infile.read()

    rayfront = deserialize(serdata)

    return rayfront

# fits reading / writing
def save_experiment_data(experiment, filename, overwrite=True):

    # This will give us a data cube containing the images, angles, and calibrated parameters
    hdu_new = fits.PrimaryHDU(experiment.images)
    hdu_mask = fits.PrimaryHDU(experiment.mask)

    # construct dictionary containing relevant experimental information
    # TODO: Add calibrated parameters if they exist
    parameters = {
        "PSGSTART": experiment.psg_starting_angle,
        "PSASTART": experiment.psa_starting_angle
    }

    mask_params = {
        "XCENLEFT": experiment.cxl,
        "YCENLEFT": experiment.cyl,
        "XCENRGHT": experiment.cxr,
        "YCENRGHT": experiment.cyr,
        "CROP_RAD": experiment.cut
    }

    # table of angles
    angles = {
        "MEANLEFT" : experiment.mean_power_left,
        "MEANRGHT" : experiment.mean_power_left,
        "PSGANGS": experiment.psg_positions_relative,
        "PSAANGS": experiment.psa_positions_relative
    }

    angle_table = fits.BinTableHDU(Table(angles))


    # Create the FITS header from the dictionary
    for key, value in parameters.items():
        hdu_new.header[key] = value

    for key, value in mask_params.items():
        hdu_mask.header[key] = value

    # hdul = fits.HDUList([hdu_new, hdu_mask, angle_table])
    hdu_new.writeto(filename + "_data.fits", overwrite=True)
    hdu_mask.writeto(filename + "_mask.fits", overwrite=True)
    angle_table.writeto(filename + "_angles.fits", overwrite=True)

