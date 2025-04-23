import numpy as np
import derpy
from time import sleep
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from tqdm import tqdm

EXPOSURE_TIMES = [] # Write exposure times here

if __name__ == "__main__":

    # initialize camera
    cam = derpy.CRED2()