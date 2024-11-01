from NKTP_DLL import *

# make a Class for Compact laser and SELECT filter

# Turn on the laser emission
def power_on():
    result = registerWriteU8('COM3', 1, 0x30, 1, -1) # devID=1 for Compact
    print('Setting emission ON.', RegisterResultTypes(result))

# Turn off laser emission
def power_off():
    result = registerWriteU8('COM3', 1, 0x30, 0, -1)
    print('Setting emission OFF.', RegisterResultTypes(result))

# Get the current overall power for laser emission as a percent
def get_overall_power():
    result = registerReadU8('COM3', 1, 0x3E, 0)
    power = result[1]
    print(f'Overall power level: {power}%')

# Set the current overall power for the laser emission as a percent
def set_overall_power(power):
    result = registerWriteU8('COM3', 1, 0x3E, power, -1)
    print(f'Setting overall power level to {power}%: {RegisterResultTypes(result)}')

