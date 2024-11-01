from NKTP_DLL import *

COMport = 'COM3'
COMPACT_devID = 1
SELECT_devID = 16

def scan_ports():
    openPorts(getAllPorts(), 1, 1)
    print('Following ports has modules:', getOpenPorts())
    portlist = getOpenPorts().split(',')
    for portName in portlist:
        result, devList = deviceGetAllTypesV2(portName)
        for devId in range(0, len(devList)):
            if (devList[devId] != 0):
                print('Comport:',portName,'Device type:',"0x%0.4X" % devList[devId],'at address:',devId)

    # Close all ports
    closeResult = closePorts('')
    print('Close result: ', PortResultTypes(closeResult))


# Functions for the SuperK COMPACT

# Get the interlock status
def get_interlock():
    result = registerReadU32(COMport, COMPACT_devID, 0x32, -1)
    LSB = result[1]

    if LSB == 2:
        print('Interlock is OK.')
    elif LSB == 1:
        print('Waiting for interlock to reset.')
    elif LSB == 0:
        print('Interlock off (interlock circuit open).')


# Disable interlock
def disable_interlock():
    result = registerWriteReadU32(COMport, COMPACT_devID, 0x32, 0, -1)
    LSB = result[1]

    if LSB == 2:
        print('Interlock is OK.')
    elif LSB == 1:
        print('Waiting for interlock to reset.')
    elif LSB == 0:
        print('Interlock off (interlock circuit open).')


# Reset interlock (make sure the physical key is turned On)
def reset_interlock():
    result = registerWriteReadU32(COMport, COMPACT_devID, 0x32, 1, -1)
    LSB = result[1]

    if LSB == 2:
        print('Interlock is OK.')
    elif LSB == 1:
        print('Waiting for interlock to reset.')
    elif LSB == 0:
        print('Interlock off (interlock circuit open).')


mode_mapping = {
    0: 'Internal frequency generator',
    1: 'External trig',
    2: 'Software trigged burst',
    3: 'Hardware trigged burst',
    4: 'External gate on',
    5: 'External gate off',
}

# Get or set the SuperK COMPACT operating mode (pulse trigger source)
def trig_mode(mode=None):
    if mode is None:
        result = registerReadU32(COMport, COMPACT_devID, 0x31, -1)
        status = mode_mapping[result[1]]
        print('Laser mode:', status)
    else:
        result = registerWriteReadU32(COMport, COMPACT_devID, 0x31, mode, -1)
        status = mode_mapping[result[1]]
        print('Laser mode:', status)


# Turn on the laser emission
def emission_on():
    result = registerWriteU8(COMport, COMPACT_devID, 0x30, 1, -1) # devID=1 for Compact
    print('Setting emission ON.', RegisterResultTypes(result))


# Turn off laser emission
def emission_off():
    result = registerWriteU8(COMport, COMPACT_devID, 0x30, 0, -1)
    print('Setting emission OFF.', RegisterResultTypes(result))


# Get or set the current overall power for laser emission as a percent
def overall_power(power=None):
    if power is None:
        result = registerReadU8(COMport, COMPACT_devID, 0x3E, -1)
        current_power = result[1]
        print(f'Overall power level: {current_power}%')
    else:
        result = registerWriteReadU8(COMport, COMPACT_devID, 0x3E, power, -1)
        current_power = result[1]
        print(f'Setting overall power level to {current_power}%.')

# def get_overall_power():
#     result = registerReadU8(COMport, COMPACT_devID, 0x3E, 0)
#     power = result[1]
#     print(f'Overall power level: {power}%')


# # Set the current overall power for the laser emission as a percent
# def set_overall_power(power):
#     result = registerWriteU8(COMport, COMPACT_devID, 0x3E, power, -1)
#     print(f'Setting overall power level to {power}%: {RegisterResultTypes(result)}')


# Get the internal pulse frequency limit in Hz
def get_max_pulse():
    result = registerReadU32(COMport, COMPACT_devID, 0x36, -1)
    max_frequency = result[1]
    print(f'Maximum possible internal frequency: {max_frequency} Hz.')


# Get the current internal pulse frequency in Hz
def get_pulse_frequency():
    result = registerReadU32(COMport, COMPACT_devID, 0x33, -1)
    frequency = result[1]
    print(f'Current internal pulse frequency: {frequency} Hz.')


# Set the current internal pulse frequency in Hz, then readout the current value
def set_pulse_frequency(target_frequency):
    result = registerWriteReadU32(COMport, COMPACT_devID, 0x33, target_frequency, -1)
    frequency = result[1]
    print(f'Current internal pulse frequency: {frequency} Hz.')


# Now functions for the SuperK SELECT

# Crystal 1 (VIS/NIR) minimum wavelength in nm
def lambda_min1():
    result = registerReadU32(COMport, SELECT_devID, 0x90, -1)
    wavelength = result[1]/1000
    print(f'Minimum wavelength for crystal 1: {int(wavelength)} nm.')


# Crystal 1 (VIS/NIR) maximum wavelength in nm
def lambda_max1():
    result = registerReadU32(COMport, SELECT_devID, 0x91, -1)
    wavelength = result[1]/1000
    print(f'Minimum wavelength for crystal 1: {int(wavelength)} nm.')


# Crystal 2 (NIR/IR) minimum wavelength in nm
def lambda_min2():
    result = registerReadU32(COMport, SELECT_devID, 0xA0, -1)
    wavelength = result[1]/1000
    print(f'Minimum wavelength for crystal 1: {int(wavelength)} nm.')


# Crystal 2 (NIR/IR) maximum wavelength in nm
def lambda_max2():
    result = registerReadU32(COMport, SELECT_devID, 0xA1, -1)
    wavelength = result[1]/1000
    print(f'Minimum wavelength for crystal 1: {int(wavelength)} nm.')

