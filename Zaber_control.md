# Getting Started with Zaber Rotation Stages

### Plugging in the Stages
The first step is, of course, to connect the stages to the computer. Here is the [manual](https://www.zaber.com/manuals/X-RSW-E) if needed.

1. If using multiple devices, **first** daisy chain all devices together using the "Prev" and "Next" connectors. Call them device 1, device 2, ….
2. Supply power to a device (I usually choose device 1) and connect device 1 to computer through its "Prev" port.

### Moving the Stages
Here I'm assuming we are using the lab computer Mica. If using other Windows computer, enter `py -3 -m pip install --user zaber-motion` into a terminal to install the package. [Here](https://software.zaber.com/motion-library/docs/tutorials/install/py) is the Python library documentation.

After connecting to the computer, we need the name of the port. On Windows it's usually `COM3`.

We then import the modules needed:


```python
from zaber_motion import Units
from zaber_motion.ascii import Connection
```

Now we can open a connection and control the device. It's recommended to use the `with` statement to ensure that the serial port closes after the program runs:


```python
with Connection.open_serial_port("COM3") as connection: # Replace COM3 with your port name
    connection.enable_alerts()

    # Detecting devices
    device_list = connection.detect_devices()
    print("Found {} devices".format(len(device_list)))

    # The first device
    device1 = device_list[0]

    # Make sure it's homed: only need to home once before power off
    axis = device1.get_axis(1)
    if not axis.is_homed():
      axis.home()

    # Move to 10˚ (from home position)
    axis.move_absolute(position = 10, unit = Units.ANGLE_DEGREES)

    # Move by an additional 5˚ (from current position) with velocity of 3˚/sec
    axis.move_relative(5, Units.ANGLE_DEGREES, velocity = 3, velocity_unit=Units.ANGULAR_VELOCITY_DEGREES_PER_SECOND)

    # Move to -26.5˚ with velocity 1 rad/sec and acceleration 1.5˚/s^2
    axis.move_absolute(-26.5, Units.ANGLE_DEGREES, velocity = 1, velocity_unit=Units.ANGULAR_VELOCITY_RADIANS_PER_SECOND,
                       acceleration = 1.5, acceleration_unit = Units.ANGULAR_ACCELERATION_DEGREES_PER_SECOND_SQUARED)

```

Or if you want to leave the connection on:


```python
connection = Connection.open_serial_port("COM3")

device_list = connection.detect_devices()
print("Found {} devices".format(len(device_list)))

device1 = device_list[0]
axis = device1.get_axis(1)
if not axis.is_homed():
      axis.home()

axis.move_absolute(-10, Units.ANGLE_RADIANS)
axis.move_relative(-5, velocity = 5.2, velocity_unit=Units.ANGULAR_VELOCITY_DEGREES_PER_SECOND)

# Don't forget to close the connection at the end of session
connection.close()

```

Some devices have multiple axes, so the `Axis` class allows control of a specific axis on the device. Axes are indexed from 1. For the rotation stages, we only need to concern axis 1.

The movements will execute one after the other, with the program waiting for the movement to finish before continuing. If want the method to not wait, we can use:


```python
# This movement will not wait
axis.move_relative(5, Units.NATIVE, wait_until_idle = False)
```

If units are not specified, the devices operate in native units related to their drive type. The library can handle the conversion to the specific native units:


```python
# Notice the first argument of the method depends on unit type
native_units = axis.settings.convert_to_native_units("pos", 1, Units.ANGLE_DEGREES)
print("1 degree to native units:", native_units)

dps = axis.settings.convert_from_native_units("maxspeed", 10000, Units.ANGULAR_VELOCITY_DEGREES_PER_SECOND)
print("10000 native units to degrees/sec:", dps)

dps2 = axis.settings.convert_from_native_units("accel", 50, Units.ANGULAR_ACCELERATION_DEGREES_PER_SECOND_SQUARED)
print("50 native units to degrees/sec^2:", dps2)
```
