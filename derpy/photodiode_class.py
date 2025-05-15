'''
# This file contains the photodiode class for the DERP
# All methods are compatible with the derp.py methods 
'''

#import statements
import pylablib.devices
from pylablib.devices import Thorlabs       
from pylablib.devices.Thorlabs import PM160
import subprocess as sub
import time as ti
import multiprocessing as mp
from multiprocessing import pool

#VISA string necessary for connecting to the optical power meter
VISA = 'USB0::0x1313::0x8078::P0047814::INSTR'

class OPM:

    def __init__(self, VISA):
        self.OPM = Thorlabs.PM160(VISA)


    def get_reading(self):
        '''
        returns a single power reading from the OPM; this will be called in the continuous output function

        in:
        OPM - instance of PM160 created by connect_OPM(visa_str)

        out:
        power (float) - power reading in Watts 
        '''

        power = self.OPM.get_power()

        return power
    
    def conditional_readout(self, loop_condition, start_time):
        '''
        used for the function 'continuous_OPM_readout; takes a measurement of power/time with 
            the OPM and appends it to the lists 'time_list' and 'power_list' if the 
            loop_condition is true; if the loop condition is false, it returns the full lists

        in:
        self - instance of the OPM class

        loop_condition - (bool) condition for whether the capture process is still running

        power_list - list of power readings in W taken from the OPM

        time_list - list of times (in seconds) at which the power readings were taken
        
        out:
        power_list

        time_list
        '''

        #we will check if argument 'loop_condition' is true or false; this will be determined 
        #by whether the capture process is still running

        if loop_condition == True:
            #if the capture process is running, we will take a power measurement and record the time
            power = self.get_reading()
            time = ti.time() - start_time

            return power, time
        
        elif loop_condition == False:

            #if the loop condition is false, we kill the loop and return the lists without appending anything
            print('Capture process not running...returned lists')

        
def continuous_OPM_readout(OPM, process_name):
    '''
    Funtion for reading out photodiode power continuously while some other process is 
    still running

    in:
        OPM - optical power monitor (instance of the Thorlabs.PM160 class from pylablib)
    
        process_name - (string) name of the file to be run as a subprocess; this should be a
            python file that is running the capture process

    out:
        power_list - list of power readings in Watts taken from the OPM

        time_list - list of times (in seconds) at which the power readings were taken    
    '''


    #initiate a list for power readings and times
    power_list = []
    time_list = []

    #initiate a subprocess...this should be running the capture process
    start_time = ti.time()
    process = sub.Popen(['python', process_name])

    #we will check if argument 'loop_condition' is true or false; this will be determined 
    #by whether the capture process is still running
    while process.poll() is None:
        loop_condition = True
        time,power = OPM.conditional_readout(loop_condition, start_time)

        #append the power and time to the lists
        time_list.append(time)
        power_list.append(power)

    #we need the output from the subprocess to be returned to the main process...this should be the numpy array image
    output = process.communicate()
    print('Capture process not running...returned power & time lists')
    return power_list, time_list

def _readout_worker(framegrabber_fn, done_evt, out_q):
    # Run the grabber, capture its return value
    result = framegrabber_fn()  
    out_q.put(result)           # send it back
    done_evt.set()              # then signal “I’m done”

def continuous_OPM_readout_multiprocessing(OPM, framegrabber_fn):
    '''
    Similar to the function above, but uses the multiprocessing package to call a function 
    directly instead of requiring a separate script

    Has trouble with overshooting on the time; doesn't close the loop fast enough

    Uses _readout_worker to run the framgrabber and send the frame back to this function 
    for return

    '''
    power_list = []
    time_list  = []
    start_time = ti.time()
    
    done = mp.Event()
    q    = mp.Queue()

    # spawn the worker
    p = mp.Process(target=_readout_worker,
                   args=(framegrabber_fn, done, q))
    p.start()

    # keep sampling until the worker signals done
    while p.is_alive():
        pwr, t = OPM.conditional_readout(True, start_time)
        power_list.append(pwr)
        time_list.append(t)
    
    # once p.is_alive() is False, you exit immediately
    p.join()
        

    # pull back the worker’s return value
    frame = q.get()
    return power_list, time_list, frame 

def time_test_takesawhile():
    '''wastes time so i can test if my other functions work'''
    ti.sleep(3)
    jawn = 'jawnbossey'
    return jawn

# def 



