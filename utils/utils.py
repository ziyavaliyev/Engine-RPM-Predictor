import numpy as np
from scipy.signal import zpk2tf
from scipy.signal import lfilter, bilinear
import sys
import os
import matplotlib.pyplot as plt
from datetime import datetime
import folium
import json

def training_details(logging, model, optimizer, criterion, num_epochs, learning_rate, dir, figure):

    logging.info(f"Model Architecture: {model}")
    logging.info("Layers and Parameters:")
    model_params = {}
    for name, param in model.named_parameters():
        param_size = list(param.size())
        logging.info(f"{name} - {param_size}")
        model_params[name] = param_size

    logging.info(f"Optimizer: {optimizer}")
    logging.info(f"Learning Rate: {learning_rate}")
    logging.info(f"Loss Function: {criterion}")
    logging.info(f"Number of Epochs: {num_epochs}")

    details = {
        "Model Architecture": str(model),
        "Layers and Parameters": model_params,
        "Optimizer": str(optimizer),
        "Learning Rate": learning_rate,
        "Loss Function": str(criterion),
        "Number of Epochs": num_epochs
    }

    file_name = f"training_details.json"
    file_path = f"{dir}/{file_name}"

    with open(file_path, 'w') as json_file:
        json.dump(details, json_file, indent=4)
    
    filename = f'eval_plot.png'
    filepath = os.path.join(dir, filename)
    plt.savefig(filepath)

    logging.info(f"Training details saved to {file_path}")

def convert_to_decimal(degrees, minutes):
    """
    Convert GPS coordinates from degrees and minutes to decimal degrees.
    
    params:
    degrees (float): The degree part of the coordinate.
    minutes (float): The minutes part of the coordinate.
    
    return:
    float: The coordinate in decimal degrees.
    """
    return degrees + minutes / 60.0

def plot_gps_data(gps_data):
    """
    Plot the GPS data on a map.
    
    Args:
    gps_data (list): A list of dictionaries containing GPS data.
    """
    route = []
    for entry in gps_data:
        lat_degrees = int(entry['Lat'] / 100)
        lat_minutes = entry['Lat'] % 100
        lon_degrees = int(entry['Lon'] / 100)
        lon_minutes = entry['Lon'] % 100
        latitude = convert_to_decimal(lat_degrees, lat_minutes)
        longitude = convert_to_decimal(lon_degrees, lon_minutes)

        route.append((latitude, longitude))

    # Create a map centered around the first coordinate in the route
    m = folium.Map(location=route[0], zoom_start=6)

    # Add the route as a PolyLine
    folium.PolyLine(route, color="blue", weight=2.5, opacity=1).add_to(m)

    # Add markers for start and end point in the route
    folium.Marker(location=route[0], popup="Start", icon=folium.Icon(color='green')).add_to(m)
    folium.Marker(location=route[-1], popup="End", icon=folium.Icon(color='red')).add_to(m)

    # Save the map to an HTML file
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("maps", exist_ok=True)
    m.save(f"maps/route_map_{current_time}.html")

def volt2pressure(volt, gain, sensitivity=-35, dBref=94):  
    """
    Convert Volts to instantaneous sound pressure (p [Pa]).
    
    Parameters
    ----------
    volt : ndarray-like or scalar
        ndarray-like or scalar containing the sound waveform in volt
        
    gain : integer
        Total gain applied to the sound (preamplifer + amplifier)
    
    sensitivity : float, optional, default is -35 (dB/V)
        Sensitivity of the microphone
    
    dBref : integer, optional, default is 94 (dBSPL)
        Pressure sound level used for the calibration of the microphone 
        (usually 94dB, sometimes 114dB)
                
    Returns
    -------
    p : ndarray-like or scalar
        ndarray-like or scalar containing the sound waveform in pressure (Pa)
        
    Examples
    --------
    >>> w, fs = maad.sound.load('../data/cold_forest_daylight.wav') 
    >>> v = maad.spl.wav2volt(wave=w)
    >>> maad.spl.volt2pressure(volt=v, gain=42) 
        array([ 0.00962983,  0.01148374,  0.01153826, ..., -0.00370198,
       -0.00195712, -0.00337482])      
    
    Same result with the function wav2pressure
    
    >>> maad.spl.wav2pressure(wave=w, gain=42)
        array([ 0.00962983,  0.01148374,  0.01153826, ..., -0.00370198,
       -0.00195712, -0.00337482])
    
    """
    # force to be ndarray
    volt = np.asarray(volt)
    
    # wav to instantaneous sound pressure level (SPL)
    # coefficient to convert Volt into pressure (Pa)
    coeff = 1/10**(sensitivity/20) 
    p = volt * coeff / 10**(gain/20)
    return p

def pressure2dBSPL (p, pRef=20e-6): 
    """
    Convert sound pressure (p [Pa]) to sound pressure level (L [dB]).
    
    Parameters
    ----------
    p : ndarray-like or scalar
        Array or scalar containing the sound pressure in Pa 
                
    pRef : Sound pressure reference in the medium (air:20e-6 Pa, water:1e-6 Pa)
                
    Returns
    -------
    L : ndarray-like or scalar
        Array or scalar containing the sound pressure level (L [dB])
        

    Examples
    --------
    >>> w, fs = maad.sound.load('../data/cold_forest_daylight.wav') 
    >>> p = maad.spl.wav2pressure(wave=w, gain=42)
    
    Get instantaneous sound pressure level (L)
    
    >>> maad.spl.pressure2dBSPL(abs(p))
        array([53.65176859, 55.18106489, 55.22220942, ..., 45.3480775 ,
        39.81175156, 44.54440589])
    
    Get equivalent sound pressure level (Leq) from the RMS of the pressure signal
    
    >>> p_rms = maad.util.rms(p)
    >>> maad.spl.pressure2dBSPL(p_rms)  
        54.53489077674256      
        
    """    
    # force to be ndarray
    p = np.asarray(p)
    
    # if p ==0 set to MIN
    p[p==0] = sys.float_info.min
    
    # Take the log of the ratio pressure/pRef
    L = 20*np.log10(p/pRef) 
    return L

def time_weighted_sound_level(pressure, sample_frequency, integration_time, reference_pressure=2.0e-5):
    """Time-weighted sound pressure level.

    :param pressure: Dynamic pressure.
    :param sample_frequency: Sample frequency.
    :param integration_time: Integration time.
    :param reference_pressure: Reference pressure.
    """
    levels = 10.0 * np.log10(integrate(pressure**2.0, sample_frequency, integration_time) / reference_pressure**2.0)
    times = np.arange(levels.shape[-1]) * integration_time
    return times, levels


def integrate(data, sample_frequency, integration_time):
    """Integrate the sound pressure squared using exponential integration.

    :param data: Energetic quantity, e.g. :math:`p^2`.
    :param sample_frequency: Sample frequency.
    :param integration_time: Integration time.
    :returns:

    Time weighting is applied by applying a low-pass filter with one real pole at :math:`-1/\\tau`.

    .. note::

        Because :math:`f_s \\cdot t_i` is generally not an integer, samples are discarded.
        This results in a drift of samples for longer signals (e.g. 60 minutes at 44.1 kHz).

    """
    integration_time = np.asarray(integration_time)
    #sample_frequency = np.asarray(sample_frequency)
    samples = data.shape[-1]
    b, a = zpk2tf([1.0], [1.0, integration_time], [1.0])
    b, a = bilinear(b, a, fs=sample_frequency)
    #b, a = bilinear([1.0], [1.0, integration_time], fs=sample_frequency) # Bilinear: Analog to Digital filter.
    n = np.floor(integration_time * sample_frequency).astype(int)
    data = data[..., 0:n * (samples // n)]
    newshape = list(data.shape[0:-1])
    newshape.extend([-1, n])
    data = data.reshape(newshape)
    #data = data.reshape((-1, n)) # Divide in chunks over which to perform the integration.
    return lfilter(
        b, a,
        data)[..., n - 1] / integration_time  # Perform the integration. Select the final value of the integration.


def fast(data, fs):
    """Apply fast (F) time-weighting.

    :param data: Energetic quantity, e.g. :math:`p^2`.
    :param fs: Sample frequency.

    .. seealso:: :func:`integrate`

    """
    return integrate(data, fs, 0.125)
    #return time_weighted_sound_level(data, fs, FAST)


def slow(data, fs):
    """Apply slow (S) time-weighting.

    :param data: Energetic quantity, e.g. :math:`p^2`.
    :param fs: Sample frequency.

    .. seealso:: :func:`integrate`

    """
    return integrate(data, fs, 1.000)
    #return time_weighted_sound_level(data, fs, SLOW)


def fast_level(data, fs):
    """Time-weighted (FAST) sound pressure level.

    :param data: Dynamic pressure.
    :param fs: Sample frequency.

    .. seealso:: :func:`time_weighted_sound_level`

    """
    return time_weighted_sound_level(data, fs, 0.125)


def slow_level(data, fs):
    """Time-weighted (SLOW) sound pressure level.

    :param data: Dynamic pressure.
    :param fs: Sample frequency.

    .. seealso:: :func:`time_weighted_sound_level`

    """
    return time_weighted_sound_level(data, fs, 1.000)
def create_dir():
    # get the current time
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    # save in plots folder
    parent_dir = os.path.join(os.getcwd(), "plots")
    os.makedirs(parent_dir, exist_ok=True)
    
    # create timestamp based folders
    timestamped_dir = os.path.join(parent_dir, f"plots_{current_time}")
    os.makedirs(timestamped_dir, exist_ok=True)
    return timestamped_dir

def create_dir_ml():
    # get the current time
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    # save in plots folder
    parent_dir = os.path.join(os.getcwd(), "plots")
    os.makedirs(parent_dir, exist_ok=True)
    ml_dir = os.path.join(parent_dir, "ml_trainings")
    os.makedirs(ml_dir, exist_ok=True)
    
    # create timestamp based folders
    timestamped_dir = os.path.join(ml_dir, f"training_{current_time}")
    os.makedirs(timestamped_dir, exist_ok=True)
    return timestamped_dir
    

def save_plot(plot_name, timestamped_dir):
    file_path = os.path.join(timestamped_dir, plot_name + ".png")

    plt.grid(True)
    
    plt.savefig(file_path)

    plt.clf()