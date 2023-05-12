import adi
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.constants import c
from scipy.fft import fft, ifft


"""
FMCW Radar implementation.

Version 1.11

Written by Gowan Rowland

==== CHANGELOG ====
Verions 1.11 patches issue with peak unity normalization. 

Version 1.1 release brings additional features to the `get_range()` function including a normalization flag to make the largest peak 
unity and range boundaries in feet.

Version 1.0 initial release.

"""
class FMCW:
    """
    Radar constructor.
    length_unit can be "feet" to return values in feet, or anything else to return meters.
    mode can be "loopback" or "coupler". Loopback is for a short loopback cable directly from RX to TX and coupler
    refers to a directional RF coupler for sampling the sent signal. 
    """
    def __init__(self, sdr, mode="loopback", length_unit = "feet"):
        self.cont_op = False # continuous operation state variable
        self.dk = 1.5
        self.sdr = sdr
        self.mode = mode
        self.samp_rate = 30.72e6
        self.length_unit = length_unit
        num_samps_tx = 2**18
        #num_samps_rx = 2**21
        #num_samps_rx = 2**22
        num_samps_rx = 2**22
        rx_lo = 800e6
        #rx_lo = 2.4e9
        rx_mode = "manual" 
        rx_gain0 = 40
        rx_gain1 = 40
        tx_lo = rx_lo
        tx_gain0 = -20
        tx_gain1 = -20
        '''Configure Rx properties'''
        self.sdr.rx_enabled_channels = [0, 1]
        self.sdr.sample_rate = int(self.samp_rate)
        self.sdr.rx_lo = int(rx_lo)
        self.sdr.gain_control_mode_chan0 = rx_mode
        self.sdr.gain_control_mode_chan1 = rx_mode

        self.sdr.rx_hardwaregain_chan0 = int(rx_gain0)
        self.sdr.rx_hardwaregain_chan1 = int(rx_gain1)
        self.sdr.rx_buffer_size = int(num_samps_rx)

        '''Configure Tx properties'''
        self.sdr.tx_rf_bandwidth = int(8e6)
        self.sdr._set_iio_attr_int("voltage1", "rf_bandwidth", True, int(8e6))
        #self.sdr.tx_rf_bandwidth = int(self.samp_rate)
        self.sdr.tx_lo = int(tx_lo)
        self.sdr.tx_cyclic_buffer = True
        self.sdr.tx_hardwaregain_chan0 = int(tx_gain0)
        self.sdr.tx_hardwaregain_chan1 = int(tx_gain1)
        self.sdr.tx_buffer_size = int(num_samps_tx)
        TX = self.generate_chirp(mode=self.mode)
        self.TX1 = TX[0]
        self.TX2 = TX[1]
    """
    This function is used to reset the transmitter filter cutoff frequencies. 
    """
    def reset_tx_filter(self, BW = 8e6):
        self.sdr.tx_rf_bandwidth = int(BW)
        self.sdr._set_iio_attr_int("voltage1", "rf_bandwidth", True, int(BW))
    """
    This function generates the FMCW transmission chirp
    """
    def generate_chirp(self, f_low = 300, BW = 8e6, mode="loopback"):
        f_high = BW - f_low
        #chirp_duration = 8e-3
        chirp_duration = 4e-3
        self.dfdt = (f_high - f_low) / chirp_duration
        Ts = 1/self.sdr.sample_rate
        t_range = np.arange(0, chirp_duration, Ts)
        FM_chirp = signal.chirp(t_range, f_low, chirp_duration, f_high, method = 'linear')


        upchirp = FM_chirp * 2**14
        fm_down_chirp = np.flip(upchirp)
        chirp = np.concatenate([upchirp,-1*fm_down_chirp])
        I = chirp 
        Q = np.zeros(len(chirp))
        IQ_send = Q + 1j*Q
        IQ2 = I + 1j*Q
        IQ_send = IQ2
        TX2_zeros = np.zeros(len(IQ_send))
        if (mode == "loopback"):
            return [IQ_send, IQ_send]
        elif (mode == "coupler"):
            return [IQ_send, TX2_zeros]

    """
    set_mode can be used to change the mode after the radar has been constructed.
    """
    def set_mode(self, mode):
        self.mode = mode
        TX = self.generate_chirp(mode=self.mode)
        self.TX1 = TX[0]
        self.TX2 = TX[1]
    """
    calc_range takes the mixed signal, passes it through an fft, and returns the range data and energy data.

    range_data is an array containing the discrete ranges
    energy is an array that describes the energy at each corresponding range index
    """
    def calc_range(self, mixed):
        energy = fft(mixed)
        N = len(mixed)
        n = np.arange(N)
        T = N/self.samp_rate
        freq = n/T 
        c_adjusted = c / np.sqrt(self.dk)
        range_data = (freq * c_adjusted)/(self.dfdt)# * 2)
        range_resolution = ((1/T)*c_adjusted)/(self.dfdt)
        max_range = 100 # m
        range_data = range_data[0:int(max_range/range_resolution)]
        energy = energy[0:int(max_range/range_resolution)]
        energy = energy/energy.sum()
        return range_data, energy
    
    """
    This function turns on TX, clears the RX buffer, and gets the mixed signal of the RX ports. Returns the mixed signal.
    """
    def get_mixed_burst(self):
        self.sdr.tx_cyclic_buffer = True # Enable cyclic buffers
        self.sdr.tx([self.TX1, self.TX2]) # start transmitting
        for i in range(0,10):
            self.sdr.rx()

        sample_rx = self.sdr.rx()
        data_rx_0 = sample_rx[0]
        data_rx_1 = sample_rx[1]
        self.sdr.tx_destroy_buffer()
        mixed = data_rx_0 * np.conjugate(data_rx_1)
        return mixed
    """
    This function is used to get the range and energy arrays in a single burst.
    """
    def get_range_burst(self):
        mixed = self.get_mixed_burst()
        range_data, energy = self.calc_range(mixed)
        if self.length_unit == "feet":
            range_data = range_data / 0.3048 # convert meters to feet
        return range_data, np.abs(energy)

    """
    This function turns on the continuous operation mode. Turns cyclic buffer on and sends out chips until continuous operation is disabled.
    """
    def enable_cont_op(self):
        self.sdr.tx_cyclic_buffer = True # Enable cyclic buffers
        self.sdr.tx([self.TX1, self.TX2]) # start transmitting
        # clear buffer
        for i in range(0,10):
            self.sdr.rx()
        self.cont_op = True
    """
    This functino disables the continuous operation mode.
    """
    def disable_cont_op(self):
        self.sdr.tx_destroy_buffer()
        self.cont_op = False

    """
    Gets the range data when radar is in continuous transmission operation.
    returns range and energy arrays

    when normalize argument flag is true, the peak of the energy return will be unity.
    the max_range argument specifies the max range to return in feet 
    """
    def get_range(self, normalize=True, min_range = 10, max_range=300):
        if not self.cont_op:
            raise Exception("Continuous transmission must be enabled before sampling range. Try burst mode or enable continuous operation.")
        sample_rx = self.sdr.rx()
        data_rx_0 = sample_rx[0]
        data_rx_1 = sample_rx[1]
        mixed = data_rx_0 * np.conjugate(data_rx_1)
        range_data, energy = self.calc_range(mixed)
        if self.length_unit == "feet":
            range_data = range_data / 0.3048 # convert meters to feet
        
        cutoff_index = np.abs(range_data - max_range).argmin()
        min_index = np.abs(range_data - min_range).argmin()
        range_data = range_data[min_index:cutoff_index]
        energy = energy[min_index:cutoff_index]
        energy = np.abs(energy)
        if normalize:
            energy = energy/np.max(energy)
        
        return range_data, energy 