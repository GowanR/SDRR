import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.constants import k, c
import sdr_utils as sutil
import adi
from time import sleep
import os



class PowerRadar():
    def __init__(self):
        ## SDR Setup
        self.sample_rate = 2e6 # Hz
        center_freq = 2e9 # Hz
        #num_samps = 1280000
        self.num_samps = 12800 # number of samples per call to rx()

        self.sdr = adi.Pluto("ip:192.168.2.1")
        self.sdr.sample_rate = int(self.sample_rate)

        # Config Tx
        self.sdr.tx_rf_bandwidth = int(self.sample_rate) # filter cutoff, just set it to the same as sample rate
        self.sdr.tx_lo = int(center_freq)
        self.sdr.tx_hardwaregain_chan0 = 0 # Increase to increase tx power, valid range is -90 to 0 dB

        # Config Rx
        self.sdr.rx_lo = int(center_freq)
        self.sdr.rx_rf_bandwidth = int(self.sample_rate)
        self.sdr.rx_buffer_size = self.num_samps
        self.sdr.gain_control_mode_chan0 = 'manual'
        self.sdr.rx_hardwaregain_chan0 = 0.0 # dB, increase to increase the receive gain, but be careful not to saturate the ADC
        
        self.IQ_send = self.create_send_signal()

    def create_send_signal(self):
        chirp_duration = 8e-3
        Ts = 1/self.sample_rate
        # trying to generate a chirp 
        t_range = np.arange(0, chirp_duration, Ts)
        FM_chirp = signal.chirp(t_range, 325, chirp_duration, 3800, method = 'linear')
        #FM_chirp = np.sin(t_range)

        tx_chirp = FM_chirp * 2**14
        shaped = sutil.raised_cos_filter(tx_chirp, β = 0.35)
        I = shaped
        Q = np.zeros(len(shaped))
        IQ_send = Q + 1j*Q
        IQ2 = I + 1j*Q
        #IQ2 = Q + 1j*I
        IQ_send = np.concatenate([IQ_send, IQ_send, IQ2, IQ2])
        return IQ_send
    
    def clear_buffer(self, n=10):
        # Clear buffer just to be safe
        for i in range (0, n):
            raw_data = self.sdr.rx()
    
    def get_received_power(self, t=0.05):
        # Start the transmitter
        self.sdr.tx_cyclic_buffer = False # Enable cyclic buffers
        self.sdr.tx(self.IQ_send) # start transmitting
        self.sdr.rx_annotated = False
        #self.clear_buffer()
        data = []
        
        for i in range (0, 10):
            data = np.concatenate([data, self.sdr.rx()])
        
        self.sdr.tx_destroy_buffer()
        psd = np.abs(np.fft.fftshift(np.fft.fft(data)))**2
        return np.sum(psd) ** (1/4)

        n = int(np.round((t * self.sample_rate)/self.num_samps ))
        data = []

        for i in range(0, n):
            self.clear_buffer()
            rx_samples = self.sdr.rx()
            psd = np.abs(np.fft.fftshift(np.fft.fft(rx_samples)))**2
            estimate_power = np.sum(psd)# / len(psd)
            data.append(estimate_power)

        self.sdr.tx_destroy_buffer()
        return data
        




# for i in range (0, 10000):
#     # Receive samples
#     rx_samples = sdr.rx()
#     # print(f"rx sample size raw: {len(rx_samples)}")
#     # Calculate power spectral density (frequency domain version of signal)
#     psd = np.abs(np.fft.fftshift(np.fft.fft(rx_samples)))**2
#     psd_dB = 10*np.log10(psd)
#     estimate = (np.sum(psd)) #** (1/4)
#     print(estimate / 1e9)
#     print(np.sum(np.abs(rx_samples)))
#     print( "#" * int(estimate / 1e10))
#     sleep(0.1)
#     os.system('clear')
    
# # Stop transmitting
