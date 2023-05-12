import adi
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import sdr_utils as sutil
import seaborn as sns
import pandas as pd
from numpy.fft import fft, ifft


sdr = adi.ad9361(uri='ip:192.168.2.1')
samp_rate = 30.72e6    # must be <=30.72 MHz if both channels are enabled
num_samps_tx = 128e2
num_samps_rx = 200e4
#num_samps = 2**18      # number of samples per buffer.  Can be different for Rx and Tx
rx_lo = 3.5e9
rx_mode = "manual"  # can be "manual" or "slow_attack"
rx_gain0 = 0
rx_gain1 = 0
tx_lo = rx_lo
tx_gain0 = -40
tx_gain1 = -89

'''Configure Rx properties'''
sdr.rx_enabled_channels = [0, 1]
sdr.sample_rate = int(samp_rate)
sdr.rx_lo = int(rx_lo)
sdr.gain_control_mode = rx_mode
sdr.rx_hardwaregain_chan0 = int(rx_gain0)
sdr.rx_hardwaregain_chan1 = int(rx_gain1)
sdr.rx_buffer_size = int(num_samps_rx)

'''Configure Tx properties'''
sdr.tx_rf_bandwidth = int(samp_rate)
sdr.tx_lo = int(tx_lo)
sdr.tx_cyclic_buffer = True
sdr.tx_hardwaregain_chan0 = int(tx_gain0)
sdr.tx_hardwaregain_chan1 = int(tx_gain1)
sdr.tx_buffer_size = int(num_samps_tx)


f_low = 70
f_high = 6000
chirp_duration = 3e-3
dfdt = (f_high - f_low) / chirp_duration
Ts = 1/samp_rate
BW = 2e6 # Hz
# trying to generate a chirp 
t_range = np.arange(0, chirp_duration, Ts)
FM_chirp = signal.chirp(t_range, f_low, chirp_duration, f_high, method = 'linear')



tx_chirp = FM_chirp * 2**14
shaped = sutil.raised_cos_filter(tx_chirp, β = 0.35)
I = shaped
Q = np.zeros(len(shaped))
IQ_send = Q + 1j*Q
IQ2 = I + 1j*Q
#IQ_send = np.concatenate([IQ_send, IQ_send, IQ_send, IQ_send, IQ_send, IQ2, IQ2])
IQ_send = IQ2

TX2_zeros = np.zeros(len(IQ_send))

# Start the transmitter
#sdr.tx_cyclic_buffer = False # Enable cyclic buffers
sdr.tx_cyclic_buffer = True # Enable cyclic buffers
sdr.tx([IQ_send, TX2_zeros]) # start transmitting

#sdr.rx_annotated = False
for i in range(0,10):
    sdr.rx()

data_rx_0 = []
data_rx_1 = []
# Clear buffer just to be safe
# for i in range (0, 10):
#     sample_rx = sdr.rx()
#     data_rx_0 = np.concatenate([data_rx_0, sample_rx[0]])
#     data_rx_1 = np.concatenate([data_rx_1, sample_rx[1]])
for i in range(100):
    sample_rx = sdr.rx()
    data_rx_0 = sample_rx[0]
    data_rx_1 = sample_rx[1]

    mixed = data_rx_0 * np.conjugate(data_rx_1)
    sr = sdr.sample_rate
    x = mixed

    X = fft(x)
    N = len(X)
    n = np.arange(N)
    T = N/sr
    freq = n/T 

    peaks = signal.find_peaks(np.abs(X))
    #print(peaks)
    Δf = freq[peaks[0][0]]
    print(f"delta f: {Δf}")
    R = (2e8 * np.abs(Δf)) / (2*dfdt)
    print(f"Range estimate: {R}")

    # plt.stem(freq[0:100], np.abs(X[0:100]), 'b', \
    #      markerfmt=" ", basefmt="-b")
    # plt.title(f"Plot #{i}")
    # plt.xlabel('Freq (Hz)')
    # plt.ylabel('FFT Amplitude |X(freq)|')
    # plt.xlim(0, 10000)
    # plt.ylim(0, 5e11)
    plt.plot(mixed.real)
    plt.plot(mixed.imag)
    plt.pause(0.05)
    plt.clf()

plt.show()

    
    
sdr.tx_destroy_buffer()

