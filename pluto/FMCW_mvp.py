import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.constants import k, c
import sdr_utils as sutil
import adi
from time import sleep
import os



def plot_spectrogram(title, w, fs):
    ff, tt, Sxx = signal.spectrogram(w, fs=fs, nperseg=256, nfft=576)
    plt.pcolormesh(tt, ff[:145], Sxx[:145], cmap='gray_r', shading='gouraud')
    plt.title(title)
    plt.xlabel('t (sec)')
    plt.ylabel('Frequency (Hz)')
    plt.grid()




## SDR Setup
sample_rate = 2e6 # Hz
center_freq = 2e9 # Hz
#num_samps = 1280000
num_samps = 12800 # number of samples per call to rx()

sdr = adi.Pluto("ip:192.168.2.1")
sdr.sample_rate = int(sample_rate)

# Config Tx
sdr.tx_rf_bandwidth = int(sample_rate) # filter cutoff, just set it to the same as sample rate
sdr.tx_lo = int(center_freq)
sdr.tx_hardwaregain_chan0 = 0 # Increase to increase tx power, valid range is -90 to 0 dB

# Config Rx
sdr.rx_lo = int(center_freq)
sdr.rx_rf_bandwidth = int(sample_rate)
sdr.rx_buffer_size = num_samps
sdr.gain_control_mode_chan0 = 'manual'
sdr.rx_hardwaregain_chan0 = 0.0 # dB, increase to increase the receive gain, but be careful not to saturate the ADC

sdr.loopback = 0

chirp_duration = 8e-3
Ts = 1/sample_rate
BW = 2e6 # Hz
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
IQ_send = np.concatenate([IQ_send, IQ_send, IQ2, IQ2])


# Start the transmitter
sdr.tx_cyclic_buffer = False # Enable cyclic buffers
sdr.tx(IQ_send) # start transmitting

sdr.rx_annotated = False


data = []
# Clear buffer just to be safe
for i in range (0, 10):
    data = np.concatenate([data, sdr.rx()])

# plot_spectrogram("title", data.real, num_samps)
# plt.show()


Ps = np.abs(np.fft.fftshift(np.fft.fft(IQ_send)))**2
Pr = np.abs(np.fft.fftshift(np.fft.fft(data)))**2

print(f"Power Sent: {np.sum(Ps)}")
print(f"Power Received: {np.sum(Pr)}")

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
    
# Stop transmitting
sdr.tx_destroy_buffer()

