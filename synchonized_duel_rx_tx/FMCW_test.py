import adi
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import pandas as pd
from numpy.fft import fft, ifft


sdr = adi.ad9361(uri='ip:192.168.2.1')
samp_rate = 30.72e6    # must be <=30.72 MHz if both channels are enabled
num_samps_tx = 2**18
num_samps_rx = 2**21
#num_samps = 2**18      # number of samples per buffer.  Can be different for Rx and Tx
rx_lo = 2.4e9
rx_mode = "manual"  # can be "manual" or "slow_attack"
rx_gain0 = 40
rx_gain1 = 50
tx_lo = rx_lo
tx_gain0 = -1
tx_gain1 = -89

'''Configure Rx properties'''
sdr.rx_enabled_channels = [0, 1]
sdr.sample_rate = int(samp_rate)
sdr.rx_lo = int(rx_lo)
sdr.gain_control_mode_chan0 = rx_mode
sdr.gain_control_mode_chan1 = rx_mode

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
'''Chirp Generation'''
BW = 5e6
f_low = 300
f_high = BW - f_low
chirp_duration = 8e-3
dfdt = (f_high - f_low) / chirp_duration
Ts = 1/sdr.sample_rate
t_range = np.arange(0, chirp_duration, Ts)
FM_chirp = signal.chirp(t_range, f_low, chirp_duration, f_high, method = 'linear')

tx_chirp = FM_chirp * 2**14
shaped = tx_chirp
fm_down_chirp = np.flip(shaped)
shaped = np.concatenate([shaped,-1*fm_down_chirp])
I = shaped
Q = np.zeros(len(shaped))
IQ_send = Q + 1j*Q
IQ2 = I + 1j*Q
IQ_send = IQ2
# zeros for second TX channel
TX2_zeros = np.zeros(len(IQ_send))


'''FMCW Transmission'''
print("Beginning TX...")
sdr.tx_cyclic_buffer = True # Enable cyclic buffers
sdr.tx([IQ_send, TX2_zeros]) # start transmitting

'''RX'''
print("Beginning RX...")
for i in range(0,10):
   sdr.rx()

sample_rx = sdr.rx()
data_rx_0 = sample_rx[0]
data_rx_1 = sample_rx[1]
    
'''Destroy Transmission Buffer''' 
sdr.tx_destroy_buffer()

'''Digital Mixing'''
mixed = data_rx_0 * np.conjugate(data_rx_1)

'''Filtering Mixed Signal for Beat Frequency'''
print("Filtering mixed signal...")

#num_taps = 100001
# decimating signal by 20
q = 20
mixed_decimated = signal.decimate(mixed, q)

num_taps = 10001
cut_off = 100 # Hz
sample_rate = sdr.sample_rate # Hz
h = signal.firwin(num_taps, cut_off, nyq=sample_rate/2)
mixed_h = np.convolve(mixed_decimated, h)

'''FFT Processing Beat Frequency'''
print("Processing FFT...")
# note: replace numpy FFT with scipy FFT for speed.
sr = sdr.sample_rate/q
x = mixed_h

X = fft(x)
N = len(X)
n = np.arange(N)
T = N/sr
freq = n/T 

peaks = signal.find_peaks(np.abs(X))
#print(peaks)
Δf = freq[peaks[0][0]]
print(f"frequency shift: {np.round(Δf,2)} Hz")

dfdt = (f_high - f_low)/chirp_duration
# note: the frequency shift below is halved because mixer doubles the frequency
# note: the range estimation can be improved with interpolation
R = (3e8 * (np.abs(Δf)/2)) / (2*dfdt)
print(f"Range: {np.round(R,2)} m")


plt.stem(freq, np.abs(X), 'b', \
         markerfmt=" ", basefmt="-b")
plt.xlabel('Freq (Hz)')
plt.ylabel('FFT Amplitude |X(freq)|')
plt.xlim(0, 1000)
plt.show()

plt.plot(mixed_h.real)
plt.plot(mixed_h.imag)
plt.legend(["real", "imag"])
plt.show()