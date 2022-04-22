import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.constants import k
import sdr_utils as sutil
import adi
from matplotlib.widgets import Slider, Button

## SDR Setup
sample_rate = 1e6 # Hz
center_freq = 915e6 # Hz
num_samps = 100000 # number of samples per call to rx()

sdr = adi.Pluto("ip:192.168.2.1")
sdr.sample_rate = int(sample_rate)

# Config Tx
sdr.tx_rf_bandwidth = int(sample_rate) # filter cutoff, just set it to the same as sample rate
sdr.tx_lo = int(center_freq)
sdr.tx_hardwaregain_chan0 = -50 # Increase to increase tx power, valid range is -90 to 0 dB

# Config Rx
sdr.rx_lo = int(center_freq)
sdr.rx_rf_bandwidth = int(sample_rate)
sdr.rx_buffer_size = num_samps
sdr.gain_control_mode_chan0 = 'manual'
sdr.rx_hardwaregain_chan0 = 0.0 # dB, increase to increase the receive gain, but be careful not to saturate the ADC

## Using quick utilities:
bin_msg = np.array(sutil.binaryify("Hello World"))
barker_code = np.array([1,1,1,0,0,0,1,0,0,1,0]) # binary barker code
bin_msg = np.concatenate([barker_code, bin_msg])
print(f"Binary message being sent: {bin_msg}")
barker_train = sutil.pulse_train(barker_code)
train = sutil.pulse_train(bin_msg)
shaped = sutil.raised_cos_filter(train, β = 0.35)

I = shaped
Q = np.zeros(len(shaped))
IQ_send = I + 1j*Q

samples = np.repeat(IQ_send, 16) # 16 samples per symbol (rectangular pulses)
samples *= 2**14 # The PlutoSDR expects samples to be between -2^14 and +2^14, not -1 and +1 like some SDRs
print(f"sample length {len(samples)}")


# Start the transmitter
sdr.tx_cyclic_buffer = True # Enable cyclic buffers
sdr.tx(samples) # start transmitting

# Clear buffer just to be safe
for i in range (0, 10):
    raw_data = sdr.rx()

# Receive samples
rx_samples = sdr.rx()
print(f"rx sample size raw: {len(rx_samples)}")
# Stop transmitting
sdr.tx_destroy_buffer()


# plt.plot(np.real(IQ_send), '.-', zorder = 1)
# plt.plot(np.imag(IQ_send), '.-', zorder = 2)
# plt.legend(["I", "Q"])
# plt.title("Original Hello World Signal")
# plt.show()
rx_samples = sutil.clock_recovery(rx_samples)
# rx_samples = sutil.phase_detector_2(rx_samples)

print(f"rx sample size: {len(rx_samples)}")

# np.set_printoptions(threshold=np.inf)

# bin_rx = np.array(sutil.simple_ecode_recv(rx_samples))
# contains_bin = np.isin(bin_msg, bin_rx)
# print(f"Message contains binary string: {contains_bin}")
# print(f"binary message: {bin_msg}")
# print(f"binary receive: {bin_rx[-1*len(bin_msg):]}")
# bin_conv = np.convolve(bin_rx, bin_msg)
# plt.plot(bin_rx)
# plt.show()



tc = np.arange(0,2*np.pi,1e-2)
plt.plot(np.cos(tc), np.sin(tc), linewidth=1, color="grey")
plt.plot(np.real(rx_samples), np.imag(rx_samples), '.', color="red")
plt.title("IQ Plot of BPSK Message")
plt.grid(True)
# plt.xlim([-1.2, 1.2])
# plt.ylim([-1.2, 1.2])
plt.show()



plt.plot(np.real(rx_samples), '.-', zorder = 1)
plt.plot(np.imag(rx_samples), '.-', zorder = 2)
plt.legend(["I", "Q"])
plt.title("Delayed, frequency shifted, and noisey signal")
plt.show()


# Correlation graph 
rx_norm = rx_samples / np.linalg.norm(rx_samples)
barker_filered = sutil.raised_cos_filter(barker_train)
plt.plot(np.correlate(barker_filered, rx_norm,'same'),'.-')
plt.grid()
plt.title("correlated rx samples with sent IQ")
plt.show()

plt.plot(barker_filered)
plt.show()

# IQ2 = sutil.add_delay(IQ, Δsample=-1, N=100)
# IQ2 = sutil.add_frequency_offset(IQ, Δf = -100)
# plt.plot(np.real(IQ2), '.-', zorder = 1)
# plt.plot(np.imag(IQ2), '.-', zorder = 2)
# plt.legend(["I", "Q"])
# plt.title("Removed frequency shift and delay. Noise remains.")
# plt.show()

