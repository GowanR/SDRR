import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.constants import k

from sdr_utils import *

# Carrier frequency
fc = 30
BW = 30e6 # 30MHz bandwidth
T = 293.15 # temperature in kelvin

Fs = 1 # sample rate
Ts = 1/Fs

message = np.array("Hello World".encode("ascii"))
bin_v = np.vectorize(bin)
bytea_v = np.vectorize(bytearray)
int_v = np.vectorize(int)

bin_message = bin_v(bytea_v(message))

m0 = int_v(list(bin_message[0][2:]))


t = np.arange(0, Ts*len(m0), Ts/100)

bpsk_m0 = BPSK_IQV(m0)
m_values = []
for n, v in enumerate(bpsk_m0):
    m_values.append(IQ_to_time(v, fc, Ts, n))

sps = 8

m = np.concatenate(m_values)
m_bin = np.repeat(m0, 100, axis=0)

m_bpsk = BPSK(m0)

print(f"binary data: {m0}")
print(f"BPSK data: {m_bpsk}")

m_send = np.array([])
for bit in m0:
    pulse = np.zeros(sps)
    pulse[0] = bit*2-1 # set the first value to either a 1 or -1
    m_send = np.concatenate((m_send, pulse)) # add the 8 samples to the signal



## Pulse shaping
# Create our raised-cosine filter
num_taps = 101
beta = 0.35


Ts = sps # Assume sample rate is 1 Hz, so sample period is 1, so *symbol* period is 8
th = np.arange(-50, 51) # remember it's not inclusive of final number
# th = np.arange(-10, 11) # remember it's not inclusive of final number
# raised-cosine filter
h = 1/Ts*np.sinc(th/Ts) * np.cos(np.pi*beta*th/Ts) / (1 - (2*beta*th/Ts)**2)

m_shaped = np.convolve(m_send, h)

plt.figure(0)
plt.plot(m_shaped, ".-")

for i in range(len(m0)):
    plt.plot([i*sps+num_taps//2+1,i*sps+num_taps//2+1], [0, m_shaped[i*sps+num_taps//2+1]], color="red")

plt.grid(True)


# figure, axis = plt.subplots(2)
# axis[0].step(t, m0, where="post")
# axis[1].plot(t, m[:len(t)])
# plt.show()

# tc = np.arange(0,2*np.pi,1e-2)
# plt.plot(np.cos(tc), np.sin(tc), linewidth=1, color="grey")
# plt.plot(np.real(bpsk_m0), np.imag(bpsk_m0), '.', color="red")

# plt.grid(True)
# plt.xlim([-1.2, 1.2])
# plt.ylim([-1.2, 1.2])
# plt.show()



# Modeling Losses
P_noise = k*T*BW
#Random noise
N = len(m_shaped)
n = ((np.random.randn(N) + 1j*np.random.randn(N))/np.sqrt(2))
power = np.var(n)
print(f"power: {power}")
print(f"power of noise: {P_noise}")

I = np.zeros(len(m_shaped))
Q = m_shaped

IQ_send = I + 1j*Q;
IQ_recieve = IQ_send + n
plt.figure(1)
plt.plot(np.imag(IQ_recieve), ".-")

for i in range(len(m0)):
    plt.plot([i*sps+num_taps//2+1,i*sps+num_taps//2+1], [0, m_shaped[i*sps+num_taps//2+1]], color="blue")

plt.grid(True)
plt.show()