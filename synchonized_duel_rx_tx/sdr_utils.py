import numpy as np

bin_v = np.vectorize(bin)
bytea_v = np.vectorize(bytearray)
int_v = np.vectorize(int)

def BPSK_IQ(bit):
    x = int(bit) * np.pi
    return np.cos(x) + 1j*np.sin(x)


BPSK_IQV = np.vectorize(BPSK_IQ)

def IQ_to_time(IQ_sample, fc, Ts, n):
    Tsn = Ts/100 # 100 samples of sinusoid per Ts
    I = np.real(IQ_sample)
    Q = np.imag(IQ_sample)
    t = np.arange(n*Ts,(n+1)*Ts,Tsn)
    M = np.sqrt(I**2 + Q**2)
    Phi = np.arctan2(Q, I)
    return M*np.cos(2*np.pi*fc*t + Phi)

IQ_to_time_v = np.vectorize(IQ_to_time)

def BPSK_convert(bit):
    return 1 if bit else -1

BPSK = np.vectorize(BPSK_convert)


"""
Simple function that spits out an array of ones and zeroes that represent a string of data
"""
def binaryify(message_str):
    message = np.array(message_str.encode("ascii"))
    bin_message = bin_v(bytea_v(message))
    msg = np.array([])
    for i in range(len(bin_message)):
        a = int_v(list(bin_message[i][2:]))
        msg = np.concatenate((msg, a), axis=None)
    return msg

"""
Creates an array of integers  from a binary message. 1 = 1, 0 = -1, and fills padding zeroes between pulses.
"""
def pulse_train(binary_message, padding_zeroes = 8):
    m_send = np.array([]) # raw impulse train with filler zeroes
    for bit in binary_message:
        pulse = np.zeros(padding_zeroes)
        pulse[0] = bit*2-1 # set the first value to either a 1 or -1
        m_send = np.concatenate((m_send, pulse)) # add the 8 samples to the signal
    
    return m_send

"""
Applied raised cosine filter to message to reduce bandwidth
"""
def raised_cos_filter(message, β=0.35, sps=8, num_taps=101):
    beta = β
    ## Pulse shaping
    # Create our raised-cosine filter
    #beta = 0.35
    Ts = sps # Assume sample rate is 1 Hz, so sample period is 1, so *symbol* period is 8
    th = np.arange(-1*(num_taps//2), (num_taps//2)+1) # remember it's not inclusive of final number
    # th = np.arange(-10, 11) # remember it's not inclusive of final number
    # raised-cosine filter
    h = 1/Ts*np.sinc(th/Ts) * np.cos(np.pi*beta*th/Ts) / (1 - (2*beta*th/Ts)**2)
    h*=Ts # added this to scale to unity
    m_shaped = np.convolve(message, h)
    return m_shaped

"""
Adds a delay to message to simulate SDR random delays
"""
def add_delay(message, Δsample=0.4, N=21):
    n = np.arange(-N//2, N//2) # ...-3,-2,-1,0,1,2,3...
    h = np.sinc(n - Δsample) # calc filter taps
    h *= np.hamming(N) # window the filter to make sure it decays to 0 on both sides
    h /= np.sum(h) # normalize to get unity gain, we don't want to change the amplitude/power
    return np.convolve(message, h) # apply filter


"""
Adds frequency offset to message to simulate VCO precision errors and drift
"""
def add_frequency_offset(message_IQ, fs = 1e6, Δf=10e3):
    fo = Δf #13000 # simulate freq offset
    Ts = 1/fs # calc sample period
    t = np.arange(0, Ts*len(message_IQ), Ts) # create time vector
    return message_IQ * np.exp(1j*2*np.pi*fo*t) # perform freq shift


"""
Adds gaussian noise to the message with specified signal noise radio assuming unity signal power
"""
def add_AWGN(message_IQ, SNR=10):
    P_signal = 1
    P_noise = P_signal/SNR
    N = len(message_IQ)
    # Generate AWGN
    n = ((np.random.randn(N) + 1j*np.random.randn(N))/np.sqrt(2)) * P_noise
    return message_IQ + n

def clock_recovery(samples, sps = 8):
    ## Mueller and Muller clock recovery technique
    mu = 0 # initial estimate of phase of sample
    out = np.zeros(len(samples) + 10, dtype=np.complex)
    out_rail = np.zeros(len(samples) + 10, dtype=np.complex) # stores values, each iteration we need the previous 2 values plus current value
    i_in = 0 # input samples index
    i_out = 2 # output index (let first two outputs be 0)
    while i_out < len(samples) and i_in+16 < len(samples):
        out[i_out] = samples[i_in + int(mu)] # grab what we think is the "best" sample
        out_rail[i_out] = int(np.real(out[i_out]) > 0) + 1j*int(np.imag(out[i_out]) > 0)
        x = (out_rail[i_out] - out_rail[i_out-2]) * np.conj(out[i_out-1])
        y = (out[i_out] - out[i_out-2]) * np.conj(out_rail[i_out-1])
        mm_val = np.real(y - x)
        mu += sps + 0.3*mm_val
        i_in += int(np.floor(mu)) # round down to nearest int since we are using it as an index
        mu = mu - np.floor(mu) # remove the integer part of mu
        i_out += 1 # increment output index
    out = out[2:i_out] # remove the first two, and anything after i_out (that was never filled out)
    samples = out # only include this line if you want to connect this code snippet with the Costas Loop later on
    return samples

# For QPSK
def phase_detector_4(sample):
    if sample.real > 0:
        a = 1.0
    else:
        a = -1.0
    if sample.imag > 0:
        b = 1.0
    else:
        b = -1.0
    return a * sample.imag - b * sample.real

# For BPSK
def phase_detector_2(sample):
    # if sample.real > 0:
    #     a = 1.0
    # else:
    #     a = -1.0
    # b = 0
    # return a * sample.imag - b * sample.real
    if sample.real > 0:
        a = 1.0
    else:
        a = -1.0
    if sample.imag > 0:
        b = 1.0
    else:
        b = -1.0
    return a * sample.imag - b * sample.real

phase_detector_2 = np.vectorize(phase_detector_2)

def simple_ecode_recv(samples):
    b_msg = (samples > 0).astype(int)
    return b_msg

def plot_IQ(plt, IQ_data, title="No Title"):
    tc = np.arange(0,2*np.pi,1e-2)
    plt.plot(np.cos(tc), np.sin(tc), linewidth=1, color="grey")
    plt.plot(np.real(IQ_data), np.imag(IQ_data), '.', color="red")
    plt.title(title)
    plt.grid(True)
    plt.show()

def plot_IQ_samples(plt, IQ_data, title="No Title"):
    plt.plot(np.real(IQ_data), '.-', zorder = 1)
    plt.plot(np.imag(IQ_data), '.-', zorder = 2)
    plt.legend(["I", "Q"])
    plt.show()