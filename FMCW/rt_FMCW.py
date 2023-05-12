from FMCW import FMCW
import adi
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

sns.set()
sdr = adi.ad9361(uri='ip:192.168.2.1')

sdr.rx_rf_bandwidth = int(10e6)
radar = FMCW(sdr)

radar.enable_cont_op() # enable continuous operation
#plt.axis([0, 10, 0, 1])

for i in range(30):
    start_time = time.time()
    dist, energy = radar.get_range() # get range and energy arrays
    print(f"Sample took {time.time() - start_time} seconds.")
    plt.clf()
    plt.xlim(-10, 500)
    plt.plot(dist, energy)
    plt.pause(0.5)

plt.show()

radar.disable_cont_op()