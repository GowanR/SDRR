from PowerRadar import PowerRadar
import numpy as np


radar = PowerRadar()

for i in range(0,100):
    pwr = radar.get_received_power()
    avg_pwr = np.mean(pwr)
    print(avg_pwr)