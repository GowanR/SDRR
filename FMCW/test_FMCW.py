from FMCW import FMCW
import adi
import matplotlib.pyplot as plt
import seaborn as sns

"""
This is a test script for the FMCW radar V1.0
"""

def main():
    print("Starting FMCW radar test...")


    sns.set()
    sdr = adi.ad9361(uri='ip:192.168.2.1')
    radar = FMCW(sdr)

    radar.enable_cont_op() # enable continuous operation
    dist1c, energy1c = radar.get_range() # get range and energy arrays
    radar.disable_cont_op()
    dist1, energy1 = radar.get_range_burst()

    print("Change loopback cable length.")
    print("Press Enter to continue...")
    input()

    radar.enable_cont_op()
    dist2c, energy2c = radar.get_range() # get range and energy arrays
    radar.disable_cont_op() # disable continuous operation

    dist2, energy2 = radar.get_range_burst()

    figure, axis = plt.subplots(2, 1)

    axis[0].plot(dist1, energy1)
    axis[0].plot(dist2, energy2)
    axis[0].set_xlim(0, 500)
    axis[0].set_xlabel("Range (ft)")
    axis[0].set_ylabel("Energy")


    axis[1].plot(dist1c, energy1c)
    axis[1].plot(dist2c, energy2c)
    axis[1].set_xlabel("Range (ft)")
    axis[1].set_ylabel("Energy")

    axis[0].legend(["Cable 1 Burst", "Cable 2 Burst"], loc=1)
    axis[1].set_xlim(0, 500)

    axis[1].legend(["Cable 1 Continuous", "Cable 2 Continuous"], loc=1)
    plt.show()

if __name__ == "__main__":
    main()