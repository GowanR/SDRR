# SDRR - Software Defined Range Radar using PlutoSDR


This code was for my EE senior design capstone with the goal of enabling the Adalm Pluto SDR to function as an FMCW range radar. This project was successful with tested results. 

In order for the radar to function correctly, the Pluto needs to be modified to enable the 2nd rx and tx ports as shown in [this video](https://youtu.be/ph0Kv4SgSuI). Once both rx/tx pairs are available, insert a small loopback cable between one of the rx/tx pairs and connect the other pair to the antennas. This is done because by default, the py-adi-iio does not provide internal timing syncronization. However, because the ADCs are clocked together, the loopback cable provides the signal reference required for mixing. 

The final code for the FMCW radar is in the `/FMCW/` directory. 