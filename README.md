# MTDR
A manifold-projection trajectory based method for chemical kinetic mechanism reduction.
The MTDR was developed in a Python runtime environment. Before running it, you need to install the Cantera environment. The reqired input parameters including a detailed mechanism file (.cti format), important species (fuel, oxidizer and products), an error limit, and simulation conditions. 
To begin the reduction process, run start_mechanism_reduction.py. 
After running the program, several output files containing important information will be generated:
(1) The final skeletal mechanism (rd_mech.cti) 
(2) The EMT error for each species (sp-geodesic-distance-error.csv) 
(3) Recordings of each step during mechanism reduction (info.log) 
(4) Ignition delay time (before and after reduction) at each temperature point (output_data.csv)

The related functions for calculating ignition delay time, flame speed and PSR temperature are built into the folder. You can easily use them to calculate the combustion parameters for both detailed and reduced mechanisms, thereby verifying the accuracy of the reduced mechanism.
