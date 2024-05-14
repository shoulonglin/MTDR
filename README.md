# MTDR
A manifold-projection trajectory based method for chemical kinetic mechanism reduction
The MTDR was developed on python runtime environment, you need to install environment for Cantera before running it. The input parameters including a detailed mechanism file (.cti format), important species (fuel, oxidizer and products), error limit, and simulation conditions are compulsory. 
Now, you can run start_mechanism_reduction.py to start the reduction. 
Later, some output files containing important information were generated, including 
(1) the final skeletal mechanism (rd_mech.cti) 
(2) the EMT error for each species (sp-geodesic-distance-error.csv) 
(3) recordings of each step during mechanism reduction (info.log) 
(4) ignition delay time (before and after reduction) at each temperature point (output_data.csv)

The correlation functions for calculating ignition delay time, flame speed and PSR temperature are built into the folder, which you can easily use them to calculate the detailed and reduced mechanisms and thus verify the accuracy of the skeletal mechanism.
