#!/bin/bash

# To install GPAW, please follow the instructions on https://gpaw.readthedocs.io/install.html

# Run GPAW to generate calculator object
mpirun python biradical.py


# Run PyZFS to compute the ZFS tensor.

#   Method 1: With pseudo wave functions
cd ZFS
mpirun python ./Run_PyZFS_GPAW.py > ZFS_OutFile.txt

#   Method 2: With all-electron wave functions
#cd ZFS_AE_R4
#mpirun python ./Run_PyZFS_GPAW.py > ZFS_OutFile.txt


