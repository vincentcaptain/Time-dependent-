#!/bin/bash
# Job name:
#SBATCH --job-name=time_dependent_protocol.py
#
# Account:
#SBATCH --account=co_noneq
#
# Partition:
#SBATCH --partition=savio2
#
# Wall clock limit:
#SBATCH --time=30:00:00
#
# Mail type:
#SBATCH --mail-type=all
#
# Mail user:
#SBATCH --mail-user=vincentcaptain@berkeley.edu
## Command(s) to run:
python2.7 time_dependent_protocol.py