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
# Processors:
#SBATCH --nodes=5
#SBATCH --exclusive
#
# Wall clock limit:
#SBATCH --time=12:00:00
#
## Command(s) to run:
python2.7 time_dependent_protocol.py
