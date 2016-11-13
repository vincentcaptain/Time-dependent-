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
python2.7 -i time_dependent_protocol.py
omega = list(range(1, 75))
o = [-i/10 for i in omega] + [i / 10 for i in omega]
omega = sorted(o)
num_cores = multiprocessing.cpu_count()
sample_size = 20
interval = [-3, 3]
steps = 60
starting = -3.5
final_p = Parallel(n_jobs = num_cores)(delayed(total_prob)(i, sample_size, interval, steps, starting) for i in omega)
flux = Parallel(n_jobs = num_cores)(delayed(process_initial_flux)(i, sample_size, starting, interval[0]) for i in omega)
np.savetxt("final_p.txt", final_p)
np.savetxt("flux.txt", flux)