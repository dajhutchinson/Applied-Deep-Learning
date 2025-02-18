# a slurm job script which you will run on BC4 to train and evaluate your network when you aren't using interactive sessions.

#!/usr/bin/env bash
#SBATCH --partition gpu
#SBATCH --time 0-00:30
#SBATCH --account comsm0045
#SBATCH --reservation comsm0045-lab5
#SBATCH --mem 64GB
#SBATCH --gres gpu:1

# get rid of any modules already loaded
module purge
# load in the module dependencies for this script
module load "languages/anaconda3/2019.07-3.6.5-tflow-1.14"

python train_cifar.py
