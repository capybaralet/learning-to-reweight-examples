#!/bin/bash
#SBATCH --qos=unkillable                      # Ask for unkillable job
#SBATCH --cpus-per-task=2                     # Ask for 2 CPUs
#SBATCH --gres=gpu:1                          # Ask for 1 GPU
#SBATCH --mem=20G                             # Ask for 20 GB of RAM
#SBATCH --time=20:00:00                        # The job will run for 3 hours
#SBATCH -o /network/tmp1/kruegerd/slurm-%j.out  # Write the log on tmp1

# 1. Load your environment
# TODO: make sure the module loads should go here...
module load cuda/10.0/cudnn/7.6
module load python/3.7/tensorflow-gpu/1.15.0rc2
module unload python
module load miniconda/3
source $CONDA_ACTIVATE
conda activate L2R
echo 1

# 2. Copy your dataset on the compute node
cp -r /network/home/kruegerd/learning-to-reweight-examples/data/ $SLURM_TMPDIR/data
echo 2

# 3. Launch your job, tell it to save the model in $SLURM_TMPDIR
#    and look for the dataset into $SLURM_TMPDIR
python -m cifar.cifar_train_background --config /network/home/kruegerd/learning-to-reweight-examples/cifar/configs/cifar-resnet-32.prototxt --data_root $SLURM_TMPDIR/data --results $SLURM_TMPDIR/results/slurm_cifar.sh/ --baseline=False --noise_ratio=0.0 --num_clean=10000
echo 3

# 4. Copy whatever you want to save on $SCRATCH
cp $SLURM_TMPDIR/results/slurm_cifar.sh/ -r /network/tmp1/kruegerd/L2R/cifar/
echo 4
