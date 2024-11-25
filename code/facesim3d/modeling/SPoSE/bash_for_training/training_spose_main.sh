# # Training script for SPoSE in main experiment

# Script must be run from project root "FaceSim3D":
# bash ./code/facesim3d/modeling/SPoSE/bash_for_training/training_spose_main.sh

# --task (specify whether you'd like the model to perform an odd-one-out (i.e., 3AFC) or similarity (i.e.,
#   2AFC) task)
# --modality (define for which modality specified task should be performed by SPoSE (e.g., behavioral,
#   text, visual))
# --triplets_dir (in case you have tripletized data, provide directory from where to load triplets)
# --results_dir (optional specification of results directory (if not provided will resort to
#   ./results/modality/version/dim/lambda/seed/))
# --plots_dir (optional specification of directory for plots (if not provided will resort to
#   ./plots/modality/version/dim/lambda/seed/)
# --learning_rate (learning rate to be used in optimizer)
# --lmbda (lambda value determines l1-norm fraction to regularize loss; will be divided by number of items
#   in the original data matrix)
# --embed_dim (embedding dimensionality, i.e., output size of the neural network)
# --batch_size (batch size)
# --epochs (maximum number of epochs to optimize SPoSE model for)
# --window_size (window size to be used for checking convergence criterion with linear regression)
# --sampling_method (sampling method; if soft, then you can specify a fraction of your training data to be
#   sampled from during each epoch; else full train set will be used)
# --steps (save model parameters and create checkpoints every <steps> epochs)
# --resume (bool) (whether to resume training at last checkpoint; if not set training will restart)
# --p (fraction of train set to sample; only necessary for *soft* sampling)
# --device (CPU or CUDA)
# --rnd_seed (random seed)
# --early_stopping (bool) (train until convergence)
# --num_threads (number of threads used by PyTorch multiprocessing)

# Set paths & params
p2data="data/main/for_SPoSE"
p2results="results/main/SPoSE"
modality="behavioral"

device="cpu"
num_threads=1
batch_size=16
epochs=300
steps=15
dims=30
seed=42
window_size=10
learning_rate=0.001
# lmbda=0.005

for cond in "2D" "3D"
do
  for lmbda in 0.005 0.01 0.05 0.1 0.5 1
  do
    echo "Running SPoSE on ${cond} data"

    # Train SPoSE
    python ./code/facesim3d/modeling/SPoSE/train.py \
    --task odd_one_out --modality ${modality} \
    --triplets_dir ${p2data}/${cond} \
    --learning_rate ${learning_rate} --lmbda ${lmbda} --embed_dim ${dims} \
    --epochs ${epochs} --window_size ${window_size} --steps ${steps} --device ${device} \
    --rnd_seed ${seed} --num_threads ${num_threads} --early_stopping --batch_size ${batch_size} \
    --results_dir ${p2results}/${cond}/${modality}/${dims}d/${lmbda}/seed${seed} \
    --plots_dir ${p2results}/${cond}/${modality}/${dims}d/${lmbda}/seed${seed}/Plots
    # --resume --sampling_method soft --p 0.8
  done
done

# ><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >> END
