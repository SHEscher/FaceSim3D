# # # Training script for VICE in main study within gender triplets

# Script must be run from project root "FaceSim3D":
# bash ./code/facesim3d/modeling/VICE/bash_for_training/training_vice_main_gender.sh

# # Training
# --modality (str) / # e.g., behavioral, text, visual, fMRI
# --triplets_dir (str) / # path/to/triplet/data
# --results_dir (str) / # optional specification of results directory (if not provided will resort to
#                         ./results/modality/init_dim/optim/prior/seed/spike/slab/pi)
# --plots_dir (str) / # optional specification of directory for plots (if not provided will resort to
#                       ./plots/modality/init_dim/optim/prior/seed/spike/slab/pi)
# --epochs (int) / # maximum number of epochs to run VICE optimization
# --burnin (int) / # minimum number of epochs to run VICE optimization (burnin period)
# --eta (float) / # learning rate
# --init_dim (int) / # initial dimensionality of the model's latent space
# --batch_size (int) / # mini-batch size
# --optim (str) / # optimizer (e.g., 'adam', 'adamw', 'sgd')
# --prior (str) / # whether to use a mixture of Gaussian's or Laplacian's in the spike-and-slab prior
#                   (i.e., 'gaussian' or 'laplace')
# --mc_samples (int) / # number of weight matrices used in Monte Carlo sampling (for computationally
#                        efficiency, M is set to 1 during training)
# --spike (float) / # sigma of the spike distribution
# --slab (float) / # sigma of the slab distribution
# --pi (float) / # probability value that determines the relative weighting of the distributions; the
#                  closer this value is to 1, the higher the probability that weights are drawn from the
#                  spike distribution
# --k (int) / # minimum number of objects whose weights are non-zero for a latent dimension (according to
#               importance scores)
# --ws (int) / # determines for how many epochs the number of latent dimensions (after pruning) is not
#                allowed to vary (ws >> 100)
# --steps (int) / # perform validation, save model parameters and create model and optimizer checkpoints
#                   every <steps> epochs
# --device (str) / # cuda or cpu
# --num_threads (int) / # number of threads used for intraop parallelism on CPU; use only if device is CPU
# --rnd_seed (int) / # random seed
# --verbose (bool) / # show print statements about model performance and evolution of latent dimensions
#                      during training (can be piped into log file)

# # Evaluation
# ./evaluation/evaluate_robustness.py
# --results_dir (str) / # path/to/models
# --n_objects (int) / # number of unique objects/items/stimuli in the dataset
# --init_dim (int) / # latent space dimensionality with which VICE was initialized at run time
# --batch_size (int) / # mini-batch size used during VICE training
# --thresh (float) / # Pearson correlation value to threshold reproducibility of dimensions (e.g., 0.8)
# --optim (str) / # optimizer that was used during training (e.g., 'adam', 'adamw', 'sgd')
# --prior (str) / # whether a Gaussian or Laplacian mixture was used in the spike-and-slab prior (i.e.,
#   'gaussian' or 'laplace')
# --spike (float) / # sigma of spike distribution
# --slab (float) / # sigma of slab distribution
# --pi (float) / # probability value that determines likelihood of samples from the spike
# --triplets_dir (str) / # path/to/triplet/data
# --mc_samples (int) / # number of weight matrices used in Monte Carlo sampling for evaluating models on
#   validation set
# --device (str) / # cpu or cuda
# --rnd_seed (int) / # random seed

# ><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

# Set paths & params
# n_faces=50  # number of faces per gender in main (total = 100)
p2data="data/main/for_SPoSE"
p2results="results/main/VICE"
modality="behavioral"

epochs=300
burnin=150
steps=15
eta=0.001
batch_size=16
ws=20
num_threads=1
device="cpu"

dims=30
k=5  # OR 10
seed=42

# Follow 3 hyperparameters (hp) were derived from hp-search on subset of data
spike=0.05  # default = 0.25 (lim -> 0. : more feature dimensions will be pruned)
slab=2.0  # default = 1. (lim -> inf. : this weights images (faces) with high coef. more, allows strong
          # differences within a dimension)
pi=0.4  # default, examples in README.md pi=0.6 (keep between [.4, .6])

optim="adam"  # default
prior="laplace" # L.M. recommended to try Laplacian prior; default: "gaussian"
mc_samples=10  # default = 10

# Note in Muttenthaler et al. (arXiv, 2022), page 8, who use default hyper-parameters above:
# "VICE’s performance is fairly insensitive to hyperparameter selection."

# Train VICE within gender triplets >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

for gender in "female" "male"; do
  for cond in "2D" "3D"; do
    echo "Running VICE on ${cond} data"
      python ./code/facesim3d/modeling/VICE/main.py \
      --modality ${modality} \
      --triplets_dir ${p2data}/${cond}/${gender} \
      --epochs ${epochs} --burnin ${burnin} --eta ${eta} --init_dim ${dims} --batch_size ${batch_size} \
      --k ${k} --ws ${ws} --optim ${optim} --prior ${prior} \
      --spike ${spike} --slab ${slab} --pi ${pi}  \
      --steps ${steps} --device ${device} --num_threads ${num_threads} --rnd_seed ${seed} \
      --mc_samples ${mc_samples} --verbose \
      --results_dir ${p2results}/${cond}/${gender} \
      --plots_dir ${p2results}/${cond}/${gender}/Plots

     # TODO: Evaluate robustness (i.e., check variance of VICE models with same parameters)
     # python ./code/facesim3d/modeling/VICE/evaluation/evaluate_robustness.py \
     # --triplets_dir ${p2data}/${cond}/${gender} --n_objects ${n_faces} --modality ${modality} --init_dim ${dims} \
     # --batch_size 16 --thresh 0.8 \
     # --optim ${optim} --prior ${prior} --spike ${spike} --slab ${slab} --pi ${pi} \
     # --device cpu --rnd_seed ${seed} \
     # --results_dir ${p2results}/${cond}/${gender} \
     # --mc_samples ${mc_samples}
  done
done

# ><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >> END
