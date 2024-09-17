# python3 -m venv venv
# source venv/bin/activate

conda create -n pytorch_cuda python=3.9
conda activate pytorch_cuda

# Make sure python prints stuff
export PYTHONUNBUFFERED=TRUE
export HF_DATASETS_CACHE="/scratch/nngu2/hf-datasets"
export HF_HOME="/scratch/nngu2/.cache"

export WANDB_CACHE_DIR="/scratch/nngu2/.cache/wandb"
export WANDB_CONFIG_DIR="/scratch/nngu2/.config/wandb"
export WANDB_MODE=online

conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
conda install peft

# datasets
conda install datasets

# transformers
conda install transformers[torch]

conda install pillow
conda install albumentations
conda install timm
conda install torchmetrics
conda install dataclasses
conda install numpy
conda install pycocotools
conda install transformers
conda install wandb
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
# pip install peft

# # datasets
# pip install datasets

# # transformers
# pip install transformers[torch]

# pip install pillow
# pip install albumentations
# pip install timm
# pip install torchmetrics
# pip install dataclasses
# pip install numpy
# pip install pycocotools
# pip install transformers
# pip install wandb