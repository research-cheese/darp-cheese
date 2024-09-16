python3 -m venv venv
source venv/bin/activate

# Make sure python prints stuff
export PYTHONUNBUFFERED=TRUE
export HF_DATASETS_CACHE="/scratch/nngu2/hf-datasets"
export HF_HOME="/scratch/nngu2/.cache"

export WANDB_CACHE_DIR="/scratch/nngu2/.cache/wandb"
export WANDB_CONFIG_DIR="/scratch/nngu2/.config/wandb"
export WANDB_MODE=online

conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install peft

# datasets
pip install datasets

# transformers
pip install transformers[torch]

pip install pillow
pip install albumentations
pip install timm
pip install torchmetrics
pip install dataclasses
pip install numpy
pip install pycocotools
pip install transformers
pip install wandb