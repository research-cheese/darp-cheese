python3 -m venv venv
source venv/bin/activate

# Make sure python prints stuff
export PYTHONUNBUFFERED=TRUE
export HF_DATASETS_CACHE="/scratch/nngu2/hf-datasets"
export WANDB_MODE=online

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