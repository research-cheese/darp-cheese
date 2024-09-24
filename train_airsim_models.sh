python src/train_airsim_models.py
python src/mariofy.py

apptainer shell --nv --bind /home/nngu2:/home/nngu2 --bind /scratch/nngu2:/scratch/nngu2 /packages/aarch64/simg/pytorch_24.05-py3.sif