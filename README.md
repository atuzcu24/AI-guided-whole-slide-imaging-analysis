source cellvit_inf_env/bin/activate
pip install --upgrade pip
pip install -r requirements_inference.txt

python cellvit_rosie_film_inference.py   --input_dir /projectnb/ec500kb/projects/Fall_2025_Projects/Project_2/AI-guided-whole-slide-imaging-analysis/AI-GUIDED-CLEAN/ProcessedDataset/v1_40x_area20_2/patches_cellvit_p256_pannuke/fold1/images   --output_dir ./cellvit_inference_results_film256   --run_dir /projectnb/ec500kb/projects/Fall_2025_Projects/Project_2/AI-guided-whole-slide-imaging-analysis/AI-GUIDED-CLEAN/ProcessedDataset/v1_40x_area20_2/patches_cellvit_p256_pannuke/logs_local_berk/2025-12-05T191217_film256   --checkpoint model_best.pth   --magnification 40

The contents of 2025-12-05T191217_film256 file can be found in HuggingFace: BerkTuzcuBU/SamH-Rosie-FiLM-256
For docker:

docker buildx build \
  --platform linux/amd64 \
  -t cellvit-inference-amd64 \
  --load \
  .

Local:
docker run --rm --gpus all \
  -v /path/to/images:/data/input \
  -v /path/to/output:/data/output \
  -v /path/to/run_dir:/data/run \
  cellvit-inference-amd64

HuggingFace:
docker run --rm --gpus all \
  -e HF_REPO=BerkTuzcuBU/CellViT-FiLM-256-Run \
  -v /path/to/images:/data/input \
  -v /path/to/output:/data/output \
  cellvit-inference-amd64

(Hugging face overwrites local run_dir)

#To test on SCC:
docker save cellvit-inference-amd64 > cellvit-inference.tar
singularity build cellvit-inference.sif docker-archive://cellvit-inference.tar


### For training:

On SCC:
cd CellVit-plus-plus

module unload python3
module load miniconda/25.3.1

setup_scc_condarc.sh
conda --version
conda env create -f environment_verbose.yaml
conda activate cellvit_env
pip install -r requirements.txt


Check GPUs:
lspci | grep -i nvidia
module list
module avail cuda
module load cuda/12.2
nvcc --version
nvidia-smi

Based on the outputs: Tesla GPUs etc. we can use
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121

Somehow libvips module was not loaded, so I ran:
module load libvips/8.13.0

Run:
python3 check_environment.py

Otherwise, please refer to CellVit's own README.md

For singularity:

Singularity doesn't create folders, different from docker so:
mkdir -p ./cellvit_inference_results_film256

singularity run --nv \
  -B /projectnb/ec500kb/projects/Fall_2025_Projects/Project_2/AI-guided-whole-slide-imaging-analysis/AI-GUIDED-CLEAN/ProcessedDataset/v1_40x_area20_2/patches_cellvit_p256_pannuke/fold1/images:/data/input \
  -B /projectnb/ec500kb/projects/Fall_2025_Projects/Project_2/AI-guided-whole-slide-imaging-analysis/AI-GUIDED-CLEAN/ProcessedDataset/v1_40x_area20_2/patches_cellvit_p256_pannuke/logs_local_berk/2025-12-05T191217_film256:/data/run \
  -B $(pwd)/cellvit_inference_results_film256:/data/output \
  cellvit-inference.sif


In progress:

Full wsi image reading:

Using cellvit_env from cellvit, and running python3 ./cellvit/detect_cells.py   --model /projectnb/ec500kb/projects/Fall_2025_Projects/Project_2/AI-guided-whole-slide-imaging-analysis/AI-GUIDED-CLEAN/ProcessedDataset/v1_40x_area20_2/patches_cellvit_p256_pannuke/logs_local_berk/2025-12-05T191217_film256/checkpoints/model_best.pth   --outdir ./test-results/x_40/compression   --geojson   --compression   --graph   process_wsi   --wsi_path ./test_database/x40_svs/JP2K-33003-2.svs should work. But it is buggy, and still in progress.


For training:

One should have the data folder structure as follows:

fold0
fold1 (the folds will be connected to train, val, test sets in the config e.g. train: 1,  means fold1)
train_configs
dataset_config.yaml

conda env:

miniconda/25.3.1, cuda/12.2, libvips/8.13.0

conda activate cellvit_env

Connect to wandb



Running: (after doing cd /projectnb/ec500kb/projects/Fall_2025_Projects/Project_2/AI-guided-whole-slide-imaging-analysis/AI-GUIDED-CLEAN/ProcessedDataset/v1_40x_area20_2)

For 128:
python3 ../../CellViT-plus-plus/cellvit/train_cellvit.py   --config ./patches_cellvit_p128_pannuke/train_configs/cellvit_segmentation.yaml
For virchow:
python3 ../../CellViT-plus-plus/cellvit/train_cellvit.py   --config ./patches_cellvit_p128_pannuke/train_configs/cellvit_segmentation_virchow.yaml
Fusion:
python3 ../../CellViT-plus-plus/cellvit/train_cellvit.py   --config ./patches_cellvit_p128_pannuke/train_configs/samh_rosie_film.yaml

For 256:
python3 ../../CellViT-plus-plus/cellvit/train_cellvit.py   --config ./patches_cellvit_p256_pannuke/train_configs/cellvit_segmentation.yaml
For virchow:
python3 ../../CellViT-plus-plus/cellvit/train_cellvit.py   --config ./patches_cellvit_p256_pannuke/train_configs/cellvit_segmentation_virchow.yaml

Fusion:
python3 ../../CellViT-plus-plus/cellvit/train_cellvit.py   --config ./patches_cellvit_p256_pannuke/train_configs/samh_rosie_film.yaml


An example config is:

# ===============================
# Fine-tuning CellViT on TCGA Patches
# ===============================


random_seed: 19
gpu: 0

data:
  dataset: PanNuke
  dataset_path: /projectnb/ec500kb/projects/Fall_2025_Projects/Project_2/AI-guided-whole-slide-imaging-analysis/AI-GUIDED-CLEAN/ProcessedDataset/v1_40x_area20_2/patches_cellvit_p256_pannuke
  input_shape: 256
  train_folds: [0]
  val_folds: [1]
  num_nuclei_classes: 6
  num_tissue_classes: 1
  #val_patch_size: 256

dataloader:
  train:
    num_workers: 6        # try 4–8; lower if the node is busy
    pin_memory: true
    persistent_workers: true
    drop_last: true       # avoids tiny tail batches
  val:
    num_workers: 4
    pin_memory: true
    persistent_workers: true
    drop_last: true

model:
  backbone: sam-h-rosie-film
  pretrained_encoder: /projectnb/ec500kb/projects/Fall_2025_Projects/Project_2/AI-guided-whole-slide-imaging-analysis/AI-GUIDED-CLEAN/CellViT-plus-plus/checkpoints/SAM/encoder_only_CellViT-SAM-H-x40-AMP.pth
  #pretrained_encoder: /projectnb/ec500kb/projects/Fall_2025_Projects/Project_2/AI-guided-whole-slide-imaging-analysis/CellViT-plus-plus/checkpoints/Virchow/CellViT-Virchow-x40-AMP.pth
  rosie_hidden_dim: 256

fusion:
  freeze_cellvit: true
  freeze_rosie: true

training:
  drop_rate: 0
  attn_drop_rate: 0.1
  drop_path_rate: 0.1
  batch_size: 2
  epochs: 80
  optimizer: AdamW
  early_stopping_patience: 80
  scheduler:
    scheduler_type: cosine
    warmup_epochs: 2
  optimizer_hyperparameter:
    lr: 0.00003
  unfreeze_epoch: 0
  sampling_strategy: random
  eval_every: 1 # Ideally we want to set 1 but it gives memory issues
  mixed_precision: true
  resume: true
  

# For starting from a checkpoint
checkpoint: null
just_load_model: false


loss:
  nuclei_type_map:
    bce:
      loss_fn: xentropy_loss
      weight: 0.5
    dice:
      loss_fn: dice_loss
      weight: 0.2
    mcfocaltverskyloss:
      loss_fn: MCFocalTverskyLoss
      weight: 0.5
      args:
        num_classes: 6

transformations:
  randomrotate90:
    p: 0.5
  horizontalflip:
    p: 0.5
  verticalflip:
    p: 0.5
  downscale:
    p: 0.15
    scale: 0.5
  blur:
    p: 0.2
    blur_limit: 10
  gaussnoise:
    p: 0.25
    var_limit: 50
  colorjitter:
    p: 0.2
    scale_setting: 0.25
    scale_color: 0.1
  superpixels:
    p: 0.1
  zoomblur:
    p: 0.1
  randomsizedcrop:
    p: 0.1
  elastictransform:
    p: 0.2
  normalize:
    mean:
    - 0.5
    - 0.5
    - 0.5
    std:
    - 0.5
    - 0.5
    - 0.5

eval_checkpoint: null #For finetuning we need this null
run_sweep: false
agent: null

dataset_config:
  tissue_types:
    Lung: 0
  nuclei_types:
    background: 0
    epithelial: 1
    lymphocyte: 2
    macrophage: 3
    neutrophil: 4
    other: 5



# --- Logging ---
logging:
  project: samh-rosie-film
  mode: online                  # or 'offline' if no internet
  wandb_dir: /projectnb/ec500kb/projects/Fall_2025_Projects/Project_2/AI-guided-whole-slide-imaging-analysis/AI-GUIDED-CLEAN/ProcessedDataset/v1_40x_area20_2/patches_cellvit_p256_pannuke/logs_local_berk
  log_dir: /projectnb/ec500kb/projects/Fall_2025_Projects/Project_2/AI-guided-whole-slide-imaging-analysis/AI-GUIDED-CLEAN/ProcessedDataset/v1_40x_area20_2/patches_cellvit_p256_pannuke/logs_local_berk
  notes: First film try rosie and samh early fusion
  log_comment: First film try same with patch size 256
  level: debug
  log_images: true


  




