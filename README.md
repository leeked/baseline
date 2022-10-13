# Spatial-Reasoning

## Collection of work done by Kevin Lee (k.lee@nyu.edu)

This repo contains the following content:

* `/NEW/`

> Newly updated files for both VGG and ViT baselines.

> Use this instead. Ignore old `/VGG16/` `/ViT_B16_224/` folders

> How to use:

> Need GPU allocated

> singularity exec --nv \
        --overlay /scratch/kl3642/train/overlay-15GB-500K.ext3:ro \
        /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
        /bin/bash -c "source /ext3/env.sh; cd /scratch/kl3642/train/spatial-reasoning; python train.py 50 --vgg --path model/"
        
> `python train.py [epoch] --[vgg or vit] --path [weights filepath]`

> `python test.py --[vgg or vit] --path [weights filepath]`


* `/groundTruth/`

> Folder containing the Ground Truth JSON files

* `/scripts/`

> Folder containing various scripts written during research period.

> These were used to create the topology graphs, RGB histograms, and datasets.

* `/VGG16/`

> IGNORE (OLD)

> This folder contains the code for the VGG16 based CNN Baseline

> Has both Python file and Notebook

* `/ViT_B16_224/`

> IGNORE (OLD)

> This folder contains the code for the Vision Transformer Baseline

> Has both Python file and Notebook
