# Local-Global Self-Supervised Visual Representation Learning

PyTorch implementation for Patch-Wise DINO. For details, see **Local-Global Self-Supervised Visual Representation Learning**.  
 [[`arXiv`](http://arxiv.org/abs/2310.18651)]

<div align="center">
  <img width="100%" alt="DINO illustration" src="./.github/Patch-Wise DINO.gif">
</div>

## Training

### Documentation
This codebase has been developed on top of the official [DINO](https://github.com/facebookresearch/dino/) implementation. Please install [PyTorch](https://pytorch.org/) and download the [Cifar10](https://www.cs.toronto.edu/~kriz/cifar.html), [ImageNet-100](https://www.kaggle.com/datasets/ambityga/imagenet100), and [ImageNet-1K](https://imagenet.stanford.edu/) dataset. It has been developed with python version 3.10, PyTorch version 2.0.1, CUDA 12.0 and torchvision 0.15.2. After activating the virtual environment, please copy the transforms.py inside the repository into the virtual_env/Lib/site-packages/torchvision/transforms/transforms.py. For a glimpse at the arguments of the proposed Patch-Wise DINO framework training please run:
```
python patch_wise_dino.py --help
```

### Vanilla Patch-Wise DINO training
Run DINO with ViT-small network on a single node with 4 GPUs for 100 epochs with the following command.
```
torchrun --nproc_per_node=4 patch_wise_dino.py --arch vit_small --data_path /path/to/imagenet-or-imagenet100-or-cifar10/train --output_dir /path/to/saving_dir
```

## Evaluation: k-NN classification on ImageNet
To evaluate a simple k-NN classifier with a single GPU on a pre-trained model, run:
```
torchrun --nproc_per_node=1 eval_knn.py --data_path /path/to/imagenet-or-imagenet100-or-cifar10
```
If you choose not to specify `--pretrained_weights`, then DINO reference weights are used by default. If you want instead to evaluate checkpoints from a run of your own, you can run for example:
```
torchrun --nproc_per_node=1 eval_knn.py --pretrained_weights /path/to/checkpoint.pth --checkpoint_key teacher --data_path /path/to/imagenet-or-imagenet100-or-cifar10 
```

## Evaluation: Linear classification on ImageNet
To train a supervised linear classifier on frozen weights on a single node with 4 gpus, run:
```
torchrun --nproc_per_node=4 eval_linear.py --data_path /path/to/imagenet-or-imagenet100-or-cifar10
```


## Evaluation: Image Retrieval on revisited Oxford and Paris
Step 1: Prepare revisited Oxford and Paris by following [this repo](https://github.com/filipradenovic/revisitop).

Step 2: Image retrieval (if you do not specify weights with `--pretrained_weights` then by default [DINO weights pretrained on Google Landmark v2 dataset](https://dl.fbaipublicfiles.com/dino/dino_vitsmall16_googlelandmark_pretrain/dino_vitsmall16_googlelandmark_pretrain.pth) will be used).

Paris:
```
torchrun --use_env --nproc_per_node=1 eval_image_retrieval.py --imsize 512 --multiscale 1 --data_path /path/to/revisited_paris_oxford/ --dataset rparis6k
```

Oxford:
```
torchrun --use_env --nproc_per_node=1 eval_image_retrieval.py --imsize 224 --multiscale 0 --data_path /path/to/revisited_paris_oxford/ --dataset roxford5k
```


## Evaluation: Copy detection on Copydays
Step 1: Prepare [Copydays dataset](https://lear.inrialpes.fr/~jegou/data.php#copydays).

Step 2 (opt): Prepare a set of image distractors and a set of images on which to learn the whitening operator.
In our paper, we use 10k random images from YFCC100M as distractors and 20k random images from YFCC100M (different from the distractors) for computing the whitening operation.

Step 3: Run copy detection:
```
torchrun --use_env --nproc_per_node=1 eval_copy_detection.py --data_path /path/to/copydays/ --whitening_path /path/to/whitening_data/ --distractors_path /path/to/distractors/
```


### Patch-Matching
To see the result of the patch-matching algorithm:
upload your image with the name test.png in the directory of the repository and run:
```
python patch_matching.py
```
It generates an image in same directory with the name result.png and prints the corresponding patches in the console.


## Citation
If you find this repository useful, please consider giving a star :star: and citation :t-rex::
```
@inproceedings{ali2023local-global,
  title={Local-Global Self-Supervised Visual Representation Learning},
  author={Javidani, Ali and Sadeghi, Mohammad Amin and Nadjar Araabi, Babak},
  booktitle={arXiv},
  year={2023}
}
```
