# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models as torchvision_models

import utils
import vision_transformer as vits
from vision_transformer import DINOHead
from augment import augmented_crop, correspondences
import math
from functools import partial
import matplotlib.pyplot as plt
import random
import warnings
warnings.filterwarnings("ignore")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def show_batch_images(images, batch_size):
    fig = plt.figure(figsize=(8, 8))
    columns = batch_size
    rows = 10
    for i in range(1, columns+1):
        for j in range(1, rows+1):
            fig.add_subplot(rows, columns, (j-1)*columns + (i))
            plt.imshow(images[j-1][i-1].permute(1, 2, 0))
    # plt.show()
    plt.savefig('foo.png')


def DataAugmentationDINO(args, image, seed):
    flip_and_color_jitter = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply(
        [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
        p=0.8
    ),
    transforms.RandomGrayscale(p=0.2),
    ])
    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # first global crop
    global_transfo1 = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=args.global_crops_scale, interpolation=Image.BICUBIC),
        flip_and_color_jitter,
        utils.GaussianBlur(1.0),
        normalize,
    ])
    # second global crop
    global_transfo2 = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=args.global_crops_scale, interpolation=Image.BICUBIC),
        flip_and_color_jitter,
        utils.GaussianBlur(0.1),
        utils.Solarization(0.2),
        normalize,
    ])
    # transformation for the local small crops
    local_transfo = transforms.Compose([
        transforms.RandomResizedCrop(96, scale=args.local_crops_scale, interpolation=Image.BICUBIC),
        flip_and_color_jitter,
        utils.GaussianBlur(p=0.5),
        normalize,
    ])

    set_seed(seed)
    global1 = augmented_crop(global_transfo1, image, patch_size=args.patch_size, global_scale=args.global_scale, local_scale=args.local_scale)

    set_seed(seed+1)
    global2 = augmented_crop(global_transfo2, image, patch_size=args.patch_size, global_scale=args.global_scale, local_scale=args.local_scale)

    local_augmented_crops = [] 
    for j in range(args.local_crops_number):
        set_seed(seed+2+j)
        local_augmented_crops.append(augmented_crop(local_transfo, image, patch_size=args.patch_size, global_scale=args.global_scale, local_scale=args.local_scale))

    augmented_crops = [global1, global2] + local_augmented_crops
    return augmented_crops

# def custom_collate(data):
#     augmented_crops = []
#     labels = []

#     for i in range(len(data)):
#         augmented_crops.append(data[i][0])
#         labels.append(data[i][1])
#     return augmented_crops, labels

@profile
def collate_function(batch, additional_arg):
    process_seed = random.randint(0, 1000000)

    # Separate the samples and targets
    samples, targets = zip(*batch)

    # Apply augmentations to each sample within the batch
    augmented_samples = []

    for i, sample in enumerate(samples):
        augmented_sample = DataAugmentationDINO(additional_arg, sample, process_seed)
        augmented_samples.append(augmented_sample)
    # show_images(augmented_samples, additional_arg.batch_size_per_gpu)
    
    # Decompose data:
    images =[
        torch.empty((len(augmented_samples),3,args.global_scale,args.global_scale)),torch.empty((len(augmented_samples),3,args.global_scale,args.global_scale)),
        torch.empty((len(augmented_samples),3,args.local_scale,args.local_scale)),torch.empty((len(augmented_samples),3,args.local_scale,args.local_scale)),
        torch.empty((len(augmented_samples),3,args.local_scale,args.local_scale)),torch.empty((len(augmented_samples),3,args.local_scale,args.local_scale)),
        torch.empty((len(augmented_samples),3,args.local_scale,args.local_scale)),torch.empty((len(augmented_samples),3,args.local_scale,args.local_scale)),
        torch.empty((len(augmented_samples),3,args.local_scale,args.local_scale)),torch.empty((len(augmented_samples),3,args.local_scale,args.local_scale)),
    ]
    for i in range(len(augmented_samples[0])):
        for j in range(len(augmented_samples)):
            images[i][j] = augmented_samples[j][i].crop_tensor_normed.detach().clone()

    corrs = [[None for _ in range(additional_arg.global_crops_number + additional_arg.local_crops_number)] for _ in range(additional_arg.global_crops_number)]

    # Calculate patch correspondences for the last image in the batch
    # which is also equal to other images in the batch:

    for iq in range(additional_arg.global_crops_number):
        for v in range(additional_arg.global_crops_number + additional_arg.local_crops_number):
            if v == iq:
                # we skip cases where student and teacher operate on the same view
                continue

            corrs[iq][v] = correspondences(augmented_sample[iq], augmented_sample[v])
    
    # Return the augmented samples, correspondences, and targets as a batch
    return images, corrs, targets


os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "nccl"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def save_arguments_to_json(args, filename):
    arguments = vars(args)  # Get the arguments as a dictionary
    with open(filename, 'w') as f:
        json.dump(arguments, f, indent=4)

torchvision_archs = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

def get_args_parser():
    parser = argparse.ArgumentParser('DINO', add_help=False)

    # Model parameters
    arch = parser.add_argument('--arch', default='vit_tiny', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base', 'xcit', 'deit_tiny', 'deit_small'] \
                + torchvision_archs + torch.hub.list("facebookresearch/xcit:main"),
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    patch_size = parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""") #################
    out_dim = parser.add_argument('--out_dim', default=1000, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
        help="Whether to use batch normalizations in projection head (Default: False)")

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # Training/Optimization parameters
    fp16 = parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    batch_size = parser.add_argument('--batch_size_per_gpu', default=40, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')######################
    parser.add_argument('--epochs', default=111, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")

    # Multi-crop parameters
    parser.add_argument('--global_crops_number', type=int, default=2, help="""Number of big
        global views to generate.""")
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=8, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")

    # Misc
    # ImageNet path: /amin/imagenet/imagenet/train
    # Cifar10 path:/home/alij/Datasets/Cifar10/train
    # Imagenet100 path: /home/alij/Datasets/Imagenet100/train

    parser.add_argument('--data_path', default='/home/alij/Datasets/Cifar10/train', type=str,
        help='Please specify path to the ImageNet training data.')
    # parser.add_argument('--output_dir', default=f"/home/alij/RESULTS/Cifar10/Ours/Network_Checkpoints/mean_patch{patch_size.default}_out{out_dim.default}_{arch.default[4:]}_fp{16 if fp16.default else 32}_batch{batch_size.default}_ours_same_batch_augmentation_on_cpu", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--output_dir', default=f"/home/alij/RESULTS/Cifar10/Ours/Network_Checkpoints/ali_batch40_on_cpu", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=20, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")

    # My added arguments:
    parser.add_argument("--global_scale", default=224, type=int, help="size of upsampled image after crop to be fed to the teacher")
    parser.add_argument("--local_scale", default=96, type=int, help="size of upsampled image after crop to be fed to the student")

    return parser


def train_dino(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ preparing data ... ============

    dataset = datasets.ImageFolder(args.data_path, transform=None)
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    partial_collate_fn = partial(collate_function, additional_arg=args)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=partial_collate_fn
    )
    print(f"Data loaded: there are {len(dataset)} images.")

    # ============ building student and teacher networks ... ============
    # we changed the name DeiT-S for ViT-S to avoid confusions
    args.arch = args.arch.replace("deit", "vit")
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.arch in vits.__dict__.keys():
        student = vits.__dict__[args.arch](
            patch_size=args.patch_size,
            drop_path_rate=args.drop_path_rate,  # stochastic depth
        )
        teacher = vits.__dict__[args.arch](patch_size=args.patch_size)
        embed_dim = student.embed_dim
    # if the network is a XCiT
    elif args.arch in torch.hub.list("facebookresearch/xcit:main"):
        student = torch.hub.load('facebookresearch/xcit:main', args.arch,
                                 pretrained=False, drop_path_rate=args.drop_path_rate)
        teacher = torch.hub.load('facebookresearch/xcit:main', args.arch, pretrained=False)
        embed_dim = student.embed_dim
    # otherwise, we check if the architecture is in torchvision models
    elif args.arch in torchvision_models.__dict__.keys():
        student = torchvision_models.__dict__[args.arch]()
        teacher = torchvision_models.__dict__[args.arch]()
        embed_dim = student.fc.weight.shape[1]
    else:
        print(f"Unknow architecture: {args.arch}")

    # multi-crop wrapper handles forward with inputs of different resolutions
    student = utils.MultiCropWrapper(student, DINOHead(
        embed_dim,
        args.out_dim,
        use_bn=args.use_bn_in_head,
        norm_last_layer=args.norm_last_layer,
    ))
    teacher = utils.MultiCropWrapper(
        teacher,
        DINOHead(embed_dim, args.out_dim, args.use_bn_in_head),
    )
    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()
    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict())
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")

    # ============ preparing loss ... ============
    dino_loss = DINOLoss(
        args.patch_size,
        args.out_dim,
        args.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
    ).cuda()

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                               args.epochs, len(data_loader))
    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        dino_loss=dino_loss,
    )
    start_epoch = to_restore["epoch"]

    start_time = time.time()
    print("Starting DINO training !")
    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)

        # ============ training one epoch of DINO ... ============
        train_stats = train_one_epoch(student, teacher, teacher_without_ddp, dino_loss,
            data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
            epoch, fp16_scaler, args)

        # ============ writing logs ... ============
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'dino_loss': dino_loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

@profile
def train_one_epoch(student, teacher, teacher_without_ddp, dino_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule,epoch,
                    fp16_scaler, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    for it, (images, corrs, _) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # show_batch_images(data, args.batch_size_per_gpu)

        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # # Decompose data:
        # images =[
        #     torch.empty((len(data),3,args.global_scale,args.global_scale)),torch.empty((len(data),3,args.global_scale,args.global_scale)),
        #     torch.empty((len(data),3,args.local_scale,args.local_scale)),torch.empty((len(data),3,args.local_scale,args.local_scale)),
        #     torch.empty((len(data),3,args.local_scale,args.local_scale)),torch.empty((len(data),3,args.local_scale,args.local_scale)),
        #     torch.empty((len(data),3,args.local_scale,args.local_scale)),torch.empty((len(data),3,args.local_scale,args.local_scale)),
        #     torch.empty((len(data),3,args.local_scale,args.local_scale)),torch.empty((len(data),3,args.local_scale,args.local_scale)),
        # ]
        # for i in range(len(data[0])):
        #     for j in range(len(data)):
        #         images[i][j] = data[j][i].crop_tensor_normed.detach().clone()

        # move images to gpu
        images = [im.cuda(non_blocking=True) for im in images]
        # teacher and student forward passes + compute dino loss
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            teacher_output = teacher(images[:2], args.batch_size_per_gpu, args.local_crops_number, args.patch_size, args.global_scale, args.local_scale)  # only the 2 global views pass through the teacher
            student_output = student(images, args.batch_size_per_gpu, args.local_crops_number, args.patch_size, args.global_scale, args.local_scale)
            loss = dino_loss(student_output, teacher_output, corrs, epoch)

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class DINOLoss(nn.Module):
    def __init__(self, patch_size, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, pow(int(224/patch_size),2) + 1, out_dim))#
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, corrs, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        total_loss_mean = 0
        total_loss_sum = 0
        n_loss_terms = 0
        lamda = 0.9

        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                
                # # Calculate patch correspondences for the first image in the batch
                # # which is also equal to other images in the batch:
                # corr = correspondences(data[0][iq], data[0][v])

                tensor1 = teacher_out[iq][:, corrs[iq][v].selected_crop1_patches, :]
                tensor2 = student_out[v][:, corrs[iq][v].selected_crop2_patches, :]

                # print('iq=', iq, 'v=', v, 'k=', k)
                # print(corr.selected_crop1_patches)
                # print(corr.selected_crop2_patches)

                # Calculate Loss:
                tensor2_softmax = F.log_softmax(tensor2, dim=-1)
                cross_entropy_loss = - tensor1 * tensor2_softmax
                loss_sum = torch.sum(cross_entropy_loss, dim=-1)
                # step_loss = loss_sum.sum()

                # total_loss_mean += loss_sum.mean()
                total_loss_sum += loss_sum.mean()
                # n_loss_terms += 1

                #Method3 loss function (mean):
                # total_loss_sum += loss_sum.sum()

                #Method2 loss function:
                # if len(loss_sum) == 1:
                #     total_loss_sum += loss_sum[0]
                # elif len(loss_sum) > 1:
                #     total_loss_sum += lamda * loss_sum[0] * (len(loss_sum)-1) + (1-lamda)*(loss_sum[1:].sum())

                #Method1 loss function:
                # total_loss_sum += lamda * loss_sum[0] 
                # if len(loss_sum) > 1:
                #     total_loss_sum += (1-lamda)*(loss_sum[1:].mean())
                
                n_loss_terms += 1
        total_loss = total_loss_sum / n_loss_terms
                
        # total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

        # Original Dino Loss Function:
        # student_out_softmax = F.log_softmax(student_out[v], dim=-1)
        # cross_entropy_loss = -q * student_out_softmax
        # loss_sum = torch.sum(cross_entropy_loss, dim=-1)
        # total_loss += loss_sum.mean()
        # total_loss /= n_loss_terms
        # self.update_center(teacher_output)
        # return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


# class DataAugmentationDINO(object):
#     def __init__(self, global_scale, local_scale, global_crops_scale, local_crops_scale, local_crops_number):
#         flip_and_color_jitter = transforms.Compose([
#             transforms.RandomHorizontalFlip(p=0.5),
#             transforms.RandomApply(
#                 [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
#                 p=0.8
#             ),
#             transforms.RandomGrayscale(p=0.2),
#         ])
#         normalize = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#         ])####################################################################

#         # first global crop
#         self.global_transfo1 = transforms.Compose([
#             transforms.RandomResizedCrop(global_scale, scale=global_crops_scale, interpolation=Image.BICUBIC),
#             flip_and_color_jitter,
#             utils.GaussianBlur(1.0),
#             normalize,
#         ])
#         # second global crop
#         self.global_transfo2 = transforms.Compose([
#             transforms.RandomResizedCrop(global_scale, scale=global_crops_scale, interpolation=Image.BICUBIC),
#             flip_and_color_jitter,
#             utils.GaussianBlur(0.1),
#             utils.Solarization(0.2),
#             normalize,
#         ])
#         # transformation for the local small crops
#         self.local_crops_number = local_crops_number
#         self.local_transfo = transforms.Compose([
#             transforms.RandomResizedCrop(local_scale, scale=local_crops_scale, interpolation=Image.BICUBIC),
#             flip_and_color_jitter,
#             utils.GaussianBlur(p=0.5),
#             normalize,
#         ])

#     def __call__(self, image):
#         global1 = augmented_crop(self.global_transfo1, image, patch_size=args.patch_size, global_scale=args.global_scale, local_scale=args.local_scale)
#         global2 = augmented_crop(self.global_transfo2, image, patch_size=args.patch_size, global_scale=args.global_scale, local_scale=args.local_scale)
        
#         local_augmented_crops = []       
#         for _ in range(self.local_crops_number):
#             local_augmented_crops.append(augmented_crop(self.local_transfo, image, patch_size=args.patch_size, global_scale=args.global_scale, local_scale=args.local_scale))

#         augmented_crops = [global1, global2] + local_augmented_crops
#         return augmented_crops


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    # Save the arguments to a JSON file
    save_arguments_to_json(args, os.path.join(args.output_dir,'arguments.json'))
    train_dino(args)
