import argparse
import copy
import json
import os
import time

import tensorboardX as tbx
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from NARF.dataset import THUmanDataset, BlenderDataset
from NARF.models.loss import SparseLoss
from NARF.models.net import NeRFGenerator, Generator, NeRFAutoEncoder
from NARF.models.foreground_model import ForegroundGenerator
from NARF.models.model_utils import random_ray_sampler, all_reduce, get_module
from NARF.utils import yaml_config, write
from NARF.visualization_utils import save_img, ssim, psnr
from train import validate, validation_func


def train(config, validation=False):
    if validation:
        dataset, data_loader = create_dataloader(config.dataset)
        validation_func(config, dataset, data_loader, rank=0, ddp=False)
    else:
        dataset, data_loader = create_dataloader(config.dataset)
        train_func(config, dataset, data_loader, rank=0, ddp=False)


def create_dataloader(config_dataset):
    batch_size = config_dataset.bs
    shuffle = True
    drop_last = True
    num_workers = config_dataset.num_workers

    dataset_train, datasets_val = create_dataset(config_dataset)
    train_loader = DataLoader(dataset_train, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle,
                              drop_last=drop_last)
    val_loaders = {key: DataLoader(datasets_val[key], batch_size=1, num_workers=num_workers, shuffle=False,
                                   drop_last=False) for key in datasets_val.keys()}
    return (dataset_train, datasets_val), (train_loader, val_loaders)


def cache_dataset(config_dataset):
    create_dataset(config_dataset, just_cache=True)


def create_dataset(config_dataset, just_cache=False):
    size = config_dataset.image_size
    dataset_name = config_dataset.name

    train_dataset_config = config_dataset.train
    val_dataset_config = config_dataset.val

    print("loading datasets")
    if dataset_name == "human":
        dataset_train = THUmanDataset(train_dataset_config, size=size, return_bone_params=True,
                                      return_bone_mask=True, random_background=False, just_cache=just_cache,
                                      load_camera_intrinsics=config_dataset.load_camera_intrinsics)
        datasets_val = dict()
        for key in val_dataset_config.keys():
            if val_dataset_config[key].data_root is not None:
                datasets_val[key] = THUmanDataset(val_dataset_config[key], size=size, return_bone_params=True,
                                                  return_bone_mask=True, random_background=False, num_repeat_in_epoch=1,
                                                  just_cache=just_cache,
                                                  load_camera_intrinsics=config_dataset.load_camera_intrinsics)
    elif dataset_name == "bulldozer":
        dataset_train = BlenderDataset(train_dataset_config, size=size, return_bone_params=True,
                                       random_background=False, just_cache=just_cache)
        datasets_val = dict()
        for key in val_dataset_config.keys():
            if val_dataset_config[key].data_root is not None:
                datasets_val[key] = BlenderDataset(val_dataset_config[key], size=size, return_bone_params=True,
                                                   random_background=False, num_repeat_in_epoch=1,
                                                   just_cache=just_cache)
    else:
        assert False, f"{dataset_name} is not supported"

    return dataset_train, datasets_val


def train_func(config, dataset, data_loader, rank, ddp=False, world_size=1):
    torch.backends.cudnn.benchmark = True

    out_dir = config.out_root
    out_name = config.out
    if rank == 0:
        writer = tbx.SummaryWriter(f"{out_dir}/runs/{out_name}")
        os.makedirs(f"{out_dir}/result/{out_name}", exist_ok=True)

    size = config.dataset.image_size
    cnn_based = False
    num_iter = config.num_iter

    dataset = dataset[0]
    num_bone = dataset.num_bone
    intrinsics = dataset.intrinsics

    gen = ForegroundGenerator(config.generator_params, size, intrinsics, num_bone,
                        ray_sampler=random_ray_sampler,
                        parent_id=dataset.output_parents)

    loss_func = SparseLoss(config.loss)

    num_gpus = torch.cuda.device_count()
    n_gpu = rank % num_gpus

    torch.cuda.set_device(n_gpu)
    gen = gen.cuda(n_gpu)

    if ddp:
        gen = torch.nn.SyncBatchNorm.convert_sync_batchnorm(gen)
        gen = nn.parallel.DistributedDataParallel(gen, device_ids=[n_gpu])

    gen_optimizer = optim.Adam(gen.parameters(), lr=config.lr, betas=(0.9, 0.999), eps=1e-12)

    if config.scheduler_gamma < 1:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(gen_optimizer, config.scheduler_gamma)

    start_time = time.time()
    iter = 0

    if config.resume or config.resume_latest:
        path = f"{out_dir}/result/{out_name}/snapshot_latest.pth" if config.resume_latest else config.resume
        if os.path.exists(path):
            snapshot = torch.load(path, map_location="cuda")
            if ddp:
                gen_module = gen.module
            else:
                gen_module = gen
            gen_module.load_state_dict(snapshot["gen"], strict=True)
            gen_optimizer.load_state_dict(snapshot["gen_opt"])
            iter = snapshot["iteration"]
            start_time = snapshot["start_time"]
            del snapshot

    train_loader, val_loaders = data_loader

    mse = nn.MSELoss()
    train_loss_color = 0
    train_loss_mask = 0

    accumulated_train_time = 0
    log = {}

    train_start = time.time()

    val_interval = config.val_interval
    print_interval = config.print_interval
    tensorboard_interval = config.tensorboard_interval
    save_interval = config.save_interval

    while iter < num_iter:
        for i, data in enumerate(train_loader):
            if rank == 0:
                print(iter)
            if (iter + 1) % print_interval == 0 and rank == 0:
                print(f"{iter + 1} iter, {(time.time() - start_time) / iter} s/iter")
            gen.train()

            batch = {key: val.cuda(non_blocking=True).float() for key, val in data.items()}
            img = batch["img"]
            mask = batch["mask"]

            if "disparity" in batch:
                disparity = batch["disparity"]
                bone_mask = batch["bone_mask"]
                part_bone_disparity = batch["part_bone_disparity"]
                keypoint_mask = batch["keypoint_mask"]

            inv_intrinsics = batch.get("inv_intrinsics")

            if "pose_to_world" in batch:
                pose_to_camera = batch["pose_to_camera"]
                pose_to_world = batch["pose_to_world"]
                bone_length = batch.get("bone_length")

            gen_optimizer.zero_grad()
            # generate image (sparse sample)
            
            # not cnn based generator
            z_f = torch.randn([1, config.generator_params.z_tri]).cuda(n_gpu)
            z_b = torch.randn([1, config.generator_params.z_tri]).cuda(n_gpu)
            nerf_color, nerf_mask, grid = gen(pose_to_camera, pose_to_world, bone_length,
                                                inv_intrinsics=inv_intrinsics,
                                                z_foreground = z_f, z_background = z_b)
            loss_color, loss_mask = loss_func(grid, nerf_color, nerf_mask, img, mask)
            loss = loss_color + loss_mask
           
            # accumulate train loss
            train_loss_color += loss_color.item() * config.dataset.bs
            train_loss_mask += loss_mask.item() * config.dataset.bs

            if (iter + 1) % tensorboard_interval == 0 and rank == 0:  # tensorboard
                write(iter, loss, "gen", writer)
            loss.backward()

            gen_optimizer.step()
            # update selector tmp
            if config.generator_params.nerf_params.selector_adaptive_tmp.gamma != 1:
                get_module(gen, ddp).enarf.update_selector_tmp()

            if config.scheduler_gamma < 1:
                scheduler.step()
            torch.cuda.empty_cache()

            if (iter + 1) % save_interval == 0 and rank == 0:                
                if ddp:
                    gen_module = gen.module
                else:
                    gen_module = gen
                save_params = {"iteration": iter,
                               "start_time": start_time,
                               "gen": gen_module.state_dict(),
                               "gen_opt": gen_optimizer.state_dict(),
                               }
                torch.save(save_params, f"{out_dir}/result/{out_name}/snapshot_latest.pth")
                torch.save(save_params, f"{out_dir}/result/{out_name}/snapshot_{(iter // 50000 + 1) * 50000}.pth")
            if (iter + 1) % val_interval == 0:
                # add train time
                accumulated_train_time += time.time() - train_start

                val_loss = validate(gen, val_loaders, config, ddp)
                torch.cuda.empty_cache()

                if ddp:
                    train_loss_color = all_reduce(train_loss_color)
                    train_loss_mask = all_reduce(train_loss_mask)

                train_loss_color = train_loss_color / (val_interval * world_size * config.dataset.bs)
                train_loss_mask = train_loss_mask / (val_interval * world_size * config.dataset.bs)

                # write log
                log_ = {"accumulated_train_time": accumulated_train_time,
                        "train_loss_color": train_loss_color,
                        "train_loss_mask": train_loss_mask}
                for key in val_loss.keys():
                    for metric in val_loss[key].keys():
                        log_[f"val_loss_{key}_{metric}"] = val_loss[key][metric]

                log[iter + 1] = log_

                if rank == 0:
                    with open(f"{out_dir}/result/{out_name}/log.json", "w") as f:
                        json.dump(log, f)

                # initialize train loss
                train_loss_color = 0
                train_loss_mask = 0

                train_start = time.time()

            iter += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="NARF/configs/default.yml")
    parser.add_argument('--default_config', type=str, default="NARF/configs/default.yml")
    parser.add_argument('--resume_latest', action="store_true")
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--validation', action="store_true")
    args = parser.parse_args()

    config = yaml_config(args.config, args.default_config, args.resume_latest, args.num_workers)

    train(config, args.validation)
