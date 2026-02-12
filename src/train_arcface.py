import argparse
import logging
import os
import shutil
import json

import numpy as np
import torch
from torch import distributed
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from backbones import get_model
from dataset import get_dataloader
from losses import CombinedMarginLoss
from lr_scheduler import PolyScheduler
from torch.optim.lr_scheduler import StepLR
from partial_fc import PartialFC, PartialFCAdamW
from utils.utils_callbacks import CallBackLogging, CallBackVerification
from utils.utils_config import get_config
from utils.utils_logging import AverageMeter, init_logging
from utils.utils_distributed_sampler import setup_seed
import math

# assert torch.__version__ >= "1.9.0", "In order to enjoy the features of the new torch, \
# we have upgraded the torch to 1.9.0. torch before than 1.9.0 may not work in the future."

try:
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    distributed.init_process_group("nccl")
except KeyError:
    world_size = 1
    rank = 0
    distributed.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:12584",
        rank=rank,
        world_size=world_size,
    )


def main(args):

    # get config
    cfg = get_config(args.config)
    # set output path
    cfg.output = args.output
    # global control random seed
    setup_seed(seed=cfg.seed, cuda_deterministic=False)

    torch.cuda.set_device(args.local_rank)

    os.makedirs(cfg.output, exist_ok=True)
    init_logging(rank, cfg.output)

    # Save config file and args for reproducibility (only on rank 0)
    if rank == 0:
        shutil.copy(args.config, os.path.join(cfg.output, os.path.basename(args.config)))
        with open(os.path.join(cfg.output, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=4)

    summary_writer = (
        SummaryWriter(log_dir=os.path.join(cfg.output, "tensorboard"))
        if rank == 0
        else None
    )
    print('device count:', torch.cuda.device_count())
    print('batch_size:', cfg.batch_size)
    train_loader,num_classes,num_image = get_dataloader(
        cfg.train_csv,
        cfg.basedir,
        args.local_rank,
        cfg.batch_size,
        cfg.dali,
        cfg.seed,
        cfg.num_workers
    )

    if args.network is not None:
        cfg.network=args.network
        
    backbone = get_model(
        cfg.network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size).cuda()

    backbone = torch.nn.parallel.DistributedDataParallel(
        module=backbone, broadcast_buffers=False, device_ids=[args.local_rank], bucket_cap_mb=16,
        find_unused_parameters=True)

    backbone.train()
    # FIXME using gradient checkpoint if there are some unused parameters will cause error
    backbone._set_static_graph()

    margin_loss = CombinedMarginLoss(
        (2**(1/2))*(math.log(num_classes-1)),
        cfg.margin_list[0],
        cfg.margin_list[1],
        cfg.margin_list[2],
        cfg.interclass_filtering_threshold
    )

    if cfg.optimizer == "sgd":
        module_partial_fc = PartialFC(
            margin_loss, cfg.embedding_size, num_classes,
            cfg.sample_rate, cfg.fp16)
        module_partial_fc.train().cuda()
        # TODO the params of partial fc must be last in the params list
        opt = torch.optim.SGD(
            params=[{"params": backbone.parameters()}, {"params": module_partial_fc.parameters()}],
            lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)

    elif cfg.optimizer == "adamw":
        module_partial_fc = PartialFCAdamW(
            margin_loss, cfg.embedding_size, num_classes,
            cfg.sample_rate, cfg.fp16)
        module_partial_fc.train().cuda()
        opt = torch.optim.AdamW(
            params=[{"params": backbone.parameters()}, {"params": module_partial_fc.parameters()}],
            lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        raise

    cfg.total_batch_size = cfg.batch_size * world_size
    cfg.warmup_step = num_image // cfg.total_batch_size * cfg.warmup_epoch
    cfg.total_step = num_image // cfg.total_batch_size * cfg.num_epoch

    lr_scheduler = PolyScheduler(
        optimizer=opt,
        base_lr=cfg.lr,
        max_steps=cfg.total_step,
        warmup_steps=cfg.warmup_step,
        last_epoch=-1
    )

    # StepLR scheduler will be created lazily when switching from PolyScheduler
    step_lr_scheduler = None
    step_lr_initialized = False

    start_epoch = 0
    global_step = 0
    if cfg.resume:
        dict_checkpoint = torch.load(os.path.join(cfg.output, f"checkpoint_gpu_{rank}.pt"), weights_only=False)
        start_epoch = dict_checkpoint["epoch"]
        global_step = dict_checkpoint["global_step"]
        backbone.module.load_state_dict(dict_checkpoint["state_dict_backbone"])
        module_partial_fc.load_state_dict(dict_checkpoint["state_dict_softmax_fc"])
        opt.load_state_dict(dict_checkpoint["state_optimizer"])
        lr_scheduler.load_state_dict(dict_checkpoint["state_lr_scheduler"])
        if "state_step_lr_scheduler" in dict_checkpoint:
            # Recreate StepLR scheduler and load its state
            step_lr_scheduler = StepLR(
                optimizer=opt,
                step_size=cfg.step_size,
                gamma=cfg.step_gamma
            )
            step_lr_scheduler.load_state_dict(dict_checkpoint["state_step_lr_scheduler"])
            step_lr_initialized = True
        del dict_checkpoint

    for key, value in cfg.items():
        num_space = 25 - len(key)
        logging.info(": " + key + " " * num_space + str(value))

    # callback_verification = CallBackVerification(
    #     val_targets=cfg.val_targets, rec_prefix=cfg.rec, summary_writer=summary_writer
    # )
    callback_logging = CallBackLogging(
        frequent=cfg.frequent,
        total_step=cfg.total_step,
        batch_size=cfg.batch_size,
        start_step = global_step,
        writer=summary_writer
    )

    loss_am = AverageMeter()
    amp = torch.amp.GradScaler('cuda', growth_interval=100)

    for epoch in range(start_epoch, cfg.num_epoch):
        print(f'Epoch {epoch + 1} / {cfg.num_epoch}')
        # Initialize StepLR scheduler when switching from PolyScheduler
        use_step_lr = cfg.step_lr_after_epoch and epoch >= cfg.step_lr_after_epoch
        if use_step_lr and not step_lr_initialized:
            # Determine starting LR based on config
            if cfg.step_lr_start == "initial":
                start_lr = cfg.lr
            elif cfg.step_lr_start == "current":
                start_lr = opt.param_groups[0]['lr']
            else:
                # Assume it's a float value
                start_lr = float(cfg.step_lr_start)

            # Set optimizer LR before creating StepLR
            for param_group in opt.param_groups:
                param_group['lr'] = start_lr

            step_lr_scheduler = StepLR(
                optimizer=opt,
                step_size=cfg.step_size,
                gamma=cfg.step_gamma
            )
            step_lr_initialized = True
            logging.info(f"Switched to StepLR at epoch {epoch} with starting LR: {start_lr}")

        if isinstance(train_loader, DataLoader):
            train_loader.sampler.set_epoch(epoch)
        for _, (img, local_labels) in enumerate(train_loader):
            global_step += 1
            local_embeddings = backbone(img)
            loss: torch.Tensor = module_partial_fc(local_embeddings, local_labels, opt)
            opt.zero_grad()

            if cfg.fp16:
                amp.scale(loss).backward()
                amp.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                amp.step(opt)
                amp.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                opt.step()

            # Use PolyScheduler (per-batch) until step_lr_after_epoch, then use StepLR (per-epoch)
            if not use_step_lr:
                lr_scheduler.step()

            current_lr = opt.param_groups[0]['lr']
            with torch.no_grad():
                loss_am.update(loss.item(), 1)
                callback_logging(global_step, loss_am, epoch, cfg.fp16, current_lr, amp)

                # if global_step % cfg.verbose == 0 and global_step > 0:
                #     callback_verification(global_step, backbone)

        # Step the StepLR scheduler at the end of each epoch (if active)
        if use_step_lr and step_lr_scheduler is not None:
            step_lr_scheduler.step()

        if cfg.save_all_states:
            checkpoint = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "state_dict_backbone": backbone.module.state_dict(),
                "state_dict_softmax_fc": module_partial_fc.state_dict(),
                "state_optimizer": opt.state_dict(),
                "state_lr_scheduler": lr_scheduler.state_dict()
            }
            if step_lr_scheduler is not None:
                checkpoint["state_step_lr_scheduler"] = step_lr_scheduler.state_dict()
            torch.save(checkpoint, os.path.join(cfg.output, f"checkpoint_gpu_{rank}.pt"))

        if False:#rank == 0 and (epoch+1)%20==0:
            path_module = os.path.join(cfg.output, f"model_{epoch+1}.pt")
            torch.save(backbone.module.state_dict(), path_module)

        if cfg.dali:
            train_loader.reset()

    if rank == 0:
        path_module = os.path.join(cfg.output, "model_last.pt")
        checkpoint = {
                # "epoch": epoch + 1,
                # "global_step": global_step,
                "state_dict_backbone": backbone.module.state_dict(),
                "state_dict_softmax_fc": module_partial_fc.state_dict(),
                # "state_optimizer": opt.state_dict(),
                # "state_lr_scheduler": lr_scheduler.state_dict()
            }
        # torch.save(backbone.module.state_dict(), path_module)
        torch.save(checkpoint, path_module)

        # from torch2onnx import convert_onnx
        # convert_onnx(backbone.module.cpu().eval(), path_module, os.path.join(cfg.output, "model.onnx"))
    distributed.destroy_process_group()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(
        description="Distributed Arcface Training in Pytorch")
    parser.add_argument("config", type=str, help="py config file")
    parser.add_argument("--local_rank", type=int, default=0, help="local_rank")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--network", type=str, default=None)
    main(parser.parse_args())
