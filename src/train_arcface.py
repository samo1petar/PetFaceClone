import argparse
import logging
import os
import shutil
import json
import math

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.functional import linear, normalize

from backbones import get_model
from dataset import Train
from losses import CombinedMarginLoss
from lr_scheduler import PolyScheduler
from torch.optim.lr_scheduler import StepLR
from utils.utils_callbacks import CallBackLogging
from utils.utils_config import get_config
from utils.utils_logging import AverageMeter, init_logging
from utils.utils_distributed_sampler import setup_seed


class ArcFaceHead(torch.nn.Module):
    """Single-GPU replacement for PartialFC with sample_rate=1.0."""

    def __init__(self, margin_loss, embedding_size, num_classes, fp16=False):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.normal(0, 0.01, (num_classes, embedding_size)))
        self.margin_softmax = margin_loss
        self.fp16 = fp16
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, embeddings, labels):
        labels = labels.squeeze().long()
        with torch.amp.autocast('cuda', enabled=self.fp16):
            norm_embeddings = normalize(embeddings)
            norm_weight = normalize(self.weight)
            logits = linear(norm_embeddings, norm_weight)
        if self.fp16:
            logits = logits.float()
        logits = logits.clamp(-1, 1)
        logits = self.margin_softmax(logits, labels.view(-1, 1))
        loss = self.ce(logits, labels)
        return loss


def make_dataloader(csv_path, basedir, batch_size, num_workers, shuffle=True):
    dataset = Train(train_csv=csv_path, basedir=basedir, local_rank=0)
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return loader, dataset.num_classes, len(dataset)


def main(args):

    # get config
    cfg = get_config(args.config)
    # set output path
    cfg.output = args.output
    # global control random seed
    setup_seed(seed=cfg.seed, cuda_deterministic=False)

    torch.cuda.set_device(0)

    os.makedirs(cfg.output, exist_ok=True)
    init_logging(0, cfg.output)

    # Save config file and args for reproducibility
    shutil.copy(args.config, os.path.join(cfg.output, os.path.basename(args.config)))
    with open(os.path.join(cfg.output, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    summary_writer = SummaryWriter(log_dir=os.path.join(cfg.output, "tensorboard"))

    print('batch_size:', cfg.batch_size)
    train_loader, num_classes, num_image = make_dataloader(
        cfg.train_csv, cfg.basedir, cfg.batch_size, cfg.num_workers, shuffle=True)

    val_loader = None
    if cfg.val_csv:
        val_loader, _, _ = make_dataloader(
            cfg.val_csv, cfg.basedir, cfg.batch_size, cfg.num_workers, shuffle=False)

    if args.network is not None:
        cfg.network=args.network
        
    backbone = get_model(
        cfg.network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size).cuda()

    backbone.train()

    margin_loss = CombinedMarginLoss(
        (2**(1/2))*(math.log(num_classes-1)),
        cfg.margin_list[0],
        cfg.margin_list[1],
        cfg.margin_list[2],
        cfg.interclass_filtering_threshold
    )

    module_fc = ArcFaceHead(
        margin_loss, cfg.embedding_size, num_classes, cfg.fp16)
    module_fc.train().cuda()

    if cfg.optimizer == "sgd":
        opt = torch.optim.SGD(
            params=[{"params": backbone.parameters()}, {"params": module_fc.parameters()}],
            lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "adamw":
        opt = torch.optim.AdamW(
            params=[{"params": backbone.parameters()}, {"params": module_fc.parameters()}],
            lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        raise

    cfg.total_batch_size = cfg.batch_size
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
        dict_checkpoint = torch.load(os.path.join(cfg.output, "checkpoint.pt"), weights_only=False)
        start_epoch = dict_checkpoint["epoch"]
        global_step = dict_checkpoint["global_step"]
        backbone.load_state_dict(dict_checkpoint["state_dict_backbone"])
        module_fc.load_state_dict(dict_checkpoint["state_dict_softmax_fc"])
        opt.load_state_dict(dict_checkpoint["state_optimizer"])
        lr_scheduler.load_state_dict(dict_checkpoint["state_lr_scheduler"])
        if "state_step_lr_scheduler" in dict_checkpoint:
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

    callback_logging = CallBackLogging(
        frequent=cfg.frequent,
        total_step=cfg.total_step,
        batch_size=cfg.batch_size,
        start_step=global_step,
        writer=summary_writer,
        rank=0,
        world_size=1,
    )

    loss_am = AverageMeter()
    amp = torch.amp.GradScaler('cuda', growth_interval=100)

    for epoch in range(start_epoch, cfg.num_epoch):
        print(f'Epoch {epoch + 1} / {cfg.num_epoch}')
        # Initialize StepLR scheduler when switching from PolyScheduler
        use_step_lr = cfg.step_lr_after_epoch and epoch >= cfg.step_lr_after_epoch
        if use_step_lr and not step_lr_initialized:
            if cfg.step_lr_start == "initial":
                start_lr = cfg.lr
            elif cfg.step_lr_start == "current":
                start_lr = opt.param_groups[0]['lr']
            else:
                start_lr = float(cfg.step_lr_start)

            for param_group in opt.param_groups:
                param_group['lr'] = start_lr

            step_lr_scheduler = StepLR(
                optimizer=opt,
                step_size=cfg.step_size,
                gamma=cfg.step_gamma
            )
            step_lr_initialized = True
            logging.info(f"Switched to StepLR at epoch {epoch} with starting LR: {start_lr}")

        for _, (img, local_labels) in enumerate(train_loader):
            global_step += 1
            img = img.cuda(non_blocking=True)
            local_labels = local_labels.cuda(non_blocking=True)

            local_embeddings = backbone(img)
            loss: torch.Tensor = module_fc(local_embeddings, local_labels)
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

        # Step the StepLR scheduler at the end of each epoch (if active)
        if use_step_lr and step_lr_scheduler is not None:
            step_lr_scheduler.step()

        if cfg.save_all_states:
            checkpoint = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "state_dict_backbone": backbone.state_dict(),
                "state_dict_softmax_fc": module_fc.state_dict(),
                "state_optimizer": opt.state_dict(),
                "state_lr_scheduler": lr_scheduler.state_dict()
            }
            if step_lr_scheduler is not None:
                checkpoint["state_step_lr_scheduler"] = step_lr_scheduler.state_dict()
            torch.save(checkpoint, os.path.join(cfg.output, "checkpoint.pt"))

        if (epoch + 1) % 4 == 0:
            path_module = os.path.join(cfg.output, f"model_{epoch+1}.pt")
            torch.save(backbone.state_dict(), path_module)

        # Validation loss
        if val_loader is not None:
            backbone.eval()
            val_loss_am = AverageMeter()
            with torch.no_grad():
                for _, (img, local_labels) in enumerate(val_loader):
                    img = img.cuda(non_blocking=True)
                    local_labels = local_labels.cuda(non_blocking=True)
                    local_embeddings = backbone(img)
                    val_loss = module_fc(local_embeddings, local_labels)
                    val_loss_am.update(val_loss.item(), 1)
            backbone.train()
            summary_writer.add_scalar('val_loss', val_loss_am.avg, global_step)
            logging.info(f"Epoch {epoch+1} validation loss: {val_loss_am.avg:.4f}")

    path_module = os.path.join(cfg.output, "model_last.pt")
    checkpoint = {
        "state_dict_backbone": backbone.state_dict(),
        "state_dict_softmax_fc": module_fc.state_dict(),
    }
    torch.save(checkpoint, path_module)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(description="ArcFace Training in Pytorch")
    parser.add_argument("config", type=str, help="py config file")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--network", type=str, default=None)
    main(parser.parse_args())
