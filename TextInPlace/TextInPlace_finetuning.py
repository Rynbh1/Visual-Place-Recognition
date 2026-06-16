#!/usr/bin/env python3
import os
import sys
import logging
import random
import numpy as np
from datetime import datetime
from tqdm import tqdm

import torch
from torch.utils.data.dataloader import DataLoader
from pytorch_metric_learning import losses, miners, distances

# Switch to the repo subfolder dynamically so imports and paths resolve natively
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "repo"))
sys.path.append(".")

from utils import util, parser, commons, test
from network import STVGLNet
from datasets import paris_75018
from backbone import setup_cfg

def main():
    torch.backends.cudnn.benchmark = True

    # Parse arguments from CLI
    args = parser.parse_arguments()
    
    # 8GB VRAM / 16GB RAM laptop-friendly optimizations:
    # 1. Physical batch size defaults to 8 to avoid OOM
    if getattr(args, "train_batch_size", 32) > 8:
        args.train_batch_size = 8
    # 2. Mixed precision enabled to save memory and speed up
    args.use_amp16 = True
    # 3. CPU threads minimized to save RAM
    args.num_workers = 2
    args.epochs_num = getattr(args, "epochs_num", 5)
    
    # Gradient accumulation steps (simulate larger batch size)
    accumulation_steps = getattr(args, "accumulation_steps", 4)
    
    # Save directory setup
    start_time = datetime.now()
    args.save_dir = os.path.join(
        "logs", 
        args.save_dir, 
        args.backbone + "_" + args.aggregation, 
        "paris_75018_finetune", 
        start_time.strftime('%Y-%m-%d_%H-%M-%S')
    )
    commons.setup_logging(args.save_dir, console="info")
    commons.make_deterministic(args.seed)
    
    logging.info("============================================================")
    logging.info("  STARTING OPTIMIZED TEXTINPLACE VPR FINE-TUNING PIPELINE   ")
    logging.info("============================================================")
    logging.info(f"Physical physical batch size: {args.train_batch_size}")
    logging.info(f"Gradient accumulation steps: {accumulation_steps} (Effective batch size: {args.train_batch_size * accumulation_steps})")
    logging.info(f"AMP (Mixed Precision): {args.use_amp16}")
    logging.info(f"Log directory: {args.save_dir}")
    logging.info("============================================================")

    # Datasets
    logging.info(f"Loading datasets from {args.datasets_folder}")
    train_ds = paris_75018.Paris75018Dataset(
        args, 
        split='train', 
        img_per_place=4, 
        min_img_per_place=4
    )
    val_ds = paris_75018.Paris75018Dataset(
        args,
        split='test'
    )
    logging.info(f"Train set: {train_ds}")
    logging.info(f"Val set: {val_ds}")

    # Model
    cfg = setup_cfg(args)
    model = STVGLNet(cfg)
    model = model.to("cuda")

    # Resume checkpoint if provided
    if args.resume:
        logging.info(f"Resuming model from {args.resume}")
        model = util.resume_model(args, model)
        
    best_r1 = 0.0
    start_epoch_num = 0
    not_improved_num = 0

    # Freeze textmodel and stage 0, unfreeze ResNet layers and aggregator
    for param in model.parameters():
        param.requires_grad = False
        
    # Unfreeze backbone's unfrozen_layers (resnet stage 2 & 3)
    for param in model.backbone.unfrozen_layers.parameters():
        param.requires_grad = True
        
    # Unfreeze BoQ aggregator
    for param in model.aggregation.parameters():
        param.requires_grad = True

    util.print_trainable_parameters(model)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # Optimizer (on active params only)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    
    # Scheduler
    if not args.resume:
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.2, total_iters=4000)
        
    # Loss & Miner
    criterion = losses.MultiSimilarityLoss(alpha=1.0, beta=50, base=0.0, distance=distances.CosineSimilarity())
    miner = miners.MultiSimilarityMiner(epsilon=0.1, distance=distances.CosineSimilarity())
    
    if args.use_amp16:
        scaler = torch.cuda.amp.GradScaler()

    # Training loop
    for epoch_num in range(start_epoch_num, args.epochs_num):
        epoch_num += 1
        logging.info(f"Start training epoch: {epoch_num:02d}")

        train_dl = DataLoader(
            train_ds, 
            batch_size=args.train_batch_size, 
            shuffle=True, 
            num_workers=args.num_workers, 
            pin_memory=True
        )
        epoch_start_time = datetime.now()
        epoch_losses = []

        model.train()
        optimizer.zero_grad()
        
        for batch_idx, (places, labels) in enumerate(tqdm(train_dl, ncols=100, desc=f"Epoch {epoch_num}/{args.epochs_num}")):
            BS, N, ch, h, w = places.shape
            images = places.view(BS*N, ch, h, w).to("cuda")
            labels = labels.view(-1).to("cuda")

            if not args.use_amp16:
                features = model(images)
                miner_outputs = miner(features, labels)
                loss = criterion(features, labels, miner_outputs)
                # Scale loss for gradient accumulation
                loss = loss / accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_dl):
                    optimizer.step()
                    optimizer.zero_grad()
            else:
                with torch.cuda.amp.autocast():
                    features = model(images)
                    miner_outputs = miner(features, labels)
                    loss = criterion(features, labels, miner_outputs)
                    # Scale loss for gradient accumulation
                    loss = loss / accumulation_steps
                
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_dl):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

            if not args.resume:
                scheduler.step()

            epoch_losses = np.append(epoch_losses, loss.item() * accumulation_steps)

            del loss, features, miner_outputs, images, labels
            
        logging.info(f"Finished epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, "
                     f"average epoch loss = {epoch_losses.mean():.4f}")

        # Compute recalls on validation set
        recalls, recalls_str = test.test(args, val_ds, model)
        logging.info(f"Recalls on val set {val_ds}: {recalls_str}")

        is_best = recalls[0] > best_r1
        if is_best:
            logging.info(f"Improved: previous best R@1 = {best_r1:.1f}, current R@1 = {recalls[0]:.1f}")
            best_r1 = recalls[0]
            not_improved_num = 0
        else:
            not_improved_num += 1
            logging.info(f"Not improved: {not_improved_num} / {args.patience}: best R@1 = {best_r1:.1f}, current R@1 = {recalls[0]:.1f}")

        # Save checkpoint
        util.save_checkpoint(args, {
            "epoch_num": epoch_num,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "recalls": recalls,
            "best_r1": best_r1,
            "not_improved_num": not_improved_num
        }, is_best, filename="last_model.pth")

        # Explicit cleanup to keep laptop memory stable
        torch.cuda.empty_cache()

        if not_improved_num == args.patience:
            logging.info(f"Performance did not improve for {not_improved_num} epochs. Stop training.")
            break

    logging.info(f"Best R@1: {best_r1:.1f}")
    logging.info(f"Trained for {epoch_num:02d} epochs, in total in {str(datetime.now() - start_time)[:-7]}")

if __name__ == "__main__":
    main()
