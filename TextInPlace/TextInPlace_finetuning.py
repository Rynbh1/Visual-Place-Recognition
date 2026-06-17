#!/usr/bin/env python3
import os
import sys
import random
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
from datetime import datetime

# Add local path for sub-modules
repo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "repo")
sys.path.append(repo_path)
sys.path.append(os.path.join(repo_path, "detectron2"))

from network import STVGLNet
from backbone import setup_cfg
from utils import util, commons
from pytorch_metric_learning import losses, miners, distances

IMAGENET_MEAN_STD = {
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225]
}

class GSVCitiesFineTuneDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform

        # Group by place_id to get indices
        self.place_to_indices = (
            self.df.groupby("place_id").apply(lambda x: x.index.tolist()).to_dict()
        )
        self.places = list(self.place_to_indices.keys())

    def __len__(self):
        return len(self.places)

    def __getitem__(self, idx):
        place_id = self.places[idx]
        img_indices = self.place_to_indices[place_id]

        # Sample two positive images of the same place
        if len(img_indices) >= 2:
            idx_anchor, idx_positive = random.sample(img_indices, 2)
        else:
            idx_anchor = img_indices[0]
            idx_positive = img_indices[0]

        row_a = self.df.iloc[idx_anchor]
        row_p = self.df.iloc[idx_positive]

        def get_path(row):
            city = row['city_id']
            # Pattern 1: Paris75018 / Paris75019 format (with % 10**5 and 7-digit place_id)
            pl_id = int(row['place_id']) % 10**5
            pl_id_str = str(pl_id).zfill(7)
            
            value = row['panoid']
            if isinstance(value, float) and value.is_integer():
                panoid = str(int(value))
            else:
                panoid = str(value)
                
            year = str(row['year']).zfill(4)
            month = str(row['month']).zfill(2)
            northdeg = str(row['northdeg']).zfill(3)
            lat, lon = f"{row['lat']:.7f}", f"{row['lon']:.7f}"
            
            name1 = f"{city}_{pl_id_str}_{year}_{month}_{northdeg}_{lat}_{lon}_{panoid}.jpg"
            path1 = os.path.join(self.img_dir, city, name1)
            if os.path.exists(path1):
                return path1
                
            # Pattern 2: MegaLoc/Standard GSV format (direct 7-digit place_id)
            name2 = f"{city}_{int(row['place_id']):07d}_{year}_{month}_{northdeg}_{lat}_{lon}_{panoid}.jpg"
            path2 = os.path.join(self.img_dir, city, name2)
            if os.path.exists(path2):
                return path2
                
            return path1

        path_a = get_path(row_a)
        path_p = get_path(row_p)

        img_a = Image.open(path_a).convert("RGB")
        img_p = Image.open(path_p).convert("RGB")

        if self.transform:
            img_a = self.transform(img_a)
            img_p = self.transform(img_p)

        return img_a, img_p, place_id

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_config = os.path.join(script_dir, "repo", "configs", "Bridge", "TotalText", "R_50_poly.yaml")

    import argparse
    parser = argparse.ArgumentParser(description="TextInPlace Fine-tuning Script")
    parser.add_argument("--train_csv", type=str,
                        default="/home/rayan/Documents/github/Visual Place Recognition/datasets/paris_75019/Dataframes/Paris75019_train.csv",
                        help="Path to the training CSV file")
    parser.add_argument("--img_dir", type=str,
                        default="/home/rayan/Documents/github/Visual Place Recognition/datasets/paris_75019/Images",
                        help="Path to the images directory")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="Physical batch size")
    parser.add_argument("--grad_accum_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--save_weights", type=str, default="textinplace_finetuned_paris.pth",
                        help="Output filename for fine-tuned weights")
    parser.add_argument("--config-file", type=str, default=default_config,
                        help="Path to Detectron2/AdelaiDet config file")
    parser.add_argument("--confidence-threshold", type=float, default=0.3,
                        help="Minimum score for instance predictions")
    parser.add_argument("--features-dim", type=int, default=768,
                        help="VPR features dimension")
    parser.add_argument("--use-amp16", action="store_true", default=True,
                        help="Use Automatic Mixed Precision")
    parser.add_argument("--opts", help="Modify config options", default=[], nargs=argparse.REMAINDER)
    args = parser.parse_args()

    # Log setup
    start_time = datetime.now()
    save_dir = os.path.join("logs", "dinov2_vitb14_cosgem", "paris_75018_finetune", start_time.strftime('%Y-%m-%d_%H-%M-%S'))
    commons.setup_logging(save_dir, console="info")

    print("=" * 60)
    print("  STARTING OPTIMIZED TEXTINPLACE VPR FINE-TUNING PIPELINE  ")
    print("=" * 60)
    print(f"Physical batch size: {args.batch_size}")
    print(f"Gradient accumulation steps: {args.grad_accum_steps} (Effective batch size: {args.batch_size * args.grad_accum_steps})")
    print(f"AMP (Mixed Precision): {args.use_amp16}")
    print(f"Log directory: {save_dir}")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # Transformations
    train_transform = T.Compose([
        T.Resize((320, 320), interpolation=T.InterpolationMode.BILINEAR),
        T.RandAugment(num_ops=2, magnitude=9),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN_STD['mean'], std=IMAGENET_MEAN_STD['std']),
    ])

    if not os.path.exists(args.train_csv):
        print(f"Error: Training CSV not found at '{args.train_csv}'")
        return

    print(f"Loading datasets from {args.img_dir}")
    dataset = GSVCitiesFineTuneDataset(args.train_csv, args.img_dir, transform=train_transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True, pin_memory=True)

    print(f"Setting up Detectron2 config from {args.config_file}...")
    cfg = setup_cfg(args)
    model = STVGLNet(cfg).to(device)

    # Print trainable parameters info
    util.print_trainable_parameters(model)
    util.print_trainable_layers(model)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-3,
    )

    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.2, total_iters=args.epochs * len(dataloader))
    criterion = losses.MultiSimilarityLoss(alpha=1.0, beta=50, base=0.0, distance=distances.CosineSimilarity())
    miner = miners.MultiSimilarityMiner(epsilon=0.1, distance=distances.CosineSimilarity())

    scaler = torch.cuda.amp.GradScaler() if args.use_amp16 and device.type == 'cuda' else None

    model.train()
    epochs = args.epochs
    for epoch in range(epochs):
        epoch_loss = 0.0
        loop = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")

        optimizer.zero_grad()
        for batch_idx, (imgs_a, imgs_p, place_ids) in enumerate(loop):
            # Concatenate anchor and positive for faster batch operations
            imgs = torch.cat([imgs_a, imgs_p], dim=0).to(device)
            labels = torch.cat([place_ids, place_ids], dim=0).to(device)

            if scaler is not None:
                with torch.cuda.amp.autocast():
                    features = model(imgs)
                    miner_outputs = miner(features, labels)
                    loss = criterion(features, labels, miner_outputs)
                # Scale loss for gradient accumulation
                scaled_loss = loss / args.grad_accum_steps
                scaler.scale(scaled_loss).backward()
            else:
                features = model(imgs)
                miner_outputs = miner(features, labels)
                loss = criterion(features, labels, miner_outputs)
                scaled_loss = loss / args.grad_accum_steps
                scaled_loss.backward()

            if (batch_idx + 1) % args.grad_accum_steps == 0:
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        print(f"Average Epoch {epoch + 1} Loss: {epoch_loss / len(dataloader):.4f}")

    # Save weights
    torch.save(model.state_dict(), args.save_weights)
    print(f"Fine-tuned model weights saved successfully to '{args.save_weights}'")

if __name__ == "__main__":
    main()
