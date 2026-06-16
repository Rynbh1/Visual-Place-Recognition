#!/usr/bin/env python3
import os
import sys
import argparse
import logging
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Subset

# Custom argument preprocessing: parse query_image and remove it from sys.argv
parser_query = argparse.ArgumentParser(add_help=False)
parser_query.add_argument("--query_image", type=str, required=True,
                          help="Path to the query photo to search for")
query_args, remaining_args = parser_query.parse_known_args()
# Resolve query image path to absolute path before changing directory
query_args.query_image = os.path.abspath(query_args.query_image)
sys.argv = [sys.argv[0]] + remaining_args

# Dynamically switch paths to repo subfolder so all modules load natively
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "repo"))
sys.path.append(".")

from utils import util, parser, commons, test
from network import STVGLNet
from datasets import paris_75018
from backbone import setup_cfg

def main():
    torch.backends.cudnn.benchmark = True

    # Parse remaining standard training/testing CLI args
    args = parser.parse_arguments()
    
    # Configure logging to console=info to keep it clean
    from datetime import datetime
    start_time = datetime.now()
    args.save_dir = os.path.join(
        "logs",
        args.save_dir,
        args.backbone + "_" + args.aggregation,
        "paris_75018_inference",
        start_time.strftime('%Y-%m-%d_%H-%M-%S')
    )
    commons.setup_logging(args.save_dir, console="info")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Check query image exists
    query_image_path = query_args.query_image
    if not os.path.exists(query_image_path):
        logging.error(f"Query image '{query_image_path}' not found.")
        return

    # Load dataset test split (reference database)
    logging.info(f"Loading reference database from datasets folder: {args.datasets_folder}")
    test_ds = paris_75018.Paris75018Dataset(args, split='test')
    
    # Extract only the database reference part of the dataset
    database_subset_ds = Subset(test_ds, list(range(test_ds.database_num)))
    database_dataloader = DataLoader(
        dataset=database_subset_ds,
        num_workers=args.num_workers,
        batch_size=args.infer_batch_size,
        pin_memory=True
    )
    logging.info(f"Reference database contains {test_ds.database_num} locations.")

    # Load model
    cfg = setup_cfg(args)
    model = STVGLNet(cfg)
    model = model.to(device)
    if args.resume:
        logging.info(f"Loading weights from {args.resume}")
        model = util.resume_model(args, model)
    model.eval()

    # Pre-extract database features
    db_features = np.empty((test_ds.database_num, args.features_dim), dtype="float32")
    
    logging.info("Extracting descriptors for the reference database...")
    with torch.no_grad():
        for inputs, indices in tqdm(database_dataloader, ncols=100):
            features = model(inputs.to(device))
            db_features[indices.numpy(), :] = features.cpu().numpy()

    # Extract features for query image
    logging.info(f"Extracting features for query image: {query_image_path}")
    query_img = Image.open(query_image_path).convert("RGB")
    query_tensor = test_ds.transform(query_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        query_feat = model(query_tensor).cpu().numpy().flatten()

    # Compute Euclidean (L2) distance to find best matches
    dists = np.sum((db_features - query_feat) ** 2, axis=1)
    top5_indices = np.argsort(dists)[:5]
    
    logging.info("\n" + "=" * 80)
    logging.info("                 INFERENCE MATCH RESULTS (TOP 5)")
    logging.info("=" * 80)
    logging.info(f" Query Photo  : {query_image_path}")
    logging.info("-" * 80)
    
    for rank, idx in enumerate(top5_indices):
        matched_path = test_ds.database_paths[idx]
        best_dist = dists[idx]
        
        # Parse coordinates and metadata from filename
        basename = os.path.basename(matched_path)
        parts = basename.replace(".jpg", "").split("_")
        try:
            matched_lat = float(parts[-3])
            matched_lon = float(parts[-2])
            matched_place = parts[-7]
        except Exception:
            matched_lat, matched_lon, matched_place = 0.0, 0.0, "unknown"
            
        logging.info(f" #{rank+1} - Place ID: {matched_place} | {basename}")
        logging.info(f"     L2 Distance : {best_dist:.4f}")
        logging.info(f"     Coordinates : Latitude = {matched_lat:.7f}, Longitude = {matched_lon:.7f}")
        logging.info(f"     Google Maps : https://www.google.com/maps/search/?api=1&query={matched_lat},{matched_lon}")
        logging.info("-" * 80)
    logging.info("=" * 80)

    # Save visualization comparative match image with the top 5 candidates
    target_size = (250, 250)
    spacing = 15
    combined_img = Image.new("RGB", (6 * target_size[0] + 7 * spacing, target_size[1] + 110), "white")
    draw = ImageDraw.Draw(combined_img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    def add_panel(img_obj, title, x_offset, border_color, subtitle=""):
        img = img_obj.resize(target_size)
        bordered = Image.new("RGB", (target_size[0] + 6, target_size[1] + 6), border_color)
        bordered.paste(img, (3, 3))
        combined_img.paste(bordered, (x_offset, spacing))
        draw.text((x_offset + 5, spacing + target_size[1] + 8), title, fill="black", font=font)
        # Handle newlines in subtitle
        y_cursor = spacing + target_size[1] + 25
        for line in subtitle.split('\n'):
            draw.text((x_offset + 5, y_cursor), line, fill="gray", font=font)
            y_cursor += 15

    add_panel(query_img, "QUERY PHOTO", spacing, (0, 102, 204))
    
    for rank, idx in enumerate(top5_indices):
        matched_path = test_ds.database_paths[idx]
        best_dist = dists[idx]
        
        # Parse coordinates and metadata from filename
        basename = os.path.basename(matched_path)
        parts = basename.replace(".jpg", "").split("_")
        try:
            matched_lat = float(parts[-3])
            matched_lon = float(parts[-2])
            matched_place = parts[-7]
        except Exception:
            matched_lat, matched_lon, matched_place = 0.0, 0.0, "unknown"
            
        match_img = Image.open(matched_path).convert("RGB")
        border_color = (46, 204, 113) if rank == 0 else (189, 195, 199)
        x_offset = (rank + 1) * target_size[0] + (rank + 2) * spacing
        add_panel(match_img, f"CANDIDATE #{rank+1}", x_offset, border_color,
                  f"Dist: {best_dist:.3f}\nPlace: {matched_place}\nLat: {matched_lat:.5f}, Lon: {matched_lon:.5f}")

    # Save output to TextInPlace/ folder
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "textinplace_match.png")
    combined_img.save(output_path)
    logging.info(f"Comparison image saved to: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    main()
