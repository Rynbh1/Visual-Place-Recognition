#!/usr/bin/env python3
import sys
import os
import random
import torch
import pandas as pd
import numpy as np
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# Add local path for sub-modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "repo"))
from lib.megaloc_model import MegaLoc

def main():
    import argparse
    parser = argparse.ArgumentParser(description="MegaLoc Single-Image Inference Script")
    parser.add_argument("--query_image", type=str, required=True,
                        help="Path to the query photo to search for")
    parser.add_argument("--weights_path", type=str, default="megaloc_finetuned_paris.pth",
                        help="Path to the model weights file")
    parser.add_argument("--db_csv", type=str,
                        default="/home/rayan/Documents/github/Visual Place Recognition/datasets/paris_75019/Dataframes/Paris75019_test.csv",
                        help="Path to the test/database CSV file to build reference database")
    parser.add_argument("--img_dir", type=str,
                        default="/home/rayan/Documents/github/Visual Place Recognition/datasets/paris_75019/Images",
                        help="Path to the database images directory")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # Check query image exists
    if not os.path.exists(args.query_image):
        print(f"Error: Query image '{args.query_image}' not found.")
        return

    # Load model
    weights_path = args.weights_path
    if not os.path.exists(weights_path):
        if os.path.exists(os.path.join("MegaLoc", weights_path)):
            weights_path = os.path.join("MegaLoc", weights_path)
        else:
            print(f"Error: Weights '{weights_path}' not found.")
            return

    print(f"Loading model with weights from {weights_path}...")
    model = MegaLoc()
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Load database images from CSV
    if not os.path.exists(args.db_csv):
        print(f"Error: Database CSV '{args.db_csv}' not found.")
        return

    df = pd.read_csv(args.db_csv)
    # Group by place_id and take first image of each place as reference
    db_rows = []
    for pid, group in df.groupby("place_id"):
        db_rows.append(group.iloc[0])
    db_df = pd.DataFrame(db_rows)
    print(f"Reference database built with {len(db_df)} locations.")

    # Image transformations
    val_transform = T.Compose([
        T.Resize((322, 322)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def get_filename(r):
        return f"{r['city_id']}_{r['place_id']:07d}_{r['year']:04d}_{r['month']:02d}_{r['northdeg']:03d}_{r['lat']:.7f}_{r['lon']:.7f}_{r['panoid']}.jpg"

    # Pre-extract database descriptors
    db_places = []
    db_paths = []
    db_coords = []
    db_descriptors = []
    
    print("Extracting reference database descriptors...")
    with torch.no_grad():
        for _, row in tqdm(db_df.iterrows(), total=len(db_df)):
            fname = get_filename(row)
            path = os.path.join(args.img_dir, row['city_id'], fname)
            if not os.path.exists(path):
                continue
            
            img = Image.open(path).convert("RGB")
            tensor = val_transform(img).unsqueeze(0).to(device)
            desc = model(tensor).cpu().numpy().flatten()
            
            db_places.append(row['place_id'])
            db_paths.append(path)
            db_coords.append((row['lat'], row['lon']))
            db_descriptors.append(desc)

    db_matrix = np.array(db_descriptors)
    if len(db_matrix) == 0:
        print("Error: No database descriptors could be extracted.")
        return

    # Extract descriptor for user query image
    print(f"Extracting descriptor for query image: {args.query_image}")
    query_img = Image.open(args.query_image).convert("RGB")
    query_tensor = val_transform(query_img).unsqueeze(0).to(device)
    with torch.no_grad():
        query_desc = model(query_tensor).cpu().numpy().flatten()

    # Calculate cosine similarity scores
    sims = np.dot(db_matrix, query_desc)
    top5_indices = np.argsort(-sims)[:5]

    print("\n" + "=" * 80)
    print("                 INFERENCE MATCH RESULTS (TOP 5)")
    print("=" * 80)
    print(f" Query Photo  : {args.query_image}")
    print("-" * 80)
    
    for rank, idx in enumerate(top5_indices):
        matched_place = db_places[idx]
        matched_path = db_paths[idx]
        matched_lat, matched_lon = db_coords[idx]
        sim = sims[idx]
        print(f" #{rank+1} - Place ID: {matched_place} | {os.path.basename(matched_path)}")
        print(f"     Sim Score   : {sim:.4f}")
        print(f"     Coordinates : Latitude = {matched_lat:.7f}, Longitude = {matched_lon:.7f}")
        print(f"     Google Maps : https://www.google.com/maps/search/?api=1&query={matched_lat},{matched_lon}")
        print("-" * 80)
    print("=" * 80)

    # Save a comparison result image with the top 5 candidates
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
        matched_place = db_places[idx]
        matched_path = db_paths[idx]
        matched_lat, matched_lon = db_coords[idx]
        sim = sims[idx]
        
        match_img = Image.open(matched_path).convert("RGB")
        border_color = (46, 204, 113) if rank == 0 else (189, 195, 199)
        x_offset = (rank + 1) * target_size[0] + (rank + 2) * spacing
        add_panel(match_img, f"CANDIDATE #{rank+1}", x_offset, border_color,
                  f"Sim: {sim:.3f}\nPlace: {matched_place}\nLat: {matched_lat:.5f}, Lon: {matched_lon:.5f}")

    output_path = "megaloc_match.png"
    combined_img.save(output_path)
    print(f"Comparison image saved to: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    main()
