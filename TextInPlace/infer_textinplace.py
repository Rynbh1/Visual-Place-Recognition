#!/usr/bin/env python3
import sys
import os
import string
import torch
import pandas as pd
import numpy as np
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# Add local path for sub-modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
repo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "repo")
sys.path.append(repo_path)
sys.path.append(os.path.join(repo_path, "detectron2"))

from network import STVGLNet_test
from backbone import setup_cfg
from lib.gsv_cities import discover_csvs, resolve_image_path

# Vocabulary for decoding text predictions
voc = list(string.printable[:-6])

def rec_decode(rec):
    s = ''
    for c in rec:
        c = int(c)
        if c < len(voc):
            s += voc[c]
        elif c == len(voc):
            return s
        else:
            s += u''
    return s

def get_detected_text(predictions):
    if len(predictions) == 0:
        return []
    pred = predictions[0]  # batch size is 1
    if "instances" not in pred:
        return []
    instances = pred["instances"].to("cpu")
    if not hasattr(instances, "recs"):
        return []
    rec_strings = []
    for rec in instances.recs:
        rec_strings.append(rec_decode(rec))
    return rec_strings

def normalize_desc(desc):
    norm = np.linalg.norm(desc)
    if norm == 0:
        return desc
    return desc / norm

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_config = os.path.join(script_dir, "repo", "configs", "Bridge", "TotalText", "R_50_poly.yaml")
    default_weights = os.path.join(script_dir, "repo", "checkpoints", "best_model.pth")

    import argparse
    parser = argparse.ArgumentParser(description="TextInPlace Single-Image Inference Script")
    parser.add_argument("--query-image", type=str, required=True,
                        help="Path to the query photo to search for")
    parser.add_argument("--weights-path", type=str, default=default_weights,
                        help="Path to the model weights file")
    parser.add_argument("--dataset_root", type=str, nargs="+",
                        default=["/media/rayan/usb/VPR Dataset/paris/gsv_cities"],
                        help="Un ou plusieurs dossiers au format GSV-Cities "
                        "(Dataframes/<Ville>.csv + Images/<Ville>/) utilises pour "
                        "construire la base de reference (split test si present).")
    parser.add_argument("--config-file", type=str,
                        default=default_config,
                        help="Path to Detectron2/AdelaiDet config file")
    parser.add_argument("--confidence-threshold", type=float, default=0.3,
                        help="Minimum score for instance predictions")
    parser.add_argument("--features-dim", type=int, default=16384,
                        help="VPR features dimension")
    parser.add_argument("--opts", help="Modify config options", default=[], nargs=argparse.REMAINDER)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # Check query image exists
    if not os.path.exists(args.query_image):
        print(f"Error: Query image '{args.query_image}' not found.")
        return

    # Check config file exists
    if not os.path.exists(args.config_file):
        print(f"Error: Config file '{args.config_file}' not found.")
        return

    # Load weights first to detect features-dim
    weights_path = args.weights_path
    if not os.path.exists(weights_path):
        print(f"Error: Weights '{weights_path}' not found.")
        return

    print(f"Loading weights checkpoint from {weights_path}...")
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Auto-detect VPR features dimension
    detected_features_dim = args.features_dim
    for k, v in state_dict.items():
        if k.endswith("aggregation.fc.weight"):
            detected_row_dim = v.shape[0]
            detected_features_dim = detected_row_dim * 512
            print(f"Detected features_dim from weights checkpoint: {detected_features_dim}")
            break

    args.features_dim = detected_features_dim

    # Load config and model
    print(f"Setting up Detectron2 config from {args.config_file}...")
    cfg = setup_cfg(args)
    model = STVGLNet_test(cfg)
    
    # If the model contains DataParallel module prefix, remove it
    if list(state_dict.keys())[0].startswith('module'):
        from collections import OrderedDict
        state_dict = OrderedDict({k.replace('module.', ''): v for (k, v) in state_dict.items()})

    # Prefix text spotting keys if they are not already prefixed
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if not k.startswith("backbone.textmodel.") and (k.startswith("dptext_detr.") or k.startswith("recognizer.") or k.startswith("bridge.")):
            new_state_dict[f"backbone.textmodel.{k}"] = v
        else:
            new_state_dict[k] = v
    state_dict = new_state_dict

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        print(f"Strict loading failed: {e}. Retrying with strict=False")
        model.load_state_dict(state_dict, strict=False)

    model = model.to(device)
    model.eval()

    # Load database images from CSV
    dataset_specs = []
    for root in args.dataset_root:
        specs = discover_csvs(root, split="test")
        if not specs:
            print(f"Error: no CSV found under {root}/Dataframes")
            return
        dataset_specs.extend(specs)

    # Group by (source_idx, place_id) and take first image of each place as reference,
    # so place_ids from different sources never collide.
    db_rows = []
    for source_idx, (csv_path, img_dir) in enumerate(dataset_specs):
        df = pd.read_csv(csv_path, dtype={"panoid": str})
        for pid, group in df.groupby("place_id"):
            db_rows.append((source_idx, pid, img_dir, group.iloc[0]))
    print(f"Reference database built with {len(db_rows)} locations.")

    # Image transformations
    val_transform = T.Compose([
        T.Resize((320, 320), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Pre-extract database descriptors and texts
    db_places = []
    db_paths = []
    db_coords = []
    db_descriptors = []
    db_texts = []
    
    print("Extracting reference database descriptors and scene texts...")
    with torch.no_grad():
        for source_idx, pid, img_dir, row in tqdm(db_rows, total=len(db_rows)):
            path = resolve_image_path(img_dir, row)
            if path is None:
                continue

            img = Image.open(path).convert("RGB")
            tensor = val_transform(img).unsqueeze(0).to(device)

            # Forward pass to get text spotting predictions and features
            predictions, frozen_features = model(tensor)

            # VPR descriptor extraction
            desc = model.vpr_branch(frozen_features).cpu().numpy().flatten()
            desc = normalize_desc(desc)

            # Decode scene text
            detected_words = get_detected_text(predictions)

            db_places.append((source_idx, pid))
            db_paths.append(path)
            db_coords.append((row['lat'], row['lon']))
            db_descriptors.append(desc)
            db_texts.append(detected_words)

    db_matrix = np.array(db_descriptors)
    if len(db_matrix) == 0:
        print("Error: No database descriptors could be extracted.")
        return

    # Extract descriptor and text for user query image
    print(f"Extracting descriptor and scene text for query image: {args.query_image}")
    query_img = Image.open(args.query_image).convert("RGB")
    query_tensor = val_transform(query_img).unsqueeze(0).to(device)
    with torch.no_grad():
        q_predictions, q_frozen_features = model(query_tensor)
        query_desc = model.vpr_branch(q_frozen_features).cpu().numpy().flatten()
        query_desc = normalize_desc(query_desc)
        query_text = get_detected_text(q_predictions)

    print(f"Query Detected Text: {query_text}")

    # Calculate cosine similarity scores
    sims = np.dot(db_matrix, query_desc)
    top5_indices = np.argsort(-sims)[:5]

    print("\n" + "=" * 80)
    print("                 INFERENCE MATCH RESULTS (TOP 5)")
    print("=" * 80)
    print(f" Query Photo  : {args.query_image}")
    print(f" Detected Text: {', '.join(query_text) if query_text else 'None'}")
    print("-" * 80)
    
    for rank, idx in enumerate(top5_indices):
        matched_place = db_places[idx]
        matched_path = db_paths[idx]
        matched_lat, matched_lon = db_coords[idx]
        matched_words = db_texts[idx]
        sim = sims[idx]
        print(f" #{rank+1} - Place ID: {matched_place} | {os.path.basename(matched_path)}")
        print(f"     Sim Score   : {sim:.4f}")
        print(f"     Detected Text: {', '.join(matched_words) if matched_words else 'None'}")
        print(f"     Coordinates : Latitude = {matched_lat:.7f}, Longitude = {matched_lon:.7f}")
        print(f"     Google Maps : https://www.google.com/maps/search/?api=1&query={matched_lat},{matched_lon}")
        print("-" * 80)
    print("=" * 80)

    # Save a comparison result image with the top 5 candidates
    target_size = (250, 250)
    spacing = 15
    combined_img = Image.new("RGB", (6 * target_size[0] + 7 * spacing, target_size[1] + 130), "white")
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
        y_cursor = spacing + target_size[1] + 25
        for line in subtitle.split('\n'):
            draw.text((x_offset + 5, y_cursor), line, fill="gray", font=font)
            y_cursor += 15

    q_sub = f"Text: {', '.join(query_text)[:20]}" if query_text else "Text: None"
    add_panel(query_img, "QUERY PHOTO", spacing, (0, 102, 204), q_sub)
    
    for rank, idx in enumerate(top5_indices):
        matched_place = db_places[idx]
        matched_path = db_paths[idx]
        matched_lat, matched_lon = db_coords[idx]
        matched_words = db_texts[idx]
        sim = sims[idx]
        
        match_img = Image.open(matched_path).convert("RGB")
        border_color = (46, 204, 113) if rank == 0 else (189, 195, 199)
        x_offset = (rank + 1) * target_size[0] + (rank + 2) * spacing
        
        c_sub = f"Sim: {sim:.3f}\nPlace: {matched_place}\nText: {', '.join(matched_words)[:15]}\nLat: {matched_lat:.5f}, Lon: {matched_lon:.5f}"
        add_panel(match_img, f"CANDIDATE #{rank+1}", x_offset, border_color, c_sub)

    output_path = "textinplace_match.png"
    combined_img.save(output_path)
    print(f"Comparison image saved to: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    main()
