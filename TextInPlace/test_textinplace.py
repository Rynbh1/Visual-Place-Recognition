#!/usr/bin/env python3
import sys
import os
import random
import string
import torch
import pandas as pd
import numpy as np
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

try:
    import faiss
except ImportError:
    faiss = None

# Add local path for sub-modules
repo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "repo")
sys.path.append(repo_path)
sys.path.append(os.path.join(repo_path, "detectron2"))

from network import STVGLNet_test
from backbone import setup_cfg
from utils import util

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

def get_image_path(row, img_dir):
    city = row['city_id']
    place_id = int(row['place_id'])
    
    # Extract year, month, northdeg, lat, lon, panoid
    year = str(row['year']).zfill(4)
    month = str(row['month']).zfill(2)
    northdeg = str(row['northdeg']).zfill(3)
    lat, lon = f"{row['lat']:.7f}", f"{row['lon']:.7f}"
    
    value = row['panoid']
    if isinstance(value, float) and value.is_integer():
        panoid = str(int(value))
    else:
        panoid = str(value)

    # Pattern 1: Paris75018 / Paris75019 dataset format
    pl_id = place_id % 10**5
    pl_id_str = str(pl_id).zfill(7)
    name1 = f"{city}_{pl_id_str}_{year}_{month}_{northdeg}_{lat}_{lon}_{panoid}.jpg"
    path1 = os.path.join(img_dir, city, name1)
    if os.path.exists(path1):
        return path1

    # Pattern 2: MegaLoc/Standard GSV format (direct 7-digit place_id)
    name2 = f"{city}_{place_id:07d}_{year}_{month}_{northdeg}_{lat}_{lon}_{panoid}.jpg"
    path2 = os.path.join(img_dir, city, name2)
    if os.path.exists(path2):
        return path2

    return None

def normalize_desc(desc):
    norm = np.linalg.norm(desc)
    if norm == 0:
        return desc
    return desc / norm

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000.0  # meters
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def scene_text_rerank(pred_indices, q_words, db_words_list):
    if len(q_words) == 0:
        return pred_indices
    
    # Filter query words to keep only those with at least one digit
    q_words_filtered = [s for s in q_words if any(char.isdigit() for char in s)]
    if len(q_words_filtered) == 0:
        return pred_indices
    
    preds_with_scores = []
    for ref_index in pred_indices:
        r_words = db_words_list[ref_index]
        r_words_filtered = [s for s in r_words if any(char.isdigit() for char in s)]
        
        if len(r_words_filtered) != 0:
            common_strings = set(q_words_filtered) & set(r_words_filtered)
            numerator = sum(len(s) for s in common_strings) if common_strings else 0
            denominator = sum(len(s) for s in q_words_filtered)
            score = numerator / denominator if denominator != 0 else 0
        else:
            score = 0
            
        preds_with_scores.append((ref_index, score))
        
    preds_with_scores.sort(key=lambda a: a[1], reverse=True)
    r_predictions, _ = zip(*preds_with_scores)
    return np.array(r_predictions)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_config = os.path.join(script_dir, "repo", "configs", "Bridge", "TotalText", "R_50_poly.yaml")

    import argparse
    parser = argparse.ArgumentParser(description="TextInPlace Testing and Evaluation Script")
    parser.add_argument("--weights_path", type=str, default="textinplace_finetuned_paris.pth",
                        help="Path to the model weights file")
    parser.add_argument("--test_csv", type=str, 
                        default="/home/rayan/Documents/github/Visual Place Recognition/datasets/paris_75019/Dataframes/Paris75019_test.csv",
                        help="Path to the test CSV file")
    parser.add_argument("--img_dir", type=str, 
                        default="/home/rayan/Documents/github/Visual Place Recognition/datasets/paris_75019/Images",
                        help="Path to the images directory")
    parser.add_argument("--config-file", type=str, default=default_config,
                        help="Path to Detectron2/AdelaiDet config file")
    parser.add_argument("--confidence-threshold", type=float, default=0.3,
                        help="Minimum score for instance predictions")
    parser.add_argument("--features-dim", type=int, default=16384,
                        help="VPR features dimension")
    parser.add_argument("--use-text-rerank", action="store_true",
                        help="Enable scene text based re-ranking during evaluation")
    parser.add_argument("--opts", help="Modify config options", default=[], nargs=argparse.REMAINDER)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilisation de l'appareil : {device}")

    # 1. Charger le modèle et les poids fine-tunés
    weights_path = args.weights_path
    if not os.path.exists(weights_path):
        if os.path.exists(os.path.join("TextInPlace", weights_path)):
            weights_path = os.path.join("TextInPlace", weights_path)
        else:
            print(f"Erreur : {weights_path} est introuvable. Entraînez d'abord le modèle.")
            return

    # Check config file exists
    if not os.path.exists(args.config_file):
        print(f"Error: Config file '{args.config_file}' not found.")
        return

    print(f"Chargement des poids depuis {weights_path}...")
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

    print(f"Initialisation du modèle avec la config de {args.config_file}...")
    cfg = setup_cfg(args)
    model = STVGLNet_test(cfg)
    
    if list(state_dict.keys())[0].startswith('module'):
        from collections import OrderedDict
        state_dict = OrderedDict({k.replace('module.', ''): v for (k, v) in state_dict.items()})

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

    # 2. Charger les données de test
    test_csv = args.test_csv
    img_dir = args.img_dir

    if not os.path.exists(test_csv):
        print(f"Erreur : {test_csv} introuvable.")
        return

    df = pd.read_csv(test_csv)
    print(f"Nombre total d'images de test : {len(df)}")

    # 3. Séparation en Database et Queries (par place_id)
    db_rows = []
    q_rows = []
    for pid, group in df.groupby("place_id"):
        if len(group) >= 2:
            db_rows.append(group.iloc[0])
            q_rows.extend([group.iloc[i] for i in range(1, len(group))])
        else:
            db_rows.append(group.iloc[0])

    db_df = pd.DataFrame(db_rows)
    q_df = pd.DataFrame(q_rows)

    print(f"-> Base de référence (Database) : {len(db_df)} images")
    print(f"-> Requêtes (Queries) : {len(q_df)} images")

    # Image transformations
    val_transform = T.Compose([
        T.Resize((320, 320), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def extract_descriptors(dataframe, desc_name):
        places = []
        paths = []
        descriptors = []
        texts = []
        coords = []
        with torch.no_grad():
            for _, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc=f"Extraction {desc_name}"):
                path = get_image_path(row, img_dir)
                if path is None or not os.path.exists(path):
                    continue
                
                img = Image.open(path).convert("RGB")
                tensor = val_transform(img).unsqueeze(0).to(device)
                
                predictions, frozen_features = model(tensor)
                desc = model.vpr_branch(frozen_features).cpu().numpy().flatten()
                desc = normalize_desc(desc)

                detected_words = get_detected_text(predictions)
                
                places.append(row['place_id'])
                paths.append(path)
                descriptors.append(desc)
                texts.append(detected_words)
                coords.append((row['lat'], row['lon']))
                
        return np.array(places), paths, np.array(descriptors), texts, coords

    # Extraction des descripteurs
    db_places, db_paths, db_matrix, db_texts, db_coords = extract_descriptors(db_df, "Database")
    q_places, q_paths, q_matrix, q_texts, q_coords = extract_descriptors(q_df, "Queries")

    if len(db_matrix) == 0 or len(q_matrix) == 0:
        print("Erreur : Aucun descripteur n'a pu être extrait.")
        return

    # 4. Calcul de l'évaluation (Recall@N)
    correct_r1 = 0
    correct_r5 = 0
    correct_r10 = 0
    n_queries = len(q_places)

    print("\nRecherche des plus proches voisins pour le Recall...")
    all_top_indices = []
    all_sims = []

    for i in range(n_queries):
        q_place = q_places[i]
        q_desc = q_matrix[i]
        q_words = q_texts[i]

        # Cosine similarity
        sims = np.dot(db_matrix, q_desc)
        
        # Initial ranking by visual descriptor
        top_indices = np.argsort(-sims)
        
        # Apply scene text reranking if requested
        if args.use_text_rerank:
            # Rerank the top 100 visual predictions using scene text
            top_100_indices = top_indices[:100]
            reranked_top_100 = scene_text_rerank(top_100_indices, q_words, db_texts)
            top_indices = np.concatenate([reranked_top_100, top_indices[100:]])

        top_places = db_places[top_indices]

        all_top_indices.append(top_indices)
        all_sims.append(sims)

        # Recall metrics using exact place ID OR distance <= 25m
        q_lat, q_lon = q_coords[i]
        
        is_correct_top = []
        for idx in top_indices[:10]:
            pred_place = db_places[idx]
            pred_lat, pred_lon = db_coords[idx]
            dist = haversine_distance(pred_lat, pred_lon, q_lat, q_lon)
            is_correct_top.append((pred_place == q_place) or (dist <= 25.0))
            
        if is_correct_top[0]:
            correct_r1 += 1
        if any(is_correct_top[:5]):
            correct_r5 += 1
        if any(is_correct_top[:10]):
            correct_r10 += 1

    print("\n" + "=" * 50)
    print("                RÉSULTATS DE TEST")
    print("=" * 50)
    print(f"  Nombre total de requêtes : {n_queries}")
    print(f"  Reranking Textuel actif  : {args.use_text_rerank}")
    print(f"  Recall@1                 : {correct_r1 / n_queries * 100:.2f}%")
    print(f"  Recall@5                 : {correct_r5 / n_queries * 100:.2f}%")
    print(f"  Recall@10                : {correct_r10 / n_queries * 100:.2f}%")
    print("=" * 50)

    # 5. Génération de l'image comparative pour une requête aléatoire
    print("\nGénération de l'image comparative (retrieval_result.png)...")
    q_idx = random.randint(0, n_queries - 1)
    q_place = q_places[q_idx]
    q_path = q_paths[q_idx]
    q_words = q_texts[q_idx]
    q_lat, q_lon = q_coords[q_idx]
    
    q_sims = all_sims[q_idx]
    q_top5_idx = all_top_indices[q_idx][:5]

    target_size = (250, 250)
    spacing = 15
    width = 6 * target_size[0] + 7 * spacing
    height = target_size[1] + 130
    
    combined_img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(combined_img)
    
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    def add_panel(img_path, title, x_offset, border_color, subtitle=""):
        img = Image.open(img_path).convert("RGB").resize(target_size)
        bordered_img = Image.new("RGB", (target_size[0] + 6, target_size[1] + 6), border_color)
        bordered_img.paste(img, (3, 3))
        combined_img.paste(bordered_img, (x_offset, spacing))
        draw.text((x_offset + 5, spacing + target_size[1] + 8), title, fill="black", font=font)
        y_cursor = spacing + target_size[1] + 25
        for line in subtitle.split('\n'):
            draw.text((x_offset + 5, y_cursor), line, fill="gray", font=font)
            y_cursor += 15

    print("=" * 60)
    print("DEBUG VISUAL PANEL DRAWING:")
    print(f"Random Query Index: {q_idx}")
    print(f"Query Place ID: {q_place} (Type: {type(q_place)})")
    print(f"Query Image Path: {q_path}")
    print(f"Query Coords: {q_lat:.7f}, {q_lon:.7f}")
    print("-" * 60)
    for i, idx in enumerate(q_top5_idx):
        pred_place_debug = db_places[idx]
        pred_path_debug = db_paths[idx]
        pred_lat_debug, pred_lon_debug = db_coords[idx]
        dist_debug = haversine_distance(pred_lat_debug, pred_lon_debug, q_lat, q_lon)
        is_correct_debug = (pred_place_debug == q_place) or (dist_debug <= 25.0)
        print(f"Match #{i+1}: Index={idx}, Place ID={pred_place_debug}, Dist={dist_debug:.1f}m, Correct={is_correct_debug}")
        print(f"         Path: {pred_path_debug}")
    print("=" * 60)

    # 1. Draw Query (Blue Border)
    q_text_snippet = ', '.join(q_words)[:18] if q_words else 'None'
    add_panel(q_path, "QUERY", spacing, (0, 102, 204), f"Place: {q_place}\nText: {q_text_snippet}")

    # 2. Draw Top-5 matches
    for i, idx in enumerate(q_top5_idx):
        pred_place = db_places[idx]
        pred_path = db_paths[idx]
        sim = q_sims[idx]
        pred_words = db_texts[idx]
        pred_lat, pred_lon = db_coords[idx]
        
        dist = haversine_distance(pred_lat, pred_lon, q_lat, q_lon)
        is_correct = (pred_place == q_place) or (dist <= 25.0)
        # Green if correct, Red if incorrect
        border_color = (46, 204, 113) if is_correct else (231, 76, 60)
        
        title = f"TOP-{i+1} " + ("(CORRECT)" if is_correct else "(WRONG)")
        pred_text_snippet = ', '.join(pred_words)[:15] if pred_words else 'None'
        subtitle = f"Sim: {sim:.3f}\nPlace: {pred_place}\nText: {pred_text_snippet}\nDist: {dist:.1f}m"
        
        x_offset = (i + 1) * target_size[0] + (i + 2) * spacing
        add_panel(pred_path, title, x_offset, border_color, subtitle)

    output_path = "retrieval_result.png"
    combined_img.save(output_path)
    print(f"Image comparative sauvegardée sous : {os.path.abspath(output_path)}")
    print("=" * 50)

if __name__ == "__main__":
    main()
