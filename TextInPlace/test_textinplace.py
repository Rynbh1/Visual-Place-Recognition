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
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
repo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "repo")
sys.path.append(repo_path)
sys.path.append(os.path.join(repo_path, "detectron2"))

from network import STVGLNet_test
from backbone import setup_cfg
from utils import util
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
    parser.add_argument("-w", "--weights_path", type=str, default="textinplace_finetuned.pth",
                        help="Path to the model weights file")
    parser.add_argument("--dataset_root", type=str, nargs="+",
                        default=["/media/rayan/usb/VPR Dataset/paris/gsv_cities"],
                        help="Un ou plusieurs dossiers au format GSV-Cities "
                        "(Dataframes/<Ville>.csv + Images/<Ville>/) utilises pour "
                        "l'evaluation (split test si present).")
    parser.add_argument("--config-file", type=str, default=default_config,
                        help="Path to Detectron2/AdelaiDet config file")
    parser.add_argument("--confidence-threshold", type=float, default=0.3,
                        help="Minimum score for instance predictions")
    parser.add_argument("--features-dim", type=int, default=16384,
                        help="VPR features dimension")
    parser.add_argument("--use-text-rerank", action="store_true",
                        help="Enable scene text based re-ranking during evaluation")
    parser.add_argument("--dist_threshold", type=float, default=100.0,
                        help="Rayon (en metres) autour de la position reelle dans lequel "
                        "une prediction est consideree correcte (defaut: 100m)")
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

    # 2. Decouvrir les CSV de test dans un ou plusieurs dossiers GSV-Cities
    dataset_specs = []
    for root in args.dataset_root:
        specs = discover_csvs(root, split="test")
        if not specs:
            print(f"Erreur : aucun CSV trouve sous {root}/Dataframes")
            return
        dataset_specs.extend(specs)

    # 3. Séparation en Database et Queries (par (source, place_id), pour ne
    #    jamais mélanger des place_id qui collisionnent entre deux sources)
    db_rows = []
    q_rows = []
    n_images = 0
    for source_idx, (csv_path, img_dir) in enumerate(dataset_specs):
        df = pd.read_csv(csv_path, dtype={"panoid": str})
        n_images += len(df)
        for pid, group in df.groupby("place_id"):
            if len(group) >= 2:
                db_rows.append((source_idx, pid, img_dir, group.iloc[0]))
                q_rows.extend(
                    (source_idx, pid, img_dir, group.iloc[i]) for i in range(1, len(group))
                )
            else:
                db_rows.append((source_idx, pid, img_dir, group.iloc[0]))

    print(f"Nombre total d'images de test : {n_images}")
    print(f"-> Base de référence (Database) : {len(db_rows)} images")
    print(f"-> Requêtes (Queries) : {len(q_rows)} images")

    # Image transformations
    val_transform = T.Compose([
        T.Resize((320, 320), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def extract_descriptors(rows, desc_name):
        places = []
        paths = []
        descriptors = []
        texts = []
        coords = []
        with torch.no_grad():
            for source_idx, pid, img_dir, row in tqdm(rows, total=len(rows), desc=f"Extraction {desc_name}"):
                path = resolve_image_path(img_dir, row)
                if path is None:
                    continue

                img = Image.open(path).convert("RGB")
                tensor = val_transform(img).unsqueeze(0).to(device)

                predictions, frozen_features = model(tensor)
                desc = model.vpr_branch(frozen_features).cpu().numpy().flatten()
                desc = normalize_desc(desc)

                detected_words = get_detected_text(predictions)

                places.append((source_idx, pid))
                paths.append(path)
                descriptors.append(desc)
                texts.append(detected_words)
                coords.append((row['lat'], row['lon']))

        return places, paths, np.array(descriptors), texts, coords

    # Extraction des descripteurs
    db_places, db_paths, db_matrix, db_texts, db_coords = extract_descriptors(db_rows, "Database")
    q_places, q_paths, q_matrix, q_texts, q_coords = extract_descriptors(q_rows, "Queries")

    if len(db_matrix) == 0 or len(q_matrix) == 0:
        print("Erreur : Aucun descripteur n'a pu être extrait.")
        return

    # 4. Calcul de l'évaluation (Recall@N)
    n_queries = len(q_places)
    db_matrix = np.ascontiguousarray(db_matrix, dtype=np.float32)
    q_matrix = np.ascontiguousarray(q_matrix, dtype=np.float32)

    # Seuls les 100 premiers candidats peuvent changer R@1/5/10 : le reranking textuel
    # ne reordonne que le top-100 et les metriques s'arretent a 10. On ne garde donc que
    # ce top-K au lieu du classement complet, et on calcule les similarites par blocs de
    # requetes. Une similarite par requete (np.dot(db_matrix, q_desc)) relit toute la
    # base a chaque iteration : sur 22 907 requetes x 3 926 references, ca fait plusieurs
    # To de trafic memoire, tous les coeurs satures pendant des dizaines de minutes, et
    # ~1,4 Go de RAM par tableau conserve. Un GEMM par bloc la relit une fois par bloc.
    top_k = min(100, len(db_matrix))
    block = 512

    all_top_indices = np.empty((n_queries, top_k), dtype=np.int32)

    print("\nRecherche des plus proches voisins pour le Recall...")
    for start in tqdm(range(0, n_queries, block), desc="Recherche kNN"):
        stop = min(start + block, n_queries)
        sims = q_matrix[start:stop] @ db_matrix.T           # [b, n_db], cosine (L2-normalise)
        # argpartition = selection O(n) du top-K, puis tri des K seuls
        part = np.argpartition(-sims, top_k - 1, axis=1)[:, :top_k]
        rows = np.arange(stop - start)[:, None]
        order = np.argsort(-sims[rows, part], axis=1)
        all_top_indices[start:stop] = part[rows, order]

    if args.use_text_rerank:
        for i in tqdm(range(n_queries), desc="Reranking textuel"):
            all_top_indices[i] = scene_text_rerank(all_top_indices[i], q_texts[i], db_texts)

    # Recall metrics using exact place ID OR distance <= dist_threshold
    db_lat = np.array([c[0] for c in db_coords], dtype=np.float64)
    db_lon = np.array([c[1] for c in db_coords], dtype=np.float64)
    # db_places contient des tuples (source_idx, place_id) -> cle texte comparable en vectoriel
    db_place_keys = np.array([f"{s}:{p}" for s, p in db_places])
    q_place_keys = [f"{s}:{p}" for s, p in q_places]

    correct_r1 = 0
    correct_r5 = 0
    correct_r10 = 0
    for i in range(n_queries):
        top10 = all_top_indices[i, :10]
        q_lat, q_lon = q_coords[i]
        dist = haversine_distance(db_lat[top10], db_lon[top10], q_lat, q_lon)
        is_correct_top = (db_place_keys[top10] == q_place_keys[i]) | (dist <= args.dist_threshold)

        if is_correct_top[0]:
            correct_r1 += 1
        if is_correct_top[:5].any():
            correct_r5 += 1
        if is_correct_top[:10].any():
            correct_r10 += 1

    print("\n" + "=" * 50)
    print("                RÉSULTATS DE TEST")
    print("=" * 50)
    print(f"  Nombre total de requêtes : {n_queries}")
    print(f"  Rayon de validation      : {args.dist_threshold:.0f}m")
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
    
    # Similarites recalculees pour cette seule requete : les conserver pour les 22 907
    # requetes couterait ~1,4 Go, alors qu'une ligne se recalcule en quelques ms.
    q_sims = db_matrix @ q_matrix[q_idx]
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
        is_correct_debug = (pred_place_debug == q_place) or (dist_debug <= args.dist_threshold)
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
        is_correct = (pred_place == q_place) or (dist <= args.dist_threshold)
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
