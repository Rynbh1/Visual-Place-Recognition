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

# Ajouter les sous-dossiers locaux au path de recherche de modules
sys.path.append("./MegaLoc")
sys.path.append("./MegaLoc/repo")
from lib.megaloc_model import MegaLoc
from lib.gsv_cities import discover_csvs, resolve_image_path

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


def main():
    import argparse
    parser = argparse.ArgumentParser(description="MegaLoc Testing Script")
    parser.add_argument("-w", "--weights_path", type=str, default="megaloc_finetuned_paris.pth",
                        help="Path to the model weights file")
    parser.add_argument("--dataset_root", type=str, nargs="+",
                        default=["/media/rayan/usb/VPR Dataset/paris/gsv_cities"],
                        help="Un ou plusieurs dossiers au format GSV-Cities "
                        "(Dataframes/<Ville>.csv + Images/<Ville>/) utilises pour "
                        "l'evaluation (split test si present).")
    parser.add_argument("--dist_threshold", type=float, default=100.0,
                        help="Rayon (en metres) autour de la position reelle dans lequel "
                        "une prediction est consideree correcte (defaut: 100m)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilisation de l'appareil : {device}")

    # 1. Charger le modèle et les poids fine-tunés
    weights_path = args.weights_path
    if not os.path.exists(weights_path):
        if os.path.exists(os.path.join("MegaLoc", weights_path)):
            weights_path = os.path.join("MegaLoc", weights_path)
        else:
            print(f"Erreur : {weights_path} est introuvable. Entraînez d'abord le modèle.")
            return

    print(f"Initialisation du modèle avec {weights_path}...")
    model = MegaLoc()
    model.load_state_dict(torch.load(weights_path, map_location=device))
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
                # La première image sert de référence (Database)
                db_rows.append((source_idx, pid, img_dir, group.iloc[0]))
                # Toutes les autres servent de requêtes (Queries)
                q_rows.extend(
                    (source_idx, pid, img_dir, group.iloc[i]) for i in range(1, len(group))
                )
            else:
                # Si le lieu n'a qu'une image, elle ne sert que de bruit dans la database
                db_rows.append((source_idx, pid, img_dir, group.iloc[0]))

    print(f"Nombre total d'images de test : {n_images}")
    print(f"-> Base de référence (Database) : {len(db_rows)} images")
    print(f"-> Requêtes (Queries) : {len(q_rows)} images")

    # Transformation d'inférence (322x322 pixels conforme au papier MegaLoc)
    val_transform = T.Compose([
        T.Resize((322, 322)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def extract_descriptors(rows, desc_name):
        places = []
        paths = []
        descriptors = []
        coords = []
        with torch.no_grad():
            for source_idx, pid, img_dir, row in tqdm(rows, total=len(rows), desc=f"Extraction {desc_name}"):
                path = resolve_image_path(img_dir, row)
                if path is None:
                    continue

                img = Image.open(path).convert("RGB")
                tensor = val_transform(img).unsqueeze(0).to(device)

                # Extraire le descripteur global de taille [8448]
                desc = model(tensor).cpu().numpy().flatten()

                places.append((source_idx, pid))
                paths.append(path)
                descriptors.append(desc)
                coords.append((row['lat'], row['lon']))

        return places, paths, np.array(descriptors), coords

    # Extraction des descripteurs
    db_places, db_paths, db_matrix, db_coords = extract_descriptors(db_rows, "Database")
    q_places, q_paths, q_matrix, q_coords = extract_descriptors(q_rows, "Queries")

    if len(db_matrix) == 0 or len(q_matrix) == 0:
        print("Erreur : Aucun descripteur n'a pu être extrait.")
        return

    # 4. Calcul de l'évaluation (Recall@N)
    n_queries = len(q_places)
    db_matrix = np.ascontiguousarray(db_matrix, dtype=np.float32)
    q_matrix = np.ascontiguousarray(q_matrix, dtype=np.float32)

    # Les metriques s'arretent a R@10 : inutile de trier toute la base par requete.
    # Une similarite par requete (np.dot(db_matrix, q_desc)) relit l'integralite des
    # descripteurs a chaque iteration -> plusieurs To de trafic memoire sur un gros
    # split, tous les coeurs satures, et un classement complet conserve pour chaque
    # requete (n_queries x n_db entiers). Un GEMM par bloc + argpartition top-K fait
    # le meme calcul, au resultat identique, en quelques secondes et 9 Mo de RAM.
    top_k = min(100, len(db_matrix))
    block = 512

    all_top_indices = np.empty((n_queries, top_k), dtype=np.int32)

    print("\nRecherche des plus proches voisins pour le Recall...")
    for start in tqdm(range(0, n_queries, block), desc="Recherche kNN"):
        stop = min(start + block, n_queries)
        sims = q_matrix[start:stop] @ db_matrix.T           # [b, n_db], cosine (L2-normalise)
        part = np.argpartition(-sims, top_k - 1, axis=1)[:, :top_k]
        rows = np.arange(stop - start)[:, None]
        order = np.argsort(-sims[rows, part], axis=1)
        all_top_indices[start:stop] = part[rows, order]

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
    print(f"  Recall@1                 : {correct_r1 / n_queries * 100:.2f}%")
    print(f"  Recall@5                 : {correct_r5 / n_queries * 100:.2f}%")
    print(f"  Recall@10                : {correct_r10 / n_queries * 100:.2f}%")
    print("=" * 50)

    # 5. Génération de l'image comparative pour une requête aléatoire
    print("\nGénération de l'image comparative (retrieval_result.png)...")
    q_idx = random.randint(0, n_queries - 1)
    q_place = q_places[q_idx]
    q_path = q_paths[q_idx]
    q_lat, q_lon = q_coords[q_idx]
    
    # Recalculee pour cette seule requete : conserver toutes les similarites couterait
    # n_queries x n_db flottants, alors qu'une ligne se recalcule en quelques ms.
    q_sims = db_matrix @ q_matrix[q_idx]
    q_top5_idx = all_top_indices[q_idx][:5]

    target_size = (250, 250)
    spacing = 15
    width = 6 * target_size[0] + 7 * spacing
    height = target_size[1] + 110
    
    combined_img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(combined_img)
    
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    # Helper pour coller un panneau d'image avec sa bordure colorée
    def add_panel(img_path, title, x_offset, border_color, subtitle=""):
        img = Image.open(img_path).convert("RGB").resize(target_size)
        
        bordered_img = Image.new("RGB", (target_size[0] + 6, target_size[1] + 6), border_color)
        bordered_img.paste(img, (3, 3))
        
        combined_img.paste(bordered_img, (x_offset, spacing))
        
        draw.text((x_offset + 5, spacing + target_size[1] + 8), title, fill="black", font=font)
        # Handle newlines in subtitle
        y_cursor = spacing + target_size[1] + 25
        for line in subtitle.split('\n'):
            draw.text((x_offset + 5, y_cursor), line, fill="gray", font=font)
            y_cursor += 15

    # 1. Dessiner la Requête (Bordure Bleue)
    add_panel(q_path, "QUERY", spacing, (0, 102, 204), f"Place: {q_place}\nLat: {q_lat:.5f}, Lon: {q_lon:.5f}")

    # 2. Dessiner les 5 meilleures correspondances
    for i, idx in enumerate(q_top5_idx):
        pred_place = db_places[idx]
        pred_path = db_paths[idx]
        sim = q_sims[idx]
        pred_lat, pred_lon = db_coords[idx]
        
        dist = haversine_distance(pred_lat, pred_lon, q_lat, q_lon)
        is_correct = (pred_place == q_place) or (dist <= args.dist_threshold)
        # Vert si correct, Rouge si incorrect
        border_color = (46, 204, 113) if is_correct else (231, 76, 60)
        
        title = f"TOP-{i+1} " + ("(CORRECT)" if is_correct else "(WRONG)")
        subtitle = f"Sim: {sim:.3f}\nPlace: {pred_place}\nDist: {dist:.1f}m"
        
        x_offset = (i + 1) * target_size[0] + (i + 2) * spacing
        add_panel(pred_path, title, x_offset, border_color, subtitle)

    output_path = "retrieval_result.png"
    combined_img.save(output_path)
    print(f"Image comparative sauvegardée sous : {os.path.abspath(output_path)}")
    print("=" * 50)

if __name__ == "__main__":
    main()
