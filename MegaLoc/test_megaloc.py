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
from repo.megaloc_model import MegaLoc

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilisation de l'appareil : {device}")

    # 1. Charger le modèle et les poids fine-tunés
    weights_path = "megaloc_finetuned_paris.pth"
    if not os.path.exists(weights_path):
        if os.path.exists("MegaLoc/megaloc_finetuned_paris.pth"):
            weights_path = "MegaLoc/megaloc_finetuned_paris.pth"
        else:
            print(f"Erreur : {weights_path} est introuvable. Entraînez d'abord le modèle.")
            return

    print(f"Initialisation du modèle avec {weights_path}...")
    model = MegaLoc()
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    model.eval()

    # 2. Charger les données de test
    test_csv = "datasets/paris_75018_gsv/Dataframes/Paris75018_test.csv"
    img_dir = "datasets/paris_75018_gsv/Images"

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
            # La première image sert de référence (Database)
            db_rows.append(group.iloc[0])
            # Toutes les autres servent de requêtes (Queries)
            q_rows.extend([group.iloc[i] for i in range(1, len(group))])
        else:
            # Si le lieu n'a qu'une image, elle ne sert que de bruit dans la database
            db_rows.append(group.iloc[0])

    db_df = pd.DataFrame(db_rows)
    q_df = pd.DataFrame(q_rows)

    print(f"-> Base de référence (Database) : {len(db_df)} images")
    print(f"-> Requêtes (Queries) : {len(q_df)} images")

    # Transformation d'inférence (322x322 pixels conforme au papier MegaLoc)
    val_transform = T.Compose([
        T.Resize((322, 322)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def get_filename(r):
        return f"{r['city_id']}_{r['place_id']:07d}_{r['year']:04d}_{r['month']:02d}_{r['northdeg']:03d}_{r['lat']:.7f}_{r['lon']:.7f}_{r['panoid']}.jpg"

    def extract_descriptors(dataframe, desc_name):
        places = []
        paths = []
        descriptors = []
        with torch.no_grad():
            for _, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc=f"Extraction {desc_name}"):
                fname = get_filename(row)
                path = os.path.join(img_dir, row['city_id'], fname)
                if not os.path.exists(path):
                    continue
                
                img = Image.open(path).convert("RGB")
                tensor = val_transform(img).unsqueeze(0).to(device)
                
                # Extraire le descripteur global de taille [8448]
                desc = model(tensor).cpu().numpy().flatten()
                
                places.append(row['place_id'])
                paths.append(path)
                descriptors.append(desc)
                
        return np.array(places), paths, np.array(descriptors)

    # Extraction des descripteurs
    db_places, db_paths, db_matrix = extract_descriptors(db_df, "Database")
    q_places, q_paths, q_matrix = extract_descriptors(q_df, "Queries")

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

        # Similarité cosine (produit scalaire simple car normés L2)
        sims = np.dot(db_matrix, q_desc)
        
        # Classer du plus similaire au moins similaire
        top_indices = np.argsort(-sims)
        top_places = db_places[top_indices]

        all_top_indices.append(top_indices)
        all_sims.append(sims)

        # Métriques de rappel
        if q_place == top_places[0]:
            correct_r1 += 1
        if q_place in top_places[:5]:
            correct_r5 += 1
        if q_place in top_places[:10]:
            correct_r10 += 1

    print("\n" + "=" * 50)
    print("                RÉSULTATS DE TEST")
    print("=" * 50)
    print(f"  Nombre total de requêtes : {n_queries}")
    print(f"  Recall@1                 : {correct_r1 / n_queries * 100:.2f}%")
    print(f"  Recall@5                 : {correct_r5 / n_queries * 100:.2f}%")
    print(f"  Recall@10                : {correct_r10 / n_queries * 100:.2f}%")
    print("=" * 50)

    # 5. Génération de l'image comparative pour une requête aléatoire
    print("\nGénération de l'image comparative (retrieval_result.png)...")
    q_idx = random.randint(0, n_queries - 1)
    q_place = q_places[q_idx]
    q_path = q_paths[q_idx]
    
    q_sims = all_sims[q_idx]
    q_top3_idx = all_top_indices[q_idx][:3]

    target_size = (300, 300)
    spacing = 15
    width = 4 * target_size[0] + 5 * spacing
    height = target_size[1] + 80
    
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
        draw.text((x_offset + 5, spacing + target_size[1] + 28), subtitle, fill="gray", font=font)

    # 1. Dessiner la Requête (Bordure Bleue)
    add_panel(q_path, "QUERY", spacing, (0, 102, 204), f"Place: {q_place}")

    # 2. Dessiner les 3 meilleures correspondances
    for i, idx in enumerate(q_top3_idx):
        pred_place = db_places[idx]
        pred_path = db_paths[idx]
        sim = q_sims[idx]
        
        is_correct = (pred_place == q_place)
        # Vert si correct, Rouge si incorrect
        border_color = (46, 204, 113) if is_correct else (231, 76, 60)
        
        title = f"TOP-{i+1} " + ("(CORRECT)" if is_correct else "(WRONG)")
        subtitle = f"Sim: {sim:.3f}\nPlace: {pred_place}"
        
        x_offset = (i + 1) * target_size[0] + (i + 2) * spacing
        add_panel(pred_path, title, x_offset, border_color, subtitle)

    output_path = "retrieval_result.png"
    combined_img.save(output_path)
    print(f"Image comparative sauvegardée sous : {os.path.abspath(output_path)}")
    print("=" * 50)

if __name__ == "__main__":
    main()
