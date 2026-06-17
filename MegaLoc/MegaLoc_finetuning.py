#!/usr/bin/env python3
import os
import random
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

# Ajouter MegaLoc-main au path pour pouvoir importer ses modules locaux
from lib.hubconf import get_trained_model
from lib.loss import MultiSimilarityLoss


# ---------------------------------------------------------------------------
# 2. Dataset au format GSV-Cities
# ---------------------------------------------------------------------------
class GSVCitiesFineTuneDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform

        self.place_to_indices = (
            self.df.groupby("place_id").apply(lambda x: x.index.tolist()).to_dict()
        )
        self.places = list(self.place_to_indices.keys())

    def __len__(self):
        return len(self.places)

    def __getitem__(self, idx):
        place_id = self.places[idx]
        img_indices = self.place_to_indices[place_id]

        # Échantillonner deux images distinctes du même lieu (Positifs)
        if len(img_indices) >= 2:
            idx_anchor, idx_positive = random.sample(img_indices, 2)
        else:
            idx_anchor = img_indices[0]
            idx_positive = img_indices[0]

        row_a = self.df.iloc[idx_anchor]
        row_p = self.df.iloc[idx_positive]

        def get_filename(r):
            return f"{r['city_id']}_{r['place_id']:07d}_{r['year']:04d}_{r['month']:02d}_{r['northdeg']:03d}_{r['lat']:.7f}_{r['lon']:.7f}_{r['panoid']}.jpg"

        file_a = os.path.join(self.img_dir, row_a["city_id"], get_filename(row_a))
        file_p = os.path.join(self.img_dir, row_p["city_id"], get_filename(row_p))

        img_a = Image.open(file_a).convert("RGB")
        img_p = Image.open(file_p).convert("RGB")

        if self.transform:
            img_a = self.transform(img_a)
            img_p = self.transform(img_p)

        return img_a, img_p, place_id


# ---------------------------------------------------------------------------
# 3. Pipeline d'entraînement
# ---------------------------------------------------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(description="MegaLoc Fine-tuning Script")
    parser.add_argument(
        "--train_csv",
        type=str,
        default="/home/rayan/Documents/github/Visual Place Recognition/datasets/paris_75019/Dataframes/Paris75019_train.csv",
        help="Path to the training CSV file",
    )
    parser.add_argument(
        "--img_dir",
        type=str,
        default="/home/rayan/Documents/github/Visual Place Recognition/datasets/paris_75019/Images",
        help="Path to the images directory",
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument(
        "--save_weights",
        type=str,
        default="megaloc_finetuned_paris.pth",
        help="Output filename for fine-tuned weights",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # Transformations d'images (224x224 pour l'entraînement + RandAugment)
    train_transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.RandAugment(num_ops=2, magnitude=9),  # Conforme au papier
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_csv = args.train_csv
    img_dir = args.img_dir

    if not os.path.exists(train_csv):
        print(f"Erreur : Exporter d'abord le dataset avec mapillary_dataset.py export")
        return

    dataset = GSVCitiesFineTuneDataset(train_csv, img_dir, transform=train_transform)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True
    )

    print("Chargement de MegaLoc pré-entraîné...")
    model = get_trained_model().to(device)
    model.train()

    # --- CONFIGURATION DU GEL DES PARAMÈTRES (Règle du Papier) ---
    # 1. Figer tout le modèle
    for param in model.parameters():
        param.requires_grad = False

    # 2. Dégeler les 4 dernières couches (Transformer Blocks) du backbone ViT-B (qui en contient 12)
    for block in model.backbone.blocks[-4:]:
        for param in block.parameters():
            param.requires_grad = True

    # Dégeler la couche finale de normalisation du backbone
    for param in model.backbone.norm.parameters():
        param.requires_grad = True

    # 3. Dégeler l'agrégateur complet (SALAD + projection linéaire)
    for param in model.aggregator.parameters():
        param.requires_grad = True

    # Optimiseur AdamW sur les paramètres actifs
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,  # Taux d'apprentissage adapté au fine-tuning
        weight_decay=1e-3,
    )

    criterion = MultiSimilarityLoss()

    epochs = args.epochs
    for epoch in range(epochs):
        epoch_loss = 0.0
        loop = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")

        for imgs_a, imgs_p, place_ids in loop:
            # Concaténer ancre et positif pour optimiser le calcul des descripteurs
            imgs = torch.cat([imgs_a, imgs_p], dim=0).to(device)
            # Dupliquer les étiquettes pour chaque couple
            labels = torch.cat([place_ids, place_ids], dim=0).to(device)

            optimizer.zero_grad()

            # Extraction globale [2*B, 8448] (déjà normalisé L2)
            embeddings = model(imgs)

            # Calcul de la Multi-Similarity Loss
            loss = criterion(embeddings, labels)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        print(f"Perte moyenne Epoch {epoch + 1} : {epoch_loss / len(dataloader):.4f}")

    # Sauvegarder
    torch.save(model.state_dict(), args.save_weights)
    print(f"Modèle fine-tuné sauvegardé avec succès sous '{args.save_weights}'")


if __name__ == "__main__":
    main()
