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
from lib.gsv_cities import discover_csvs, resolve_image_path


# ---------------------------------------------------------------------------
# 1. Dataset au format GSV-Cities, un lieu -> img_per_place images (defaut 4,
#    echantillonnees avec remise si le lieu en a moins). C'est le protocole
#    GSV-Cities / MegaLoc (papier MegaLoc Sec. 2 : sous-batch de "4 images
#    ... from 32 different places/classes"), qui donne a la Multi-Similarity
#    Loss plusieurs positifs/negatifs par lieu a miner dans le batch, au lieu
#    d'une simple paire ancre/positif.
#
#    Une instance = UNE source (= UN --dataset_root, ex: tout gsv-cities/, ou
#    tout VPR Dataset/paris/gsv_cities/), qui peut regrouper plusieurs CSV (une
#    ville = un CSV chez GSV-Cities). Le training loop instancie un
#    dataset+loader par source et fait un backward() separe par source a
#    chaque step (papier MegaLoc, Algorithm 1 "Memory-Efficient GPU
#    Training") : ca evite toute collision de place_id entre sources (chaque
#    loss est calculee independamment sur son propre batch) et ca reduit le
#    pic memoire, le graphe d'activations d'une source etant libere avant de
#    passer a la suivante. Important : le decoupage en sources se fait par
#    --dataset_root, PAS par CSV individuel — sinon les 23 villes de
#    gsv-cities deviendraient 23 sources et donc 23 backward() sequentiels
#    par step, ce qui ralentit l'entrainement d'un facteur ~10 sans benefice.
# ---------------------------------------------------------------------------
class GSVCitiesQuadrupletDataset(Dataset):
    def __init__(self, specs, img_per_place=4, transform=None):
        """specs : liste de (csv_path, img_dir) regroupees en une seule source."""
        self.img_per_place = img_per_place
        self.transform = transform
        self.dfs = []
        self.img_dirs = []
        self.place_to_indices = {}

        for df_idx, (csv_path, img_dir) in enumerate(specs):
            df = pd.read_csv(csv_path, dtype={"panoid": str})
            self.dfs.append(df)
            self.img_dirs.append(img_dir)
            groups = df.groupby("place_id").apply(lambda x: x.index.tolist()).to_dict()
            for place_id, indices in groups.items():
                self.place_to_indices[(df_idx, place_id)] = indices

        self.place_keys = list(self.place_to_indices.keys())

    def _resolve_path(self, img_dir, r):
        path = resolve_image_path(img_dir, r)
        if path is None:
            raise FileNotFoundError(
                f"Image introuvable sous {img_dir}/{r['city_id']} pour "
                f"place_id={r['place_id']}, panoid={r['panoid']}"
            )
        return path

    def __len__(self):
        return len(self.place_keys)

    def __getitem__(self, idx):
        df_idx, place_id = self.place_keys[idx]
        df = self.dfs[df_idx]
        img_dir = self.img_dirs[df_idx]
        img_indices = self.place_to_indices[(df_idx, place_id)]

        if len(img_indices) >= self.img_per_place:
            chosen = random.sample(img_indices, self.img_per_place)
        else:
            chosen = random.choices(img_indices, k=self.img_per_place)

        imgs = []
        for i in chosen:
            img = Image.open(self._resolve_path(img_dir, df.iloc[i])).convert("RGB")
            if self.transform:
                img = self.transform(img)
            imgs.append(img)
        return torch.stack(imgs), idx


def cycle(dataloader):
    """Reitere indefiniment un DataLoader en le re-melangeant a chaque tour
    (contrairement a itertools.cycle qui figerait l'ordre du 1er passage)."""
    while True:
        for batch in dataloader:
            yield batch


# ---------------------------------------------------------------------------
# 2. Pipeline d'entraînement
# ---------------------------------------------------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(description="MegaLoc Fine-tuning Script")
    parser.add_argument(
        "--dataset_root",
        type=str,
        nargs="+",
        default=[
            "/media/rayan/usb/gsv-cities",
            "/media/rayan/usb/VPR Dataset/paris/gsv_cities",
        ],
        help="Un ou plusieurs dossiers au format GSV-Cities "
        "(contenant Dataframes/<Ville>.csv + Images/<Ville>/). "
        "Si <Ville>_train.csv existe, il est utilise en priorite sur <Ville>.csv.",
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument(
        "--img_per_place", type=int, default=4,
        help="Images echantillonnees par lieu et par micro-batch (protocole GSV-Cities/MegaLoc : 4)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="Lieux par micro-batch ET PAR SOURCE (images par micro-batch = batch_size * img_per_place)",
    )
    parser.add_argument(
        "--save_weights",
        type=str,
        default="megaloc_finetuned.pth",
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

    print(f"Sources d'entrainement ({len(args.dataset_root)} dataset_root) :")
    loaders = []
    for root in args.dataset_root:
        specs = discover_csvs(root, split="train")
        if not specs:
            print(f"Erreur : aucun CSV trouve sous {root}/Dataframes")
            return
        ds = GSVCitiesQuadrupletDataset(specs, args.img_per_place, transform=train_transform)
        print(f"  {root} -> {len(specs)} CSV, {len(ds)} lieux")
        loaders.append(
            DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)
        )

    # Une epoque couvre la source la plus grande ; les sources plus petites bouclent
    # (re-melangees a chaque tour, cf. cycle()).
    steps_per_epoch = max(len(l) for l in loaders)

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
        iterators = [cycle(l) for l in loaders]
        loop = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch + 1}/{epochs}")

        for _ in loop:
            optimizer.zero_grad()
            step_loss = 0.0

            # Un backward() par source (Algorithm 1 de MegaLoc) : les gradients
            # s'accumulent, le graphe d'activations est libere apres chaque
            # backward() -> pic memoire = celui d'UNE source a la fois.
            for it in iterators:
                imgs, place_idx = next(it)
                b, n, c, h, w = imgs.shape
                imgs = imgs.view(b * n, c, h, w).to(device)
                labels = place_idx.repeat_interleave(n).to(device)

                # Extraction globale [B*img_per_place, 8448] (déjà normalisé L2)
                embeddings = model(imgs)
                loss = criterion(embeddings, labels)
                (loss / len(loaders)).backward()
                step_loss += loss.item()

            optimizer.step()

            epoch_loss += step_loss / len(loaders)
            loop.set_postfix(loss=step_loss / len(loaders))

        print(f"Perte moyenne Epoch {epoch + 1} : {epoch_loss / steps_per_epoch:.4f}")

    # Sauvegarder
    torch.save(model.state_dict(), args.save_weights)
    print(f"Modèle fine-tuné sauvegardé avec succès sous '{args.save_weights}'")


if __name__ == "__main__":
    main()
