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
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
repo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "repo")
sys.path.append(repo_path)
sys.path.append(os.path.join(repo_path, "detectron2"))

from network import STVGLNet
from backbone import setup_cfg
from utils import util, commons
from pytorch_metric_learning import losses, miners, distances
from lib.gsv_cities import discover_csvs, resolve_image_path

IMAGENET_MEAN_STD = {
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225]
}

# ---------------------------------------------------------------------------
# Dataset au format GSV-Cities, un lieu -> img_per_place images (defaut 4,
# echantillonnees avec remise si le lieu en a moins). Ce protocole (4 images
# par lieu) est celui utilise pour l'entrainement de GSV-Cities/MegaLoc et pour
# le fine-tuning de TextInPlace lui-meme (papier TextInPlace, Sec. III-E.2 :
# "batch size set to 64 places, each represented by 4 images"), au lieu d'une
# simple paire ancre/positif qui prive la Multi-Similarity Loss (et son miner)
# d'exemples intra-lieu pour miner des positifs/negatifs difficiles.
#
# Une instance = UNE source (= UN --dataset_root, ex: tout gsv-cities/, ou
# tout VPR Dataset/paris/gsv_cities/), qui peut regrouper plusieurs CSV (une
# ville = un CSV chez GSV-Cities). Le training loop instancie un dataset+loader
# par source et fait un backward() separe par source a chaque step (cf.
# MegaLoc, Algorithm 1 "Memory-Efficient GPU Training") : ca evite toute
# collision de place_id entre sources (chaque loss est calculee independamment)
# et ca reduit le pic memoire, le graphe d'activations d'une source etant
# libere avant de passer a la suivante. Important : le decoupage en sources se
# fait par --dataset_root, PAS par CSV individuel — sinon les 23 villes de
# gsv-cities deviendraient 23 sources et donc 23 backward() sequentiels par
# step, ce qui ralentit l'entrainement d'un facteur ~10 sans aucun benefice.
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

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_config = os.path.join(script_dir, "repo", "configs", "Bridge", "TotalText", "R_50_poly.yaml")
    default_init = os.path.join(script_dir, "repo", "checkpoints", "best_model.pth")
    default_spotter = os.path.join(script_dir, "repo", "checkpoints", "Bridge_tt.pth")

    import argparse
    parser = argparse.ArgumentParser(description="TextInPlace Fine-tuning Script")
    parser.add_argument("--dataset_root", type=str, nargs="+",
                        default=[
                            "/media/rayan/usb/gsv-cities",
                            "/media/rayan/usb/VPR Dataset/paris/gsv_cities",
                        ],
                        help="Un ou plusieurs dossiers au format GSV-Cities "
                        "(contenant Dataframes/<Ville>.csv + Images/<Ville>/). "
                        "Si <Ville>_train.csv existe, il est utilise en priorite sur <Ville>.csv.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--img_per_place", type=int, default=4,
                        help="Images echantillonnees par lieu et par micro-batch "
                        "(protocole GSV-Cities/MegaLoc/TextInPlace : 4)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Lieux par micro-batch ET PAR SOURCE "
                        "(images par micro-batch = batch_size * img_per_place)")
    parser.add_argument("--grad_accum_steps", type=int, default=4,
                        help="Micro-batches accumules par source avant chaque optimizer.step() "
                        "(cf. MegaLoc Algorithm 1 : backward() separe par source, "
                        "gradients accumules, memoire liberee entre chaque)")
    parser.add_argument("--save_weights", type=str, default="textinplace_finetuned.pth",
                        help="Output filename for fine-tuned weights")
    parser.add_argument("--config-file", type=str, default=default_config,
                        help="Path to Detectron2/AdelaiDet config file")
    parser.add_argument("--confidence-threshold", type=float, default=0.3,
                        help="Minimum score for instance predictions")
    parser.add_argument("--features-dim", type=int, default=16384,
                        help="VPR features dimension (row_dim = features-dim // 512 ; "
                        "auto-detecte depuis --init-weights si fourni). "
                        "ATTENTION : l'ancien defaut 768 donnait row_dim=1, soit un "
                        "descripteur de 512 dims au lieu des 16384 du modele publie.")
    parser.add_argument("--init-weights", type=str, default=default_init,
                        help="Checkpoint TextInPlace complet servant de point de depart "
                        "(tete VPR + spotter). Chaine vide = entrainement from scratch. "
                        "Sans lui, STVGLNet part d'une agregation BoQ aleatoire : ce n'est "
                        "plus un fine-tuning.")
    parser.add_argument("--spotter-weights", type=str, default=default_spotter,
                        help="Poids du text spotter, injectes dans cfg.MODEL.WEIGHTS. "
                        "Base.yaml n'en definit aucun et Backbone.__init__ ne charge le "
                        "spotter que 'if cfg.MODEL.WEIGHTS' : sans ca il reste ALEATOIRE, "
                        "gele, et les features d'entree de la branche VPR aussi.")
    parser.add_argument("--max-steps", type=int, default=0,
                        help="Arrete apres N optimizer.step() (0 = illimite). Pour verifier "
                        "qu'un entrainement demarre sans attendre une epoque complete.")
    parser.add_argument("--use-amp16", action="store_true", default=True,
                        help="Use Automatic Mixed Precision")
    parser.add_argument("--opts", help="Modify config options", default=[], nargs=argparse.REMAINDER)
    args = parser.parse_args()

    # Le spotter est charge par Backbone.__init__ via cfg.MODEL.WEIGHTS, donc avant setup_cfg.
    if args.spotter_weights and os.path.exists(args.spotter_weights):
        args.opts = ["MODEL.WEIGHTS", args.spotter_weights] + list(args.opts)
    elif args.spotter_weights:
        print(f"Attention : spotter introuvable ({args.spotter_weights}) -> "
              "il restera aleatoire et ne detectera jamais de texte.")

    # La dimension du descripteur doit correspondre au checkpoint de depart, sinon
    # aggregation.fc est de la mauvaise forme et le chargement echoue.
    if args.init_weights and os.path.exists(args.init_weights):
        init_sd = torch.load(args.init_weights, map_location="cpu", weights_only=False)
        init_sd = init_sd.get("model_state_dict", init_sd)
        for k, v in init_sd.items():
            if k.endswith("aggregation.fc.weight"):
                detected = v.shape[0] * 512
                if detected != args.features_dim:
                    print(f"features-dim ajuste a {detected} (lu dans {args.init_weights})")
                    args.features_dim = detected
                break
    else:
        init_sd = None
        if args.init_weights:
            print(f"Attention : --init-weights introuvable ({args.init_weights}) -> "
                  "entrainement FROM SCRATCH (agregation BoQ aleatoire).")

    # Log setup
    start_time = datetime.now()
    save_dir = os.path.join("logs", "dinov2_vitb14_cosgem", "paris_finetune", start_time.strftime('%Y-%m-%d_%H-%M-%S'))
    commons.setup_logging(save_dir, console="info")

    print("=" * 60)
    print("  STARTING OPTIMIZED TEXTINPLACE VPR FINE-TUNING PIPELINE  ")
    print("=" * 60)
    print(f"Places per micro-batch (per source): {args.batch_size} x {args.img_per_place} img/place")
    print(f"Grad accum steps (per source): {args.grad_accum_steps}")
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

    print(f"Sources d'entrainement ({len(args.dataset_root)} dataset_root) :")
    loaders = []
    for root in args.dataset_root:
        specs = discover_csvs(root, split="train")
        if not specs:
            print(f"Error: no CSV found under {root}/Dataframes")
            return
        ds = GSVCitiesQuadrupletDataset(specs, args.img_per_place, transform=train_transform)
        print(f"  {root} -> {len(specs)} CSV, {len(ds)} lieux")
        loaders.append(DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2,
                                   drop_last=True, pin_memory=True))

    # Une epoque couvre la source la plus grande ; les sources plus petites bouclent
    # (re-melangees a chaque tour, cf. cycle()).
    steps_per_epoch = max(len(l) for l in loaders) // args.grad_accum_steps

    print(f"Setting up Detectron2 config from {args.config_file}...")
    cfg = setup_cfg(args)
    print(f"Spotter (cfg.MODEL.WEIGHTS) : {cfg.MODEL.WEIGHTS or 'AUCUN -> aleatoire'}")
    print(f"Descripteur                 : {args.features_dim} dims "
          f"(row_dim={args.features_dim // 512})")
    model = STVGLNet(cfg)

    if init_sd is not None:
        # Meme normalisation que test_textinplace.py : prefixe module. laisse par
        # DataParallel, et cles du spotter a re-prefixer en backbone.textmodel.*
        from collections import OrderedDict

        if next(iter(init_sd)).startswith("module"):
            init_sd = OrderedDict({k.replace("module.", ""): v for k, v in init_sd.items()})
        normalised = OrderedDict()
        for k, v in init_sd.items():
            if k.startswith(("dptext_detr.", "recognizer.", "bridge.")) and not k.startswith(
                "backbone.textmodel."
            ):
                normalised[f"backbone.textmodel.{k}"] = v
            else:
                normalised[k] = v
        try:
            model.load_state_dict(normalised, strict=True)
            print(f"Point de depart             : {args.init_weights} (charge, strict)")
        except RuntimeError as exc:
            missing, unexpected = model.load_state_dict(normalised, strict=False)
            print(f"Point de depart             : {args.init_weights} (charge, NON strict)")
            print(f"  cles manquantes : {len(missing)} | inattendues : {len(unexpected)}")
            print(f"  cause : {str(exc)[:200]}")
    else:
        print("Point de depart             : AUCUN -> entrainement from scratch "
              "(agregation BoQ aleatoire)")

    model = model.to(device)

    # Print trainable parameters info
    util.print_trainable_parameters(model)
    util.print_trainable_layers(model)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-3,
    )

    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.2, total_iters=args.epochs * steps_per_epoch)
    criterion = losses.MultiSimilarityLoss(alpha=1.0, beta=50, base=0.0, distance=distances.CosineSimilarity())
    miner = miners.MultiSimilarityMiner(epsilon=0.1, distance=distances.CosineSimilarity())

    scaler = torch.amp.GradScaler('cuda') if args.use_amp16 and device.type == 'cuda' else None

    model.train()
    epochs = args.epochs
    n_micro = len(loaders) * args.grad_accum_steps
    global_step = 0
    stop = False
    for epoch in range(epochs):
        if stop:
            break
        epoch_loss = 0.0
        n_steps = 0
        iterators = [cycle(l) for l in loaders]
        loop = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch + 1}/{epochs}")

        for _ in loop:
            optimizer.zero_grad()
            step_loss = 0.0

            # Un backward() par micro-batch, par source (Algorithm 1 de MegaLoc) :
            # les gradients s'accumulent, le graphe d'activations est libere apres
            # chaque backward() -> pic memoire = celui d'UNE source a la fois.
            for it in iterators:
                for _ in range(args.grad_accum_steps):
                    imgs, place_idx = next(it)
                    b, n, c, h, w = imgs.shape
                    imgs = imgs.view(b * n, c, h, w).to(device)
                    labels = place_idx.repeat_interleave(n).to(device)

                    if scaler is not None:
                        with torch.amp.autocast('cuda'):
                            features = model(imgs)
                            miner_outputs = miner(features, labels)
                            loss = criterion(features, labels, miner_outputs)
                        scaler.scale(loss / n_micro).backward()
                    else:
                        features = model(imgs)
                        miner_outputs = miner(features, labels)
                        loss = criterion(features, labels, miner_outputs)
                        (loss / n_micro).backward()

                    step_loss += loss.item()

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()

            epoch_loss += step_loss / n_micro
            n_steps += 1
            global_step += 1
            loop.set_postfix(loss=step_loss / n_micro)

            if args.max_steps and global_step >= args.max_steps:
                print(f"\n--max-steps {args.max_steps} atteint -> arret anticipe.")
                stop = True
                break

        print(f"Average Epoch {epoch + 1} Loss: {epoch_loss / max(1, n_steps):.4f}")

    # Save weights
    torch.save(model.state_dict(), args.save_weights)
    print(f"Fine-tuned model weights saved successfully to '{args.save_weights}'")

if __name__ == "__main__":
    main()
