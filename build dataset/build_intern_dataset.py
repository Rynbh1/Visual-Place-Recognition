#!/usr/bin/env python3
"""Pipeline de creation d'un dataset VPR a partir de l'API Mapillary.

Entree : un nom de ville (ou un arrondissement + code postal).
Sortie : un dossier d'images street-level geolocalisees + un manifeste
         (id, GPS, cap/orientation, date, type de camera, sequence...).

La pipeline se deroule en DEUX phases pour pouvoir valider avant de telecharger :

  1) estimate : geocode la zone (Nominatim/OSM), balaye la couverture Mapillary
                via les tuiles vectorielles, compte les images, estime la
                taille et la duree, puis ecrit un plan (dataset_plan.json).

  2) download : lit le plan valide, recupere les metadonnees + les images
                en parallele (avec reprise et backoff), ecrit le manifeste.

Exemples :
    export MAPILLARY_TOKEN="MLY|xxx|yyy"
    python build_intern_dataset.py estimate --city "Paris" --postal 75011
    python build_intern_dataset.py download --plan datasets/paris_75011/dataset_plan.json
    python build_intern_dataset.py export --dataset datasets/paris_75011
"""

import argparse
import sys
from pathlib import Path

# Add the script's parent directory to sys.path to allow relative imports of lib package
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib.estimate import cmd_estimate
from lib.download import cmd_download
from lib.reorganize import cmd_reorganize
from lib.export import cmd_export


def build_parser():
    p = argparse.ArgumentParser(description="Pipeline dataset VPR via Mapillary.")
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--city", required=True, help="Nom de la ville")
    common.add_argument(
        "--postal",
        default=None,
        help="Code postal (obligatoire pour un arrondissement)",
    )
    common.add_argument(
        "--image-type",
        choices=["pano", "flat", "all"],
        default="flat",
        help="pano=panoramas 360 (base reference P2E), "
        "flat=perspectives (requetes), all=tout",
    )
    common.add_argument(
        "--resolution", choices=["256", "1024", "2048", "original"], default="2048"
    )
    common.add_argument(
        "--min-dist",
        type=float,
        default=5.0,
        help="Distance min entre 2 images en m (anti-doublons)",
    )
    common.add_argument(
        "--place-size",
        type=float,
        default=25.0,
        help="Taille de cellule d'un lieu en m (1 sous-dossier/lieu)",
    )
    common.add_argument(
        "--max-images", type=int, default=0, help="Plafond d'images (0 = illimite)"
    )
    common.add_argument("--workers", type=int, default=12)
    common.add_argument("--outdir", default="datasets")
    common.add_argument(
        "--method",
        choices=["pano-crop", "shared", "both"],
        default="pano-crop",
        help="pano-crop=panoramas 360 recadres vers le centre ; "
        "shared=perspectives partagees telles quelles ; "
        "both=les deux dans le meme lieu (pano_<cap>.jpg + flat_<cap>.jpg)",
    )
    common.add_argument(
        "--views",
        type=int,
        default=4,
        help="(pano-crop) nb de vues par lieu, reparties sur le pourtour du carre",
    )
    common.add_argument(
        "--min-views",
        type=int,
        default=None,
        help="(pano-crop) nb min de panos pour garder un carre "
        "(defaut = --views ; mettre 1 pour tout garder)",
    )
    common.add_argument(
        "--fov", type=float, default=90.0,
        help="(pano-crop) champ de vision horizontal des crops en degres",
    )
    common.add_argument(
        "--crop-w", type=int, default=1024, help="(pano-crop) largeur des crops"
    )
    common.add_argument(
        "--crop-h", type=int, default=768, help="(pano-crop) hauteur des crops"
    )

    pe = sub.add_parser(
        "estimate", parents=[common], help="Geocode + balaye + estime + ecrit le plan"
    )
    pe.set_defaults(func=cmd_estimate)

    pd = sub.add_parser("download", help="Telecharge a partir d'un plan valide")
    pd.add_argument("--plan", required=True, help="Chemin du dataset_plan.json")
    pd.add_argument(
        "--assume-north-aligned",
        action="store_true",
        help="(pano-crop) supposer les panoramas alignes au Nord "
        "(centre image = Nord) au lieu de centre = compass_angle",
    )
    pd.add_argument(
        "--min-quality",
        type=float,
        default=0.5,
        help="(flat) seuil sur le quality_score Mapillary (ex: 0.5). "
        "Defaut: pas de filtre (le champ varie selon les images)",
    )
    pd.add_argument(
        "--min-sharpness",
        type=float,
        default=5.0,
        help="(flat) seuil de nettete local (variance de Laplacien) ; "
        "rejette images noires/blanches/floues. Mettre 0 pour desactiver",
    )
    pd.set_defaults(func=cmd_download)

    pr = sub.add_parser(
        "reorganize",
        help="Restructure un dataset deja telecharge (sans re-DL) "
        "en sous-dossiers de lieux + nommage par angle",
    )
    pr.add_argument(
        "--dataset", required=True, help="Dossier du dataset (contenant manifest.jsonl)"
    )
    pr.add_argument(
        "--place-size",
        type=float,
        default=25.0,
        help="Taille de cellule d'un lieu en m",
    )
    pr.set_defaults(func=cmd_reorganize)

    px = sub.add_parser(
        "export",
        help="Exporte un dataset telecharge au format GSV-Cities "
        "(entrainable par MegaLoc) + split train/test geographique",
    )
    px.add_argument(
        "--dataset", required=True, help="Dossier du dataset (contenant manifest.jsonl)"
    )
    px.add_argument("--out", default=None, help="Dossier de sortie (defaut: <dataset>/gsv_cities)")
    px.add_argument("--city-id", default=None,
                    help="Nom de ville (city_id GSV-Cities ; defaut: nom du dossier)")
    px.add_argument("--test-ratio", type=float, default=0.2,
                    help="Proportion de blocs en test (defaut: 0.2)")
    px.add_argument("--test-block", type=float, default=250.0,
                    help="Taille des blocs train/test en m (defaut: 250)")
    px.add_argument("--separation", type=float, default=50.0,
                    help="Zone tampon train/test en m, anti-fuite (defaut: 50)")
    px.add_argument("--min-imgs", type=int, default=4,
                    help="Nb min d'images par lieu pour le garder (defaut: 4, "
                    "= img_per_place de GSV-Cities)")
    px.set_defaults(func=cmd_export)
    return p


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
