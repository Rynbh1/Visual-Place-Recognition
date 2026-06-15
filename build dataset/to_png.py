#!/usr/bin/env python3
"""Convertit une ou plusieurs images vers le format PNG.

Usage:
    python to_png.py <chemin>                # fichier seul ou dossier
    python to_png.py <chemin> -o <sortie>    # dossier de sortie
    python to_png.py <chemin> -r             # parcours recursif des dossiers

Si <chemin> est un fichier  -> convertit ce fichier.
Si <chemin> est un dossier  -> convertit toutes les images qu'il contient.
"""

import argparse
import sys
from pathlib import Path

from PIL import Image

# Support optionnel des formats HEIC/HEIF (iPhone) si pillow_heif est installe.
try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except ImportError:
    pass

# Extensions traitees comme des images.
IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif",
    ".webp", ".ico", ".ppm", ".pgm", ".pbm", ".tga", ".heic", ".heif",
}


def convert_one(src: Path, out_dir: Path | None) -> bool:
    """Convertit une image en PNG. Renvoie True si succes."""
    dest_dir = out_dir if out_dir is not None else src.parent
    dest = dest_dir / (src.stem + ".png")

    if dest.resolve() == src.resolve():
        print(f"  skip  {src.name} (deja en .png)")
        return True

    try:
        with Image.open(src) as im:
            # Conserve la transparence si presente, sinon RGB.
            if im.mode in ("RGBA", "LA", "P"):
                im = im.convert("RGBA")
            else:
                im = im.convert("RGB")
            dest_dir.mkdir(parents=True, exist_ok=True)
            im.save(dest, "PNG")
        print(f"  ok    {src.name} -> {dest}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  ERREUR {src.name}: {exc}", file=sys.stderr)
        return False


def collect_images(path: Path, recursive: bool) -> list[Path]:
    if path.is_file():
        return [path]
    pattern = "**/*" if recursive else "*"
    return sorted(
        p for p in path.glob(pattern)
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Convertit des images en PNG.")
    parser.add_argument("input", type=Path, help="Fichier image ou dossier")
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Dossier de sortie (defaut: a cote de la source)",
    )
    parser.add_argument(
        "-r", "--recursive", action="store_true",
        help="Parcourt les sous-dossiers",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Chemin introuvable: {args.input}", file=sys.stderr)
        return 1

    images = collect_images(args.input, args.recursive)
    if not images:
        print("Aucune image a convertir.")
        return 0

    print(f"{len(images)} image(s) a convertir.")
    ok = sum(convert_one(img, args.output) for img in images)
    print(f"\nTermine: {ok}/{len(images)} converties.")
    return 0 if ok == len(images) else 1


if __name__ == "__main__":
    raise SystemExit(main())
