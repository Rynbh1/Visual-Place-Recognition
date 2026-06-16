import csv
import json
from pathlib import Path
from lib.layout import build_layout

def cmd_reorganize(args):
    out_dir = Path(args.dataset)
    manifest_path = out_dir / "manifest.jsonl"
    if not manifest_path.exists():
        raise SystemExit(f"manifest.jsonl introuvable dans {out_dir}")

    rows = [json.loads(l) for l in manifest_path.read_text().splitlines() if l.strip()]
    layout = build_layout(rows, args.place_size)

    moved, missing = 0, 0
    new_rows = []
    for row in rows:
        place_id, fname = layout[str(row["id"])]
        rel = f"places/{place_id}/{fname}"
        dest = out_dir / rel
        # ancien emplacement : valeur 'file' du manifeste, sinon images/{id}.jpg
        old = row.get("file", f"{row['id']}.jpg")
        src = out_dir / old
        if not src.exists():
            src = out_dir / "images" / Path(old).name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            pass  # deja en place
        elif src.exists():
            src.rename(dest)
            moved += 1
        else:
            missing += 1
        row["place"] = place_id
        row["file"] = rel
        new_rows.append(row)

    # reecrit le manifeste (jsonl + csv) avec place + chemins relatifs
    with manifest_path.open("w", encoding="utf-8") as mf:
        for row in new_rows:
            mf.write(json.dumps(row, ensure_ascii=False) + "\n")
    if new_rows:
        ordered = [
            "id",
            "place",
            "file",
            "lon",
            "lat",
            "compass_angle",
            "captured_at",
            "camera_type",
            "is_pano",
            "width",
            "height",
            "sequence",
        ]
        keys = [k for k in ordered if k in new_rows[0]]
        with (out_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
            wri = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            wri.writeheader()
            wri.writerows(new_rows)

    # supprime l'ancien dossier images/ s'il est vide
    old_img = out_dir / "images"
    if old_img.is_dir() and not any(old_img.iterdir()):
        old_img.rmdir()

    n_places = len({r["place"] for r in new_rows})
    print(f"  {moved} fichiers deplaces, {missing} manquants.")
    print(f"  {len(new_rows)} images reparties en {n_places} lieux.")
    print(f"  Structure : {out_dir / 'places'}/place_NNNNNN/img_<angle>.jpg")
    return 0
