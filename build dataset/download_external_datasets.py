#!/usr/bin/env python3
"""Telecharge les datasets VPR : OSV-5M, MegaScenes, GSV-Cities, SF-XL.

Affiche une ESTIMATION (taille + duree) par dataset, puis telecharge ceux
demandes via --download. Par defaut (sans --download) : estimation seule.

Contraintes reelles (cf. recherche) :
  - OSV-5M    : ~259 Go (HuggingFace osv5m/osv5m, public, scriptable).
  - MegaScenes: 3,2 To complet -> IMPOSSIBLE en entier. Le script telecharge
    TOUT Paris (cache paris_lists/) PUIS remplit l'espace disque restant en
    MAXIMISANT LA DIVERSITE (le plus de categories/scenes distinctes possible).
    --extern-disk pointe un disque externe pour disposer de plus d'espace.
  - GSV-Cities: ~24 Go (Kaggle amaralibey/gsv-cities, besoin token API Kaggle).
  - SF-XL     : small ~4.7 Go / processed ~366 Go / raw ~2.6 To. Telechargement
    par RSYNC depuis un serveur PUBLIC (aucun token) :
    rsync://vandaldata.polito.it/sf_xl/<version>

=== IDENTIFIANTS (dans secrets_local.py) ===
  - KAGGLE_API_TOKEN : token Kaggle (nouveau format KGAT_...), pour GSV-Cities.
    Obtenir : https://www.kaggle.com/settings -> "Create New API Token".
  - SF-XL : aucun token (rsync public).

Exemples :
    python download_datasets.py                              # estimations seules
    python download_datasets.py --download osv
    python download_datasets.py --download megascenes --extern-disk /mnt/ext
    python download_datasets.py --download gsv,sfxl --sfxl-version small --yes
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import lib.download_paris_datasets as paris  # reutilise la logique Paris validee

# Identifiants (fichier git-ignore ; voir secrets_local.py).
try:
    import secrets_local
    KAGGLE_API_TOKEN = getattr(secrets_local, "KAGGLE_API_TOKEN", "")
    MSLS_URLS = getattr(secrets_local, "MSLS_URLS", [])
except ImportError:
    KAGGLE_API_TOKEN = ""
    MSLS_URLS = []

SFXL_RSYNC = "rsync://vandaldata.polito.it/sf_xl"

GB = 1024 ** 3
SCRIPT_DIR = Path(__file__).resolve().parent

# Tailles connues (Go) pour les estimations et le budget disque.
OSV_FULL_GB = 259.0
GSV_GB = 24.0
SFXL_GB = {"small": 4.7, "processed": 366.0, "raw": 2600.0}
MS_FULL_GB = 3200.0
MSLS_TOTAL_GB = 56.0
MS_AVG_BYTES = 290 * 1024          # taille moyenne mesuree d'une image MegaScenes
MS_PARIS_GB = 49643 * MS_AVG_BYTES / GB  # ~13.7 Go
DISK_MARGIN_GB = 10              # marge de securite laissee libre


def gb(n_bytes):
    return n_bytes / GB


# --------------------------------------------------------------------------
# OSV-5M (complet, HuggingFace)
# --------------------------------------------------------------------------
def osv_download_full(dest, workers):
    from huggingface_hub import snapshot_download
    print("  OSV-5M : telechargement complet depuis HuggingFace (~259 Go)...")
    snapshot_download(
        repo_id="osv5m/osv5m", repo_type="dataset",
        local_dir=str(dest / "osv5m"),
        allow_patterns=["images/*", "*.csv", "*.json"],
        max_workers=max(1, workers),
    )
    print(f"  OSV-5M -> {dest / 'osv5m'}")


# --------------------------------------------------------------------------
# MegaScenes (Paris + remplissage diversite selon l'espace disque)
# --------------------------------------------------------------------------
def ms_select_by_diversity(meta_dir, budget_bytes, recompute=False):
    """Choisit les cles S3 a telecharger : TOUT Paris, puis un maximum de
    categories DISTINCTES (diversite) tant qu'on tient dans budget_bytes.

    Strategie : on couvre d'abord le plus de categories possibles (1 image
    chacune), puis on approfondit (plus d'images/categorie) si le budget reste.

    La selection est mise en cache sur disque (selected_keys.json). Le budget
    reel depend de l'espace disque LIBRE, qui diminue au fur et a mesure du
    telechargement : sans cache, relancer le script apres une interruption
    recalculerait un budget plus petit et choisirait un sous-ensemble DIFFERENT
    (donc pas une vraie reprise). Avec le cache, la liste cible est figee des
    le premier run ; les fichiers deja presents sont simplement sautes par
    `_ms_download_one`, ce qui donne une reprise fiable. Utiliser
    `recompute=True` pour forcer un nouveau calcul (efface le cache).
    """
    import json
    import pyarrow.parquet as pq

    cache_path = meta_dir / "selected_keys.json"
    if cache_path.exists() and not recompute:
        selected = json.loads(cache_path.read_text(encoding="utf-8"))
        print(f"  MegaScenes : reprise depuis la selection en cache "
              f"({len(selected)} images, ~{gb(len(selected) * MS_AVG_BYTES):.1f} Go) "
              f"-> {cache_path}")
        return selected

    # categories.json (cat -> scene id) ; deja telecharge si cache Paris construit
    cats_path = meta_dir / "categories.json"
    if not cats_path.exists():
        paris.download_file(paris.MS_CATEGORIES_URL, cats_path, "MegaScenes categories.json")
    cat2sid = json.loads(cats_path.read_text(encoding="utf-8"))

    pq_path = meta_dir / "images_index.parquet"
    if not pq_path.exists():
        paris.download_file(paris.MS_PARQUET_URL, pq_path, "MegaScenes parquet")

    # 1) regroupe les images (cle relative) par categorie ; marque Paris.
    by_cat = {}          # cat -> [image_rel, ...]
    paris_keywords = ["paris"]
    pf = pq.ParquetFile(pq_path)
    for batch in pf.iter_batches(columns=["cat", "image"], batch_size=300_000):
        cs = batch.column("cat").to_pylist()
        ims = batch.column("image").to_pylist()
        for c, im in zip(cs, ims):
            if c and im and c in cat2sid:
                by_cat.setdefault(c, []).append(im)

    def full_key(cat, im):
        return f"{paris._ms_scene_path(cat2sid[cat])}/{im}"

    paris_cats = [c for c in by_cat if paris._is_paris_cat(c, paris_keywords)]
    other_cats = sorted(c for c in by_cat if c not in set(paris_cats))

    selected = []
    # TOUT Paris (toujours)
    for c in paris_cats:
        for im in by_cat[c]:
            selected.append(full_key(c, im))
    paris_bytes = len(selected) * MS_AVG_BYTES
    remaining = max(0, budget_bytes - paris_bytes)
    max_extra = int(remaining // MS_AVG_BYTES)

    print(f"  MegaScenes : {len(selected)} images Paris (~{gb(paris_bytes):.1f} Go), "
          f"budget restant pour la diversite : {gb(remaining):.1f} Go "
          f"(~{max_extra} images)")

    if max_extra > 0 and other_cats:
        if len(other_cats) >= max_extra:
            # budget < nb categories -> on echantillonne les categories (stride)
            # pour couvrir le plus large possible, 1 image par categorie retenue.
            stride = len(other_cats) / max_extra
            picks = [other_cats[int(i * stride)] for i in range(max_extra)]
            for c in picks:
                selected.append(full_key(c, by_cat[c][0]))
        else:
            # toutes les categories + on approfondit (cap par categorie).
            # cap k tel que sum(min(count,k)) ~ max_extra.
            lo, hi = 1, max(len(v) for v in by_cat.values())
            counts = [len(by_cat[c]) for c in other_cats]
            def total(k):
                return sum(min(ci, k) for ci in counts)
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if total(mid) <= max_extra:
                    lo = mid
                else:
                    hi = mid - 1
            k = lo
            for c in other_cats:
                for im in by_cat[c][:k]:
                    selected.append(full_key(c, im))
            print(f"    -> toutes les categories couvertes, ~{k} image(s)/categorie")

    n_cats_covered = len(paris_cats) + (len(other_cats)
                                        if max_extra >= len(other_cats)
                                        else min(max_extra, len(other_cats)))
    print(f"  MegaScenes : {len(selected)} images selectionnees "
          f"(~{gb(len(selected) * MS_AVG_BYTES):.1f} Go), "
          f"{n_cats_covered} categories distinctes")
    cache_path.write_text(json.dumps(selected), encoding="utf-8")
    return selected


def ms_download_selection(keys, out_dir, workers):
    out_dir.mkdir(parents=True, exist_ok=True)
    import os
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm
    already = sum(1 for k in keys if (out_dir / k).exists() and (out_dir / k).stat().st_size > 0)
    if already:
        print(f"  MegaScenes : {already}/{len(keys)} images deja presentes, reprise du telechargement...")
    total = 0
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        futs = [pool.submit(paris._ms_download_one, k, out_dir) for k in keys]
        for f in tqdm(as_completed(futs), total=len(futs), desc="  MegaScenes"):
            total += f.result()
        pool.shutdown(wait=True)
    except KeyboardInterrupt:
        # Ne PAS attendre les threads en cours (bloques jusqu'a 120s sur leur
        # requete HTTP) : les telechargements deja passes en .tmp -> rename
        # sont surs a interrompre, donc on sort immediatement (os._exit
        # court-circuite le join des threads fait par l'atexit de CPython).
        print(f"\n  MegaScenes : interrompu ({total} nouvelles images telechargees). "
              f"Relance le script pour reprendre.")
        pool.shutdown(wait=False, cancel_futures=True)
        os._exit(130)
    print(f"  MegaScenes : {total} nouvelles images -> {out_dir}")


# --------------------------------------------------------------------------
# GSV-Cities (Kaggle)
# --------------------------------------------------------------------------
def gsv_download(dest):
    if not KAGGLE_API_TOKEN:
        print("  GSV-Cities : ERREUR -- renseigne KAGGLE_API_TOKEN dans "
              "secrets_local.py (https://www.kaggle.com/settings -> "
              "Create New API Token).", file=sys.stderr)
        return
    os.environ["KAGGLE_API_TOKEN"] = KAGGLE_API_TOKEN
    import kaggle  # import APRES avoir pose le token (sinon auth fail a l'import)
    out = dest / "gsv-cities"
    out.mkdir(parents=True, exist_ok=True)
    print("  GSV-Cities : telechargement Kaggle (~24 Go)...")
    kaggle.api.dataset_download_files("amaralibey/gsv-cities", path=str(out), unzip=True)
    print(f"  GSV-Cities -> {out}")


# --------------------------------------------------------------------------
# SF-XL (rsync, serveur public ; aucun token)
# --------------------------------------------------------------------------
def sfxl_download(dest, version):
    out = dest / "sf-xl"
    out.mkdir(parents=True, exist_ok=True)
    src = f"{SFXL_RSYNC}/{version}"
    print(f"  SF-XL ({version}) : rsync depuis {src} ...")
    cmd = ["rsync", "-rhz", "--info=progress2", "--ignore-existing",
           f"{src}", str(out) + "/"]
    rc = subprocess.call(cmd)
    if rc != 0:
        print(f"  SF-XL : rsync a echoue (code {rc})", file=sys.stderr)
    else:
        print(f"  SF-XL -> {out / version}")


# --------------------------------------------------------------------------
# MSLS (Mapillary Street-Level Sequences)
# URLs CDN temporaires stockees dans secrets_local.MSLS_URLS (expirent).
# --------------------------------------------------------------------------
def msls_download(dest, urls):
    if not urls:
        print("  MSLS : ERREUR -- MSLS_URLS vide dans secrets_local.py.\n"
              "         Reconnecte-toi sur https://www.mapillary.com/dataset/places "
              "et copie les nouveaux liens.", file=sys.stderr)
        return
    out = dest / "msls"
    out.mkdir(parents=True, exist_ok=True)
    for url in urls:
        fname = url.split("?")[0].rsplit("/", 1)[-1]
        fpath = out / fname
        if fpath.exists() and fpath.stat().st_size > 0:
            print(f"  MSLS : {fname} deja present, skip.")
            continue
        print(f"  MSLS : {fname}...")
        # wget -c : reprise automatique si interruption
        rc = subprocess.call(["wget", "-c", "-q", "--show-progress",
                              "-O", str(fpath), url])
        if rc != 0:
            print(f"  MSLS : echec {fname} (code {rc}) — lien peut-etre expire.",
                  file=sys.stderr)
    print(f"  MSLS -> {out}")


# --------------------------------------------------------------------------
def compute_ms_budget(args, dest, enabled):
    """Espace (octets) alloue a MegaScenes selon le disque cible et les autres
    datasets actives sur le meme disque."""
    if getattr(args, "size", None) is not None:
        other_datasets_gb = 0.0
        if "osv" in enabled:
            other_datasets_gb += OSV_FULL_GB
        if "gsv" in enabled:
            other_datasets_gb += GSV_GB
        if "sfxl" in enabled:
            other_datasets_gb += SFXL_GB[args.sfxl_version]
        if "msls" in enabled:
            other_datasets_gb += MSLS_TOTAL_GB
        
        ms_budget_gb = max(0.0, args.size - other_datasets_gb)
        return ms_budget_gb * GB
    if args.megascenes_budget_gb is not None:
        return args.megascenes_budget_gb * GB
    target = Path(args.extern_disk) if args.extern_disk else dest
    target.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(target).free
    reserved = 0.0
    same_disk = (not args.extern_disk) or (
        Path(args.extern_disk).resolve() == dest.resolve())
    if same_disk:
        if "osv" in enabled:
            reserved += OSV_FULL_GB * GB
        if "gsv" in enabled:
            reserved += GSV_GB * GB
        if "sfxl" in enabled:
            reserved += SFXL_GB[args.sfxl_version] * GB
    return max(0, free - DISK_MARGIN_GB * GB - reserved)


def print_estimates(args, dest, enabled):
    target = Path(args.extern_disk) if args.extern_disk else dest
    free = shutil.disk_usage(target if target.exists() else dest).free
    ms_budget = compute_ms_budget(args, dest, enabled)
    ms_est = min(MS_FULL_GB * GB, max(MS_PARIS_GB * GB, ms_budget))
    print("\n" + "=" * 64)
    print(f"  ESTIMATIONS  (disque cible libre : {gb(free):.0f} Go"
          + (f", extern : {args.extern_disk}" if args.extern_disk else "") + ")")
    print("-" * 64)
    msls_n = len(MSLS_URLS)
    msls_note = f"{msls_n} fichiers" if msls_n else "MSLS_URLS vide dans secrets_local.py"
    print(f"  OSV-5M       : ~{OSV_FULL_GB:.0f} Go   (HuggingFace, complet)")
    print(f"  MegaScenes   : ~{gb(ms_est):.0f} Go   (Paris ~{MS_PARIS_GB:.0f} Go "
          f"+ diversite jusqu'a remplir le budget ; complet = {MS_FULL_GB/1024:.1f} To)")
    print(f"  GSV-Cities   : ~{GSV_GB:.0f} Go    (Kaggle)")
    print(f"  SF-XL        : ~{SFXL_GB[args.sfxl_version]:.0f} Go  "
          f"(version '{args.sfxl_version}' ; small 4.7 / processed 366 Go / raw 2.6 To)")
    print(f"  MSLS         : ~{MSLS_TOTAL_GB:.0f} Go   (CDN Mapillary, {msls_note})")
    print("-" * 64)
    if enabled:
        # duree tres approximative a ~15 Mo/s pour les gros transferts
        total = 0.0
        if "osv" in enabled: total += OSV_FULL_GB
        if "megascenes" in enabled: total += gb(ms_est)
        if "gsv" in enabled: total += GSV_GB
        if "sfxl" in enabled: total += SFXL_GB[args.sfxl_version]
        if "msls" in enabled: total += MSLS_TOTAL_GB
        print(f"  A TELECHARGER: {', '.join(enabled)}  ~ {total:.0f} Go  "
              f"(~{total*1024/15/3600:.1f} h a 15 Mo/s)")
    print("=" * 64)


def main():
    ap = argparse.ArgumentParser(description="Telecharge les datasets VPR (estimation + download).")
    ap.add_argument("--download", default="megascenes",
                    help="Datasets a telecharger : osv,megascenes,gsv,sfxl,msls (defaut: megascenes seul)")
    ap.add_argument("--dest", type=Path, default="/media/rayan/usb1")
    ap.add_argument("--extern-disk", default=None,
                    help="Chemin d'un disque externe (cible + budget de MegaScenes)")
    ap.add_argument("--size", type=float, default=None,
                    help="Taille totale cible pour le telechargement en Go (ajuste MegaScenes)")
    ap.add_argument("--megascenes-budget-gb", type=float, default=None,
                    help="Force le budget MegaScenes (Go) au lieu de l'auto-detection")
    ap.add_argument("--recompute-selection", action="store_true",
                    help="Ignore le cache de selection MegaScenes et la recalcule "
                         "(sinon reprise automatique depuis selected_keys.json)")
    ap.add_argument("--sfxl-version", choices=["small", "processed", "raw"], default="small")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()

    dest = args.dest
    dest.mkdir(parents=True, exist_ok=True)
    enabled = [d.strip() for d in args.download.split(",") if d.strip()]
    valid = {"osv", "megascenes", "gsv", "sfxl", "msls"}
    bad = set(enabled) - valid
    if bad:
        print(f"Datasets inconnus : {bad}. Valides : {valid}", file=sys.stderr)
        return 2

    print_estimates(args, dest, enabled)

    if not enabled:
        print("\n(Estimation seule. Ajoute --download osv,megascenes,gsv,sfxl pour telecharger.)")
        return 0

    if not args.yes:
        if input("\nLancer le telechargement ? [y/N] ").strip().lower() not in ("y", "yes", "o", "oui"):
            print("Annule.")
            return 0

    print("\n=== Telechargement ===")
    if "osv" in enabled:
        osv_download_full(dest, args.workers)
    if "megascenes" in enabled:
        meta_dir = SCRIPT_DIR / "datasets" / "paris_external" / "_metadata"
        budget = compute_ms_budget(args, dest, enabled)
        keys = ms_select_by_diversity(meta_dir, budget, recompute=args.recompute_selection)
        ms_target = (Path(args.extern_disk) if args.extern_disk else dest) / "megascenes"
        ms_download_selection(keys, ms_target, args.workers)
    if "gsv" in enabled:
        gsv_download(dest)
    if "sfxl" in enabled:
        sfxl_download(dest, args.sfxl_version)
    if "msls" in enabled:
        msls_download(dest, MSLS_URLS)
    print(f"\nTermine -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
