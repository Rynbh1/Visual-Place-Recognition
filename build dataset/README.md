# 🗺️ build dataset — VPR data construction

Two independent tools:

1. **`build_intern_dataset.py`** — builds an *in-house* VPR dataset from the **Mapillary** API for a city or arrondissement, and exports it in the **GSV-Cities** format that MegaLoc / TextInPlace train on directly.
2. **`download_external_datasets.py`** — estimates then downloads public VPR datasets (OSV-5M, MegaScenes, GSV-Cities, SF-XL, MSLS).

Plus two utilities: `extract_zips.py` (parallel extraction with resume) and `to_png.py` (image conversion, HEIC included).

## 📁 Structure

```
build dataset/
├── build_intern_dataset.py       # Mapillary pipeline: estimate / download / reorganize / export
├── download_external_datasets.py # Public datasets: estimation + download
├── extract_zips.py               # Parallel zip extraction with resume
├── to_png.py                     # Image → PNG conversion (HEIC support)
├── secrets_local.py              # ⛔ git-ignored — API tokens
├── paris_lists/                  # ⛔ git-ignored — cached Paris lists (MegaScenes / OSV-5M)
├── datasets/                     # ⛔ git-ignored — download output
└── lib/
    ├── CONSTANTS.py              # Mapillary/Panoramax/KartaView endpoints, average sizes, Leaflet map template
    ├── utils.py                  # Token lookup, folder slug
    ├── geocode.py                # Nominatim/OSM geocoding (bbox + administrative polygon)
    ├── tilling.py                # z14 vector-tile sweep + point-in-polygon
    ├── layout.py                 # Spatial cell binning → one place_id per cell
    ├── download.py               # Parallel image download + manifest.jsonl
    ├── estimate.py               # `estimate` command → dataset_plan.json
    ├── supplement.py             # Panoramax IGN / KartaView top-up + HTML coverage map
    ├── reorganize.py             # Restructure an already downloaded dataset
    ├── export.py                 # GSV-Cities export + geographic train/test split
    └── download_paris_datasets.py# Paris subsets of MegaScenes / OSV-5M
```

## 🔑 Requirements

```bash
pip install requests pillow tqdm pandas huggingface_hub remotezip kaggle pillow_heif
```

Mapillary token, either through an environment variable (takes precedence) or in `secrets_local.py`:

```bash
export MAPILLARY_TOKEN="MLY|xxxxx|yyyyy"
```

`secrets_local.py` (git-ignored) holds `MAPILLARY_TOKEN`, `KAGGLE_API_TOKEN` and `MSLS_URLS`. See the [root README](../README.md#-secrets).

---

## 1️⃣ `build_intern_dataset.py` — Mapillary dataset

The pipeline is deliberately split in **two phases**: you estimate the volume before launching a download that may run for hours.

```
estimate  →  dataset_plan.json  →  download  →  manifest.jsonl  →  export  →  GSV-Cities
                                                       └─ reorganize (optional)
```

### `estimate` — geocode, sweep, quantify

Geocodes the area via Nominatim (bbox + administrative polygon), sweeps Mapillary coverage through zoom-14 vector tiles, filters images (type, polygon, minimum distance), then writes `datasets/<slug>/dataset_plan.json` with the image count and the estimated size and duration.

```bash
python build_intern_dataset.py estimate --city "Paris" --postal 75018
python build_intern_dataset.py estimate --city "Paris" --postal 75011 --image-type flat --resolution 2048
```

| Option | Default | Purpose |
|---|---|---|
| `--city` | *(required)* | City name |
| `--postal` | – | Postal code (mandatory for an arrondissement) |
| `--image-type` | `flat` | `pano` (360°, reference base), `flat` (perspective), `all` |
| `--resolution` | `2048` | `256` / `1024` / `2048` / `original` |
| `--min-dist` | `5.0` | Minimum distance between two images (m) — de-duplication |
| `--place-size` | `25.0` | Cell size defining a place (m) |
| `--max-images` | `0` | Image cap (0 = unlimited) |
| `--method` | `both` | `pano-crop` (360° cropped towards the cell centre), `shared` (perspective images as-is), `both` |
| `--views`, `--min-views`, `--fov`, `--crop-w`, `--crop-h` | `4`, `=views`, `90`, `1024`, `768` | `pano-crop` parameters |
| `--workers`, `--outdir` | `12`, `datasets` | Parallelism and output folder |

> ⚠️ **Mapillary's 360° coverage is very sparse** (~194 panoramas in the 18th arrondissement), so `pano-crop` tops out at a few dozen/hundred images. Perspective images (`flat` + `shared`) are 10–20× more numerous (~6,954 for the 75011) and are the route chosen at arrondissement scale.

### `download` — fetch the images

Reads the validated plan, downloads metadata and images in parallel (resume + backoff), writes `manifest.jsonl`.

```bash
python build_intern_dataset.py download --plan datasets/paris_75018/dataset_plan.json
```

| Option | Default | Purpose |
|---|---|---|
| `--plan` | *(required)* | Path to `dataset_plan.json` |
| `--assume-north-aligned` | off | (`pano-crop`) assume the equirectangular centre is North instead of `compass_angle` |
| `--min-quality` | `0.5` | (`flat`) threshold on Mapillary's `quality_score`, applied only to images where the field is present (⚠️ the help text claims "no filter by default", but the default really is `0.5`) |
| `--min-sharpness` | `5.0` | (`flat`) Laplacian variance: rejects black/white/blurry frames (`0` disables) |

> 🧭 If panoramic crops look visibly rotated, re-run with `--assume-north-aligned`.

### `reorganize` — restructure without re-downloading

Regroups an existing dataset into `places/place_NNNNNN/img_<heading>.jpg` from `manifest.jsonl`, recomputing the cells.

```bash
python build_intern_dataset.py reorganize --dataset datasets/paris_75018 --place-size 25
```

### `export` — GSV-Cities format

Produces `Images/<City>/` plus `Dataframes/<City>.csv`, `<City>_train.csv`, `<City>_test.csv`.

```bash
python build_intern_dataset.py export --dataset datasets/paris_75018 --city-id Paris75018
```

| Option | Default | Purpose |
|---|---|---|
| `--out` | `<dataset>/gsv_cities` | Output folder |
| `--city-id` | folder name | GSV-Cities `city_id` (filename prefix) |
| `--test-ratio` | `0.2` | Fraction of blocks moved to test |
| `--test-block` | `250.0` | Geographic train/test block size (m) |
| `--separation` | `50.0` | Anti-leak buffer: a test place closer than N m to a train place is dropped |
| `--min-imgs` | `4` | Minimum images per place (= `img_per_place` of the GSV-Cities protocol) |

The split is **geographic by blocks**, not random: otherwise two views of the same pavement would land on either side of the split and Recall would be inflated.

---

## 2️⃣ `download_external_datasets.py` — public datasets

Without `--download` the script only **prints estimates** (size + duration). With it, it asks for confirmation (unless `--yes`) then downloads.

```bash
python download_external_datasets.py                                   # estimates only
python download_external_datasets.py --download osv
python download_external_datasets.py --download megascenes --extern-disk /mnt/ext
python download_external_datasets.py --download gsv,sfxl --sfxl-version small --yes
```

| Dataset | Size | Access | Notes |
|---|---|---|---|
| **OSV-5M** | ~259 GB | HuggingFace `osv5m/osv5m`, public | Images live in `images/{split}/{NN}.zip`; the `thumb_original_url` links have expired |
| **MegaScenes** | 3.2 TB total | Public S3 `megascenes.s3.us-west-2.amazonaws.com` | Full download impossible: the script takes **all of Paris** (~14–18 GB) then fills the disk **maximising the number of distinct categories** |
| **GSV-Cities** | ~24 GB | Kaggle `amaralibey/gsv-cities` | Needs `KAGGLE_API_TOKEN` (new `KGAT_...` format, via env var — not username+key) |
| **SF-XL** | 4.7 GB / 366 GB / 2.6 TB | `rsync://vandaldata.polito.it/sf_xl/<version>` | No token, public rsync server |
| **MSLS** | ~56 GB | Mapillary CDN | Temporary links in `secrets_local.MSLS_URLS`, regenerate when they expire (403/404) |

| Option | Default | Purpose |
|---|---|---|
| `--download` | `megascenes` | Comma-separated list: `osv,megascenes,gsv,sfxl,msls` |
| `--dest` | `/media/rayan/usb1` | Target folder |
| `--extern-disk` | – | External disk used both as target **and** as the MegaScenes budget |
| `--size` | – | Total target size in GB (adjusts the MegaScenes share) |
| `--megascenes-budget-gb` | – | Force the MegaScenes budget instead of auto-detection |
| `--recompute-selection` | off | Ignore the `selected_keys.json` cache and recompute the selection |
| `--sfxl-version` | `small` | `small` / `processed` / `raw` |
| `--workers`, `--yes` | `12`, off | Parallelism, skip confirmation |

### Paris subset specifics (`lib/download_paris_datasets.py`)

Paris lists are cached in `paris_lists/` so the ~3 GB of metadata is not re-read on every run.

- **MegaScenes exposes no coordinates**: the Paris filter matches the `paris` **token** in the category name (token match — a substring match would pull in "parish") → 49,643 images.
- The S3 path is non-obvious: `images/{sid//1000:03d}/{sid%1000:03d}/<image column>`, where `sid` is the **parent** category id from `metadata/categories.json`. The parquet's `image` column is based on the **sub**-category, so `sid` must be resolved through `cat`, not by parsing `image`.
- **Spaces** in parquet filenames become **underscores** on S3 (Wikimedia Commons convention): without `key.replace(" ", "_")` you get 404s.
- **OSV-5M**: Paris is filtered from `train.csv`/`test.csv` on the bbox `2.224, 48.815, 2.470, 48.902`, then extracted selectively with `remotezip.RemoteZip` (HTTP range reads). Because the test split is geographic, **no** Paris image is in it: all 1,034 Paris images sit in `train`, and since an id's shard is unknown, all 98 train zips must be opened.

---

## 3️⃣ Utilities

### `extract_zips.py`

Parallel extraction with resume: compares the zip entry count against the already-extracted file count (all present → skip, partial → `unzip -n`, empty → `unzip -o`).

```bash
python extract_zips.py /media/rayan/usb1/osv5m/images/train -w 2
python extract_zips.py dir1 dir2 -o /output --dry-run
```

Use `-w 1` or `2` on a slow USB drive: beyond that the heads spend their time seeking.

### `to_png.py`

```bash
python to_png.py photo.HEIC              # single file
python to_png.py folder/ -o output/ -r   # folder, recursive
```

HEIC/HEIF support (iPhone photos) is enabled when `pillow_heif` is installed.

## 📤 Expected output

```
datasets/paris_75018/
├── dataset_plan.json      # written by `estimate`
├── manifest.jsonl         # written by `download` (one line per image)
├── places/ | images/      # raw images
└── gsv_cities/            # written by `export`
    ├── Dataframes/Paris75018[_train|_test].csv
    └── Images/Paris75018/*.jpg
```

That `gsv_cities/` folder is what you pass as `--dataset_root` to [MegaLoc](../MegaLoc/README.md) and [TextInPlace](../TextInPlace/README.md), and as `--dataset-path` to the [pipeline](../vpr_pipeline/README.md).
