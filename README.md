# 📍 Visual Place Recognition — Paris

End-to-end **visual place recognition** (VPR) stack: building a geolocated street-level dataset, fine-tuning two retrieval models (MegaLoc and TextInPlace), then running a hybrid inference pipeline that fuses the visual signal, scene text, and a conditional geometric verification step.

Goal: given a photo taken in the street, find the matching place in a reference database and emit its GPS coordinates — or **abstain** when confidence is insufficient.

## 🗂️ Repository layout

```
Visual Place Recognition/
├── build dataset/     # Dataset construction (Mapillary) + public dataset downloads
├── MegaLoc/           # Global visual retrieval: fine-tuning / testing / inference
├── TextInPlace/       # Scene-text + visual retrieval: fine-tuning / testing / inference
├── vpr_pipeline/      # Unified hybrid pipeline (retrieval + OCR + LLM + RANSAC)
├── evaluate_vpr.ipynb # Notebook: side-by-side evaluation of both models and the pipeline
└── Test pics/         # Query photos for manual testing
```

Each folder has its own README:

| Module | Role | README |
|---|---|---|
| `build dataset` | Generates a GSV-Cities dataset from the Mapillary API; also fetches OSV-5M, MegaScenes, GSV-Cities, SF-XL | [build dataset/README.md](build%20dataset/README.md) |
| `MegaLoc` | 8448-d global descriptor (DINOv2 ViT-B + SALAD), fine-tuned on Paris | [MegaLoc/README.md](MegaLoc/README.md) |
| `TextInPlace` | Retrieval combining a visual descriptor with detected scene text (Bridge/AdelaiDet) | [TextInPlace/README.md](TextInPlace/README.md) |
| `vpr_pipeline` | End-to-end evaluation and inference CLI | [vpr_pipeline/README.md](vpr_pipeline/README.md) |

## 🔄 Workflow

```
   Mapillary API                        Public datasets
 (street-level photos)          (OSV-5M, MegaScenes, GSV-Cities, SF-XL)
         │                                     │
         ▼                                     ▼
  build_intern_dataset.py            download_external_datasets.py
  estimate → download → export                 │
         │                                     │
         └──────────────┬──────────────────────┘
                        ▼
                GSV-Cities formatted dataset
        Dataframes/<City>[_train|_test].csv
        Images/<City>/*.jpg
                        │
          ┌─────────────┴─────────────┐
          ▼                           ▼
  MegaLoc_finetuning.py     TextInPlace_finetuning.py
          │                           │
          ▼                           ▼
   *.pth (Paris weights)       *.pth (Paris weights)
          │                           │
          └─────────────┬─────────────┘
                        ▼
                   vpr_pipeline
          eval (Recall@K) / infer (GPS or abstention)
```

## 📦 Shared dataset format — GSV-Cities

Every module reads and writes the same layout:

```
<dataset_root>/
├── Dataframes/
│   ├── <City>.csv          # full manifest
│   ├── <City>_train.csv    # training split (optional)
│   └── <City>_test.csv     # test split (optional)
└── Images/
    └── <City>/
        └── <City>_<place_id:07d>_<year:04d>_<month:02d>_<northdeg:03d>_<lat>_<lon>_<panoid>.jpg
```

CSV columns used: `city_id`, `place_id`, `year`, `month`, `northdeg`, `lat`, `lon`, `panoid`.
Image paths are resolved through `lib/gsv_cities.py` (`discover_csvs`, `resolve_image_path`), duplicated identically in `MegaLoc/lib/` and `TextInPlace/lib/`.

Shared convention: **a place (`place_id`) must hold at least 4 images** — that is the GSV-Cities/MegaLoc protocol (`img_per_place=4`), which gives the Multi-Similarity Loss several positives per place inside the batch.

## ⚙️ Installation

```bash
python3 -m venv venv
source venv/bin/activate

# Common base
pip install torch torchvision numpy pandas pillow tqdm requests scikit-learn

# Retrieval / index
pip install faiss-cpu huggingface_hub safetensors pytorch-metric-learning

# Dataset construction
pip install remotezip kaggle pillow_heif

# Pipeline extras
pip install easyocr          # OCR fallback
pip install lightglue        # RANSAC geometric verification

# Notebook
pip install jupyter matplotlib
```

Tested with Python 3.12, torch 2.12, CUDA. Everything runs on CPU (`--cpu` flag in the pipeline) but descriptor extraction is then very slow.

TextInPlace additionally requires cloning the upstream repository and installing Detectron2/AdelaiDet — see [TextInPlace/README.md](TextInPlace/README.md).

## 🔑 Secrets

Credentials live in `build dataset/secrets_local.py` (**git-ignored, never commit it**):

| Key | Used for | Where to get it |
|---|---|---|
| `MAPILLARY_TOKEN` | Internal dataset construction | https://www.mapillary.com/dashboard/developers |
| `KAGGLE_API_TOKEN` | GSV-Cities download (`KGAT_...` format) | https://www.kaggle.com/settings → *Create New API Token* |
| `MSLS_URLS` | Temporary MSLS CDN links (they expire, regenerate as needed) | https://www.mapillary.com/dataset/places |

`MAPILLARY_TOKEN` can also be passed as an environment variable, which takes precedence:

```bash
export MAPILLARY_TOKEN="MLY|xxxxx|yyyyy"
```

SF-XL requires no token (public rsync server).

## 🚀 Quick start

```bash
# 1. Build a dataset for one arrondissement
cd "build dataset"
python build_intern_dataset.py estimate --city "Paris" --postal 75018
python build_intern_dataset.py download --plan datasets/paris_75018/dataset_plan.json
python build_intern_dataset.py export   --dataset datasets/paris_75018

# 2. Fine-tune MegaLoc on it
cd ../MegaLoc
python MegaLoc_finetuning.py \
    --dataset_root "../build dataset/datasets/paris_75018/gsv_cities" \
    --save_weights Results/trainings/megaloc_finetuned_paris.pth

# 3. Evaluate
python test_megaloc.py -w Results/trainings/megaloc_finetuned_paris.pth \
    --dataset_root "../build dataset/datasets/paris_75018/gsv_cities"

# 4. Geolocate a photo with the full pipeline
cd ..
python -c "from vpr_pipeline.main import main; main()" infer \
    --image-path "Test pics/IMG_0733.png" \
    --dataset-path "build dataset/datasets/paris_75018/gsv_cities" \
    --use-ransac
```

> ℹ️ `infer` relies on relative imports while `eval` uses an absolute one, so the two sub-commands are not launched the same way. See the [known limitations](vpr_pipeline/README.md#-known-limitations).

## 📓 Evaluation notebook

[`evaluate_vpr.ipynb`](evaluate_vpr.ipynb) evaluates MegaLoc, TextInPlace and the pipeline on one shared database/query split under one metric definition — which the standalone `test_*.py` scripts do not provide, since they default to different correctness radii. It ends with a single-image inference demo on `Test pics/IMG_0733.png` plus a map of the retrieved locations.

Results are reported **per model family, never as a cross-model ranking**: MegaLoc is a general-purpose global retriever, TextInPlace a scene-text-assisted one, and putting their Recall side by side on street imagery with no legible signage would describe the dataset rather than the models. Comparisons stay inside a family — released checkpoint vs local training, with and without text re-ranking.

```bash
jupyter notebook evaluate_vpr.ipynb
```

Every section degrades gracefully: a missing checkpoint or an uninstalled dependency (Detectron2, LightGlue, Ollama) skips that section instead of breaking the notebook.

## 🚫 What is not versioned

`.gitignore` excludes heavyweight artefacts and secrets: `venv/`, `__pycache__/`, `*.pth`, `*.png`, `build dataset/datasets/`, `build dataset/paris_lists/`, `build dataset/secrets_local.py`, `TextInPlace/repo/`, `TextInPlace/weights/`, `TextInPlace/logs/`.

As a result, a fresh clone requires **re-downloading the weights, re-cloning `TextInPlace/repo/`, and rebuilding the datasets**.

## 📚 References

- **MegaLoc: One Retrieval to Place Them All** — [arXiv:2502.17237](https://arxiv.org/abs/2502.17237)
- **TextInPlace** — [GitHub](https://github.com/HqiTao/TextInPlace)
- **GSV-Cities** — [GitHub](https://github.com/amaralibey/gsv-cities) · [Kaggle](https://www.kaggle.com/datasets/amaralibey/gsv-cities)
- **To Match or Not to Match** (conditional RANSAC, CVPR 2025)
- **LightGlue** — [GitHub](https://github.com/cvg/LightGlue)
