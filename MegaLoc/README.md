# 🧠 MegaLoc — Global visual retrieval

Fine-tuning, evaluation and inference for **MegaLoc** (*One Retrieval to Place Them All*, [arXiv:2502.17237](https://arxiv.org/abs/2502.17237)) on a GSV-Cities dataset — typically the Paris dataset produced by [build dataset](../build%20dataset/README.md).

MegaLoc outputs a single **L2-normalised 8448-dimensional global descriptor**, so similarity between two images is just a dot product, which makes retrieval over a whole database nearly free.

## 📁 Structure

```
MegaLoc/
├── MegaLoc_finetuning.py    # Fine-tuning on one or more GSV-Cities sources
├── test_megaloc.py          # Recall@1/5/10 evaluation + comparison sheet
├── infer_megaloc.py         # Single-photo geolocation
├── lib/
│   ├── megaloc_model.py     # Architecture: DINOv2 ViT-B/14 + SALAD aggregator (optimal transport)
│   ├── hubconf.py           # get_trained_model() — pretrained weights from HuggingFace
│   ├── loss.py              # Multi-Similarity Loss (α=2.0, β=50.0, base=0.5)
│   └── gsv_cities.py        # discover_csvs() / resolve_image_path()
└── Results/
    ├── trainings/           # ⛔ git-ignored — .pth weights
    └── test/                # evaluation output
```

## 🏛️ Architecture

```
Image ──► DINOv2 ViT-B/14 ──► SALAD (Sinkhorn / optimal transport) ──► Linear ──► L2Norm ──► [B, 8448]
          768 dim, 12 blocks     64 clusters × 256 + 256-d token
```

- Input dimensions are automatically rounded to a multiple of 14 (patch size).
- Training at **224×224** with RandAugment; inference at **322×322** (as in the paper).
- Pretrained weights are fetched by `hf_hub_download("gberton/MegaLoc", "model.safetensors")` on first call.

## 🔑 Requirements

```bash
pip install torch torchvision pandas numpy pillow tqdm huggingface_hub safetensors
```

Plus a GSV-Cities dataset (see the [root README](../README.md#-shared-dataset-format--gsv-cities)).

## 🎓 Fine-tuning

```bash
python MegaLoc_finetuning.py \
    --dataset_root "/media/rayan/usb/gsv-cities" "/media/rayan/usb/VPR Dataset/paris/gsv_cities" \
    --epochs 5 --lr 1e-5 --batch_size 8 \
    --save_weights Results/trainings/megaloc_finetuned_paris.pth
```

| Option | Default | Purpose |
|---|---|---|
| `--dataset_root` | 2 local paths | One or more GSV-Cities folders. `<City>_train.csv` wins over `<City>.csv` when present |
| `--epochs` | `5` | Number of epochs |
| `--lr` | `1e-5` | Learning rate (AdamW, `weight_decay=1e-3`) |
| `--img_per_place` | `4` | Images sampled per place and per micro-batch |
| `--batch_size` | `8` | **Places** per micro-batch and **per source** (images = `batch_size × img_per_place`) |
| `--save_weights` | `megaloc_finetuned.pth` | Output file |

### What training actually does

**Per-place sampling.** One dataset item = one `place_id` → `img_per_place` images (sampled with replacement if the place has fewer). This is the GSV-Cities/MegaLoc protocol (paper §2: sub-batches of 4 images from 32 distinct places); it gives the Multi-Similarity Loss several positives and negatives per place to mine *within the batch*, instead of a plain anchor/positive pair.

**One source = one `--dataset_root`.** The training loop instantiates one dataset+loader per source and performs a **separate `backward()` per source** at each step (Algorithm 1, "Memory-Efficient GPU Training"): `place_id`s from two sources can never collide since each loss is computed independently, and a source's activation graph is freed before moving to the next — peak memory is therefore that of a single source.

> ⚠️ The split is **per `--dataset_root`, not per CSV**. Passing GSV-Cities' 23 cities as 23 roots would produce 23 sequential `backward()` calls per step: ~10× slower, for no benefit.

**Parameter freezing.** Everything is frozen except the **last 4 Transformer blocks** of the ViT-B (out of 12), the backbone's final `norm`, and the **entire aggregator** (SALAD + linear projection).

**Epoch length.** One epoch covers the largest source; smaller sources loop (`cycle()`, reshuffled on each pass).

## 📊 Evaluation

```bash
python test_megaloc.py \
    -w Results/trainings/megaloc_finetuned_paris.pth \
    --dataset_root "/media/rayan/usb/VPR Dataset/paris/gsv_cities" \
    --dist_threshold 100
```

| Option | Default | Purpose |
|---|---|---|
| `-w`, `--weights_path` | `megaloc_finetuned_paris.pth` | Weights to evaluate (falls back to `MegaLoc/<name>`) |
| `--dataset_root` | local path | One or more GSV-Cities roots (`_test` split when present) |
| `--dist_threshold` | `100.0` | Radius in metres within which a prediction counts as correct |

**Protocol.** For each `place_id` the first image becomes the database reference and the rest become queries; a single-image place stays in the database as a distractor. Places are keyed by `(source, place_id)` so two sources never collide. A prediction is correct if the `place_id` matches **or** the haversine distance is ≤ `--dist_threshold`.

Output: Recall@1/5/10 in the console, plus a `retrieval_result.png` sheet (query in blue, top-5 bordered green/red depending on correctness) for one randomly drawn query.

## 🔍 Single-photo inference

```bash
python infer_megaloc.py \
    --query-image "../Test pics/IMG_0733.png" \
    -w Results/trainings/megaloc_finetuned_paris.pth \
    --dataset_root "/media/rayan/usb/VPR Dataset/paris/gsv_cities"
```

Builds the reference database (first image of each `place_id`), extracts the query descriptor, prints the **top-5** with similarity score, coordinates and a Google Maps link, then writes `megaloc_match.png`.

> 💡 Database descriptors are recomputed on every run. For repeated inference against the same database, prefer [`vpr_pipeline`](../vpr_pipeline/README.md) and its cached FAISS index.

## ⚠️ Gotchas

- **Two lat/lon naming conventions** coexist: full precision (Kaggle GSV-Cities dataset) and rounded to 7 decimals (Paris export). A trailing zero can vanish from the CSV value (`48.881378` vs `48.8813780` in the filename). `resolve_image_path()` tries both, row by row, and returns `None` when neither matches — a missing image is silently skipped during evaluation, but raises `FileNotFoundError` during training.
- `Results/trainings/*.pth` is **git-ignored**: weights must be regenerated or fetched from outside the repository. A MegaLoc checkpoint weighs ~914 MB.
- `lib/gsv_cities.py` is duplicated verbatim in [TextInPlace/lib/](../TextInPlace/lib/) — any fix must be applied on both sides.
