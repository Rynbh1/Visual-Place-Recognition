# 🔤 TextInPlace — Visual + scene-text retrieval

Fine-tuning, evaluation and inference for **TextInPlace** ([HqiTao/TextInPlace](https://github.com/HqiTao/TextInPlace)) on a GSV-Cities dataset.

The idea: in a street, shop signs, plaques and street numbers carry place information that a purely visual descriptor dilutes. TextInPlace hangs a retrieval head (BoQ) off a **frozen scene-text spotter** (DPText-DETR, via Detectron2/AdelaiDet) and can re-rank candidates using the detected strings.

## ⛔ Heavyweight prerequisites — install these first

Neither the upstream code nor the weights are versioned here (`repo/`, `weights/`, `logs/` are git-ignored).

### 1. Clone the upstream repository into `repo/`

```bash
cd TextInPlace
git clone https://github.com/HqiTao/TextInPlace.git repo
```

Every script prepends `repo/` and `repo/detectron2/` to `sys.path` and imports `network`, `backbone`, `utils`, `aggregations` from there.

### 2. Install Detectron2 / AdelaiDet

Follow the instructions in `repo/` (`repo/requirements.txt`, `repo/setup.py`). This is the most fragile step of the whole project: Detectron2 must be built against the exact torch/CUDA versions installed.

### 3. Get the text-spotter weights

```
TextInPlace/repo/checkpoints/best_model.pth   # upstream weights, default --weights-path for infer_textinplace.py
TextInPlace/weights/textinplace_finetuned.pth # locally fine-tuned weights (~440 MB)
```

### 4. Config file

All commands default to:

```
repo/configs/Bridge/TotalText/R_50_poly.yaml
```

overridable with `--config-file`, and patchable key by key with `--opts KEY VALUE ...` (forwarded to Detectron2).

## 📁 Structure

```
TextInPlace/
├── TextInPlace_finetuning.py  # Fine-tuning on one or more GSV-Cities sources
├── test_textinplace.py        # Recall@1/5/10 evaluation (+ optional text re-ranking)
├── infer_textinplace.py       # Single-photo geolocation
├── lib/gsv_cities.py          # discover_csvs() / resolve_image_path()  (copy of MegaLoc/lib/)
├── repo/                      # ⛔ git-ignored — cloned upstream repository
├── weights/                   # ⛔ git-ignored — checkpoints
└── logs/                      # ⛔ git-ignored — timestamped training logs
```

## 🏛️ Architecture

```
Image ─► DPText-DETR (frozen) ─┬─► text predictions ──► decoded strings (string.printable vocabulary)
                               │
                               └─► early features ─► ResNet50 layer2+layer3 ─► BoQ ─► descriptor
                                                      (trainable)             64 queries, 2 layers
```

- `STVGLNet` (training) does a single forward pass; `STVGLNet_test` splits `forward()` (text detection) from `vpr_branch()` (descriptor), so detection is paid for only once.
- The text spotter is **fully frozen**: only ResNet50 `layer2`/`layer3` and the BoQ aggregation learn.
- Input resolution: **320×320** for both training and testing.

## 🎓 Fine-tuning

```bash
python TextInPlace_finetuning.py \
    --dataset_root "/media/rayan/usb/VPR Dataset/paris/gsv_cities" \
    --epochs 5 --batch_size 8 --grad_accum_steps 4 \
    --save_weights weights/textinplace_finetuned.pth
```

| Option | Default | Purpose |
|---|---|---|
| `--dataset_root` | 2 local paths | One or more GSV-Cities roots (`<City>_train.csv` preferred) |
| `--epochs` / `--lr` | `5` / `1e-5` | AdamW `weight_decay=1e-3` + `LinearLR` decaying to 0.2× |
| `--img_per_place` | `4` | Images per place and per micro-batch |
| `--batch_size` | `8` | **Places** per micro-batch and **per source** |
| `--grad_accum_steps` | `4` | Micro-batches accumulated per source before each `optimizer.step()` |
| `--use-amp16` | on | Mixed precision (`torch.amp`) |
| `--init-weights` | `repo/checkpoints/best_model.pth` | Full TextInPlace checkpoint to start from (VPR head + spotter). Empty string = train from scratch |
| `--spotter-weights` | `repo/checkpoints/Bridge_tt.pth` | Text-spotter weights, injected into `cfg.MODEL.WEIGHTS` before the model is built |
| `--features-dim` | `16384` | VPR dimension — `row_dim = features_dim // 512`; auto-adjusted to match `--init-weights` |
| `--max-steps` | `0` | Stop after N `optimizer.step()` (0 = unlimited). Checks a run starts without waiting for a full epoch |
| `--confidence-threshold` | `0.3` | Minimum score for text instances (does **not** reach the detector — see below) |
| `--save_weights` | `textinplace_finetuned.pth` | Output file |

The first three defaults are the fix for the three defects documented below. Verified on a 3-step
run: the spotter loads from `Bridge_tt.pth`, `features-dim` auto-adjusts to 16384 (`row_dim = 32`),
`best_model.pth` loads **strict**, and the resulting checkpoint detects scene text out of the box
(3/3 images) — none of which was true of `weights/textinplace_finetuned.pth`.

**Loss**: `MultiSimilarityLoss(alpha=1.0, beta=50, base=0.0)` over cosine distance, with `MultiSimilarityMiner(epsilon=0.1)` — the miner picks hard positives/negatives inside the batch, which assumes several images per place (hence `img_per_place=4`, paper §III-E.2: "batch size set to 64 places, each represented by 4 images").

**One source = one `--dataset_root`.** As with MegaLoc, one `backward()` per source and per micro-batch (MegaLoc's Algorithm 1): no `place_id` collision across sources, and peak memory limited to one source at a time. The split is per root, **not per CSV** — otherwise GSV-Cities' 23 cities would become 23 sequential backward() calls per step (~10× slower).

Logs are written to `logs/dinov2_vitb14_cosgem/paris_finetune/<timestamp>/`.

> ⚠️ **`--features-dim` trap**: the default is `768` for training but `16384` for evaluation and inference. Both of the latter **auto-detect** the dimension from the checkpoint's `aggregation.fc.weight` (`features_dim = row_dim × 512`), so a checkpoint is always read correctly whatever value you pass; however, two training runs launched with different `--features-dim` produce non-interchangeable weights.

## 📊 Evaluation

```bash
python test_textinplace.py \
    -w weights/textinplace_finetuned.pth \
    --dataset_root "/media/rayan/usb/VPR Dataset/paris/gsv_cities" \
    --use-text-rerank --dist_threshold 100
```

| Option | Default | Purpose |
|---|---|---|
| `-w`, `--weights_path` | `textinplace_finetuned.pth` | Weights to evaluate (falls back to `TextInPlace/<name>`) |
| `--dataset_root` | local path | GSV-Cities roots (`_test` split when present) |
| `--use-text-rerank` | off | Enable scene-text re-ranking |
| `--dist_threshold` | `100.0` | Validation radius (m) for a prediction |
| `--config-file`, `--confidence-threshold`, `--features-dim`, `--opts` | see above | Detectron2 configuration |

**Protocol** — identical to [MegaLoc's](../MegaLoc/README.md#-evaluation): first image of each `place_id` in the database, the rest as queries, places keyed by `(source, place_id)`, correctness by `place_id` **or** haversine distance ≤ threshold. Output: Recall@1/5/10 + a `retrieval_result.png` sheet.

**Text re-ranking** (`--use-text-rerank`): re-ranks only the **top 100** visual candidates, and only from strings **containing at least one digit** (street numbers, postcodes — purely alphabetic words are too generic to discriminate). The score is a common-string length ratio:

```
score = Σ len(common strings) / Σ len(query strings)
```

With no numeric string in the query, the visual ranking is left untouched — so if the spotter finds
no digits at all, re-ranking is a mathematical no-op and Recall is identical to the digit.

## 🐛 The local fine-tuning starts from nothing

`TextInPlace_finetuning.py:191` is `model = STVGLNet(cfg).to(device)` — **no weights are loaded**:
not `MODEL.WEIGHTS` for the text spotter, not `repo/checkpoints/best_model.pth` for the VPR head.
Only ResNet50 `layer2`/`layer3` come pretrained (ImageNet, via `torchvision`). The run is therefore
a 5-epoch training from scratch, not a fine-tune — with a randomly initialised BoQ head, a randomly
initialised frozen spotter, and a 512-d descriptor (see the `--features-dim` trap below).

The released checkpoint `repo/checkpoints/best_model.pth` **is** a trained TextInPlace: epoch 5,
16384-d descriptor, spotter matching `Bridge_tt.pth`. Measured on 200 places / 879 queries at
320 px:

| Checkpoint | R@1 (25 m) | R@5 | R@10 | descriptor |
|---|---|---|---|---|
| upstream `best_model.pth` | **50.17 %** | 66.44 % | 71.22 % | 16384-d |
| local `textinplace_finetuned.pth` | 49.72 % | 66.21 % | 71.33 % | 512-d |

Five epochs from scratch land within half a point of the released model — that is what the missing
initialisation costs. Starting the fine-tuning from `best_model.pth` (and passing `MODEL.WEIGHTS`)
is what would turn Paris data into an actual gain.

## 🐛 The text branch is silently dead — three cumulative causes

Measured on this repository's `weights/textinplace_finetuned.pth`: **0 strings detected over 1079
images**. Three independent defects stack up, and fixing any one of them alone changes nothing.

### 1. The text spotter is never loaded, so it stays random

`configs/Bridge/Base.yaml` defines no `MODEL.WEIGHTS`, and `Backbone.__init__` loads the spotter
only `if cfg.MODEL.WEIGHTS`. Fine-tuning therefore built a **randomly initialised** DPText-DETR,
froze it (`requires_grad = False`) and saved it into the checkpoint. Evidence: 1 of 1067 spotter
tensors match `repo/checkpoints/Bridge_tt.pth`, versus 1045 of 1067 for the upstream
`best_model.pth`. Detection scores top out at **0.012** — nothing can pass any threshold.

```bash
# fine-tuning and evaluation both need the spotter weights
--opts MODEL.WEIGHTS TextInPlace/repo/checkpoints/Bridge_tt.pth
```

This is **not only an OCR problem**: `Backbone.frozen_layers` is the spotter's ResNet `stem` +
`stages[0]`, so the VPR head was trained on top of random early features as well. The 512-d
descriptor works *despite* its input, not because of it — re-running the fine-tuning with the
weights above is the real fix.

### 2. Loading the spotter is not enough — `frozen_layers` overwrites it

`Backbone.frozen_layers` is an `nn.Sequential` over the *same module objects* as
`textmodel…backbone.stem` and `stages[0]`. The checkpoint stores those tensors under **both**
names (55 duplicated entries), so `load_state_dict` puts the checkpoint's random copy right back
over whatever `MODEL.WEIGHTS` loaded. Any loader that mixes the two sources must drop both
`backbone.textmodel.*` and `backbone.frozen_layers.*`.

### 3. `--confidence-threshold` does not reach the detector, and 320 px is too small

`setup_cfg()` writes that flag to the `RETINANET`, `ROI_HEADS`, `FCOS`, `MEInst` and
`PANOPTIC_FPN` knobs, but DPText-DETR reads `MODEL.TRANSFORMER.INFERENCE_TH_TEST`, left at `0.4`.
With a properly loaded spotter the scores reach only 0.2–0.3 on this imagery, so `0.4` still
rejects everything. And both scripts feed 320×320 while the config expects `MIN_SIZE_TEST: 1000`:
measured on 5 images, **0/5** yield text at 320 px against **5/5** at 1000 px.

Working combination, verified:

```bash
--opts MODEL.WEIGHTS TextInPlace/repo/checkpoints/Bridge_tt.pth \
       MODEL.TRANSFORMER.INFERENCE_TH_TEST 0.15
# plus a 1000×1000 input transform instead of 320×320
```

### Result once all three are fixed

`evaluate_vpr.ipynb` applies the three fixes and runs the text branch as a **separate pass**
(pretrained spotter at 1000 px) from the descriptor pass (checkpoint as trained, 320 px), because
swapping the spotter changes the features the VPR head was trained on. Measured on 200 places /
879 queries:

| | Result |
|---|---|
| strings detected | **13,265** over 1,079 images — 879/879 queries, 200/200 database images |
| typical strings | `RESTAURANT`, `WELCOME`, `WHISTLESTOP` |
| digit-bearing strings | **0** — so the upstream re-rank rule stays a no-op |
| R@1 within 25 m, descriptor only | 49.72 % |
| R@1 with IDF-weighted text fusion (`w = 0.3`) | 49.26 % (−0.46 pts) |

The text branch is verified working, and it does **not** improve retrieval on this split: house
numbers are never recognised at this resolution, and the strings that are recognised are
storefront words repeating across a single neighbourhood. The notebook prints a fusion-weight
sweep showing monotone degradation (−8.65 pts at `w = 1.0`), so this is a property of the data,
not a tuning miss.

The notebook exposes the knobs as `TEXTINPLACE_SPOTTER_WEIGHTS`, `TEXTINPLACE_OCR_THRESHOLD` and
`TEXTINPLACE_OCR_INPUT_SIZE`, and prints a sanity check telling you whether the spotter in memory
is the pretrained one or a random tensor soup.

## 🔍 Single-photo inference

```bash
python infer_textinplace.py \
    --query-image "../Test pics/IMG_0733.png" \
    --weights-path weights/textinplace_finetuned.pth \
    --dataset_root "/media/rayan/usb/VPR Dataset/paris/gsv_cities"
```

Builds the database (first image per `place_id`), extracts descriptors and texts, prints the top-5 with similarity, coordinates, Maps link and detected text, then writes a comparison sheet.

## ⚠️ Checkpoint loading

Evaluation and inference normalise the `state_dict` before loading:

1. unwrap `model_state_dict` if the checkpoint is a training dictionary;
2. strip the `module.` prefix left by `DataParallel`;
3. re-prefix `dptext_detr.` / `recognizer.` / `bridge.` keys to `backbone.textmodel.*`;
4. `load_state_dict(strict=True)`, with an automatic fallback to `strict=False` — **a "Strict loading failed" message means partially loaded weights, and therefore silently wrong results**.

## 🔗 Use inside the pipeline

`vpr_pipeline` can use TextInPlace as its scene-text extractor (`--weights-path-textinplace`), falling back to EasyOCR when `repo/` or the weights are missing. See [vpr_pipeline/README.md](../vpr_pipeline/README.md).
