# 🔗 vpr_pipeline — Unified VPR pipeline

Hybrid CLI pipeline combining **MegaLoc** (global visual retrieval + FAISS index), **TextInPlace/EasyOCR** (scene text), **Ollama/Qwen3** (LLM geographic-coherence validation) and **LightGlue/RANSAC** (conditional geometric verification with a kill-switch).

Two sub-commands: `eval` (Recall@1/5/10 over a dataset) and `infer` (geolocate one photo, or abstain).

## 📁 Structure

```
vpr_pipeline/
├── __init__.py              # "Hybrid VPR pipeline: MegaLoc visual retrieval + OCR late fusion."
├── main.py                  # CLI entry point (eval / infer) + Recall@K + result printing
├── retrieval.py             # MegaLocRetriever — descriptor extraction + FAISS IndexFlatIP
├── ocr.py                   # SceneTextExtractor (TextInPlace or EasyOCR) + OllamaGeoFilter
├── fusion.py                # LateFusionOrchestrator + RetrievalCandidate
└── geometric_verifier.py    # GeometricVerifier — conditional RANSAC + kill-switch
```

## 🏗️ Architecture

```
┌──────────────┐
│  Query Image │
└──────┬───────┘
       ├──────────────────────────────────────────┐
       ▼                                          ▼
┌──────────────┐                          ┌───────────────┐
│   MegaLoc    │                          │ TextInPlace / │
│  Retriever   │                          │   EasyOCR     │
│ (FAISS Top-K)│                          │ (Scene Text)  │
└──────┬───────┘                          └───────┬───────┘
       │  visual_score (cosine)                   │  OCR texts
       │                                          ▼
       │                                  ┌───────────────┐
       │                                  │  Ollama/Qwen3 │
       │                                  │  (Geo Filter) │
       │                                  └───────┬───────┘
       │  visual_score_norm                       │  (coherent, confidence)
       ▼                                          ▼
┌─────────────────────────────────────────────────────────┐
│                Late Fusion Orchestrator                  │
│   fused = α · visual_norm + (1 − α) · text_confidence    │
│   dynamic α: 1.0 (no text) → 0.3 (confident LLM)         │
└──────────────────────────┬──────────────────────────────┘
                           ▼
                  ┌────────────────┐
                  │  Ambiguous?    │──── No ───► Emit GPS
                  └────────┬───────┘
                           │ Yes
                           ▼
                  ┌────────────────┐
                  │  RANSAC        │
                  │  (LightGlue)   │
                  │  P(correct)    │
                  └────────┬───────┘
                    ┌──────┴──────┐
                P ≥ 0.5       P < 0.5
                    ▼             ▼
                Emit GPS    Kill-switch (abstain)
```

## ⚠️ External dependencies

This module **depends on the other two modules** of the project.

### 1. MegaLoc weights *(always required)*

```
MegaLoc/Results/trainings/megaloc_finetuned_paris.pth   (~914 MB)
```

→ see [MegaLoc/README.md](../MegaLoc/README.md).

### 2. TextInPlace code + weights *(if `--use-ocr`)*

```
TextInPlace/repo/     ← ⛔ must be cloned: git clone https://github.com/HqiTao/TextInPlace.git repo
TextInPlace/weights/textinplace_finetuned.pth   (~440 MB)
```

→ see [TextInPlace/README.md](../TextInPlace/README.md).

### 3. Ollama *(if `--use-ocr`)*

```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull qwen3:8b                      # ~5 GB RAM
curl http://localhost:11434/api/tags      # check the server responds
```

If Ollama is unreachable, `infer` automatically falls back to visual-only mode with a warning.

### 4. LightGlue *(if `--use-ransac`)*

```bash
pip install lightglue
```

### 5. GSV-Cities dataset

```
<dataset-path>/
├── Dataframes/
│   ├── Paris.csv
│   ├── Paris_train.csv     ← reference database for `infer`
│   └── Paris_test.csv      ← queries + database for `eval`
└── Images/
    └── Paris/*.jpg
```

### 6. Python packages

```bash
pip install torch torchvision faiss-cpu pandas tqdm pillow requests
pip install easyocr      # OCR fallback (optional)
pip install lightglue    # geometric verification (optional)
```

## 🚀 Usage

> ℹ️ `infer` uses relative imports and must therefore be launched **as a module**; `eval` imports `retrieval` absolutely and must be launched **from inside the `vpr_pipeline/` folder**. See [Known limitations](#-known-limitations).

### Recall@K evaluation

```bash
cd vpr_pipeline
python main.py eval \
    --dataset-path "/media/rayan/usb/VPR Dataset/paris/gsv_cities" \
    --weights-path-megaloc ../MegaLoc/Results/trainings/megaloc_finetuned_paris.pth \
    --index-cache /tmp/paris.faiss \
    --top-k 10
```

### Full inference (visual + OCR + RANSAC)

```bash
python -c "from vpr_pipeline.main import main; main()" infer \
    --image-path "Test pics/IMG_0733.png" \
    --dataset-path "/media/rayan/usb/VPR Dataset/paris/gsv_cities" \
    --weights-path-megaloc MegaLoc/Results/trainings/megaloc_finetuned_paris.pth \
    --weights-path-textinplace TextInPlace/weights/textinplace_finetuned.pth \
    --use-ocr --use-ransac \
    --zone "Paris, France" --ollama-model qwen3:8b
```

### Visual-only inference

```bash
python -c "from vpr_pipeline.main import main; main()" infer \
    --image-path "Test pics/IMG_0733.png" \
    --dataset-path "/media/rayan/usb/VPR Dataset/paris/gsv_cities"
```

## ⚙️ Options

### Shared by both sub-commands

| Option | Default | Purpose |
|---|---|---|
| `--dataset-path` | *(required)* | GSV-Cities root (`Dataframes/` + `Images/`) |
| `--csv-path` | auto | Explicit CSV (`eval` → `_test.csv`, `infer` → `_train.csv`) |
| `--weights-path-megaloc` | `MegaLoc/Results/trainings/megaloc_finetuned_paris.pth` | MegaLoc checkpoint |
| `--weights-path-textinplace` | `TextInPlace/weights/textinplace_finetuned.pth` | TextInPlace checkpoint |
| `--top-k` | `10` | Number of candidates returned by FAISS |
| `--index-cache` | – | `.faiss` file: built on first run, reloaded afterwards |
| `--no-fp16` | off | Force fp32 on GPU (doubles VRAM) |
| `--cpu` | off | CPU inference (slow, useful for debugging) |

### `infer` only

| Option | Default | Purpose |
|---|---|---|
| `--image-path` | *(required)* | Query photo |
| `--use-ocr` | off | Enable the text + LLM branch |
| `--alpha` | `0.7` | Base weight of the visual branch |
| `--top-k-ocr` | `5` | Candidates sent to the LLM (cost control) |
| `--ollama-model` / `--ollama-url` | `qwen3:8b` / `http://localhost:11434` | LLM server |
| `--zone` | `Paris, France` | Zone injected into the prompt |
| `--use-ransac` | off | Enable conditional geometric verification |
| `--ransac-min-confidence` | `0.5` | Kill-switch threshold |
| `--ransac-logistic-w` / `-b` | `0.05` / `-2.0` | Parameters of `P(correct) = σ(w·i + b)` (~40 inliers ≈ 50%) |

## 🧩 How each stage works

### Retrieval — `retrieval.py`

`IndexFlatIP` over L2-normalised MegaLoc descriptors (dot product = exact cosine similarity, no approximation). The model is **lazily** loaded on the first encode, in fp16 by default; `free_vram()` releases the GPU allocation between stages. With `--index-cache`, the index is serialised to disk and reloaded as-is on subsequent runs.

> ⚠️ The FAISS cache stores no fingerprint of the dataset or the weights: after re-training or changing the database, **delete the `.faiss` file**, otherwise indexed and queried descriptors no longer match.

### Text branch — `ocr.py`

`SceneTextExtractor` loads TextInPlace (`STVGLNet_test`, 320×320 input, VPR dimension auto-detected from the checkpoint) or, absent weights, EasyOCR (`fr`+`en`, CRAFT + CRNN, ~700 MB VRAM). Tokens shorter than 2 characters and low-confidence detections are dropped to limit the noise sent to the LLM.

`OllamaGeoFilter` queries a local LLM: for each candidate it asks whether the detected texts are geographically plausible at those coordinates, requiring a JSON reply `{"coherent": bool, "confidence": float, "reason": str}`. `is_available()` probes the server before use.

### Late fusion — `fusion.py`

```
fused_score = α · visual_score_norm + (1 − α) · text_confidence
α = 1.0                                   if no text detected
α = max(0.3, base_alpha · (1 − 0.5·conf)) otherwise
```

Visual scores are min–max normalised **within the candidate set**. The 0.3 floor guarantees the visual branch is never ignored, and the LLM is only called on the first `--top-k-ocr` candidates.

### Geometric verification — `geometric_verifier.py`

Following *To Match or Not to Match: Revisiting Image Matching for Reliable Visual Place Recognition* (CVPR 2025). Systematic RANSAC re-ranking **degrades** modern retrievers: in adverse conditions (night, rain, seasonal change) local gradients are destroyed, RANSAC finds more inliers on an accidental texture match, and demotes the correct candidate the global model had ranked first.

RANSAC therefore becomes an **uncertainty estimator, never a re-ranker**. It fires only when:

1. **quantitative doubt** — top-1 fused score < `0.55`;
2. **perceptual aliasing** — top-1/top-2 gap < `0.08` **and** the two candidates at least 100 m apart (architectural "false twins").

It then yields `P(correct) = σ(w · inliers + b)` (SuperPoint 1024 keypoints + LightGlue). If `P < --ransac-min-confidence`, the **kill-switch** fires: no coordinate is emitted. Silence is preferred to a confidently wrong answer.

## 📏 Metrics

- **Recall@K** — fraction of queries whose top-K contains at least one correct candidate: same `place_id` **or** haversine distance ≤ **25 m** (hard-coded in `compute_recall`, stricter than the 100 m default of `test_megaloc.py` — the two numbers are therefore not directly comparable).
- **`eval` protocol** — for each `place_id`, the first row (sorted by `year, month, northdeg, panoid`) becomes the reference and the rest become queries.

## 🚧 Known limitations

- **`python -m vpr_pipeline` does not work**: the package has no `__main__.py`. Adding one containing `from .main import main; main()` would make the command documented at the top of `main.py` valid.
- **`eval` and `infer` are not launched the same way**: `cmd_eval` does `from retrieval import MegaLocRetriever` (absolute import → requires cwd `vpr_pipeline/`) while `cmd_infer` does `from .retrieval import ...` (relative import → requires package context). Making `cmd_eval`'s imports relative would fix both points at once.
- **The EasyOCR fallback is unreachable from the CLI**: `--weights-path-textinplace` has a non-null default, and `SceneTextExtractor.extract()` picks TextInPlace as soon as `weights_path is not None`. EasyOCR only kicks in when the extractor is instantiated from Python with `weights_path=None`.
- `--use-ransac` annotates the **top-1** candidate only; the others keep `geom_verified = None`.

## 📚 References

- **MegaLoc** — [arXiv:2502.17237](https://arxiv.org/abs/2502.17237)
- **TextInPlace** — [GitHub](https://github.com/HqiTao/TextInPlace)
- **To Match or Not to Match** (CVPR 2025) — conditional RANSAC
- **LightGlue** — [GitHub](https://github.com/cvg/LightGlue)
- **Ollama** — [ollama.ai](https://ollama.ai/)
