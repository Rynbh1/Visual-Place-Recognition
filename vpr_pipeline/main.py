"""
VPR Pipeline — CLI entry point.

Usage:
    # Evaluate Recall@K on a GSV-Cities structured dataset:
    python -m vpr_pipeline eval \
        --dataset-path datasets/paris_75018 \
        --weights-path MegaLoc/Results/trainings/megaloc_finetuned_paris.pth \
        --top-k 10

    # Single-image inference with late fusion:
    python -m vpr_pipeline infer \
        --image-path query.jpg \
        --dataset-path datasets/paris_75018 \
        --weights-path MegaLoc/Results/trainings/megaloc_finetuned_paris.pth \
        --use-ocr


Dataset format (GSV-Cities layout):
    <dataset-path>/
      Dataframes/
        Paris75018.csv          ← overall metadata/manifest
        Paris75018_train.csv    ← database references (first row per place)
        Paris75018_test.csv     ← test queries/database
      Images/
        Paris75018/
          Paris75018_0000023_...jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import torch

# Allow running this file directly (`python vpr_pipeline/main.py ...`) in
# addition to `python -m vpr_pipeline`: without a parent package, the relative
# imports below would fail.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "vpr_pipeline"


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------



def _load_entries_from_csv(
    csv_path: Path,
    images_dir: Path,
) -> tuple[list[dict], list[dict]]:
    """Parse a GSV-Cities style CSV layout into (db_entries, query_entries).

    Convention:
        - The first row (deterministically sorted by year, month, northdeg, panoid)
          for each place_id is the database ref.
        - All remaining rows in that place_id group are treated as queries.

    Returns:
        db_entries: list of {path, place_id, lat, lon}
        query_entries: list of {path, place_id}
    """
    df = pd.read_csv(csv_path)
    # Ensure deterministic sort order (equivalent to alphabetical sorting of the filenames)
    df = df.sort_values(by=["place_id", "year", "month", "northdeg", "panoid"]).reset_index(drop=True)

    db_entries: list[dict] = []
    query_entries: list[dict] = []

    for place_id, group in df.groupby("place_id"):
        # First row -> database reference
        db_row = group.iloc[0]
        db_fname = (
            f"{db_row['city_id']}_{int(db_row['place_id']):07d}_"
            f"{int(db_row['year']):04d}_{int(db_row['month']):02d}_"
            f"{int(db_row['northdeg']):03d}_{db_row['lat']:.7f}_"
            f"{db_row['lon']:.7f}_{db_row['panoid']}.jpg"
        )
        db_img_path = images_dir / db_row["city_id"] / db_fname
        if db_img_path.exists():
            db_entries.append({
                "path": str(db_img_path),
                "place_id": str(place_id),
                "lat": float(db_row["lat"]),
                "lon": float(db_row["lon"]),
            })

            # Remaining rows -> queries
            for _, q_row in group.iloc[1:].iterrows():
                q_fname = (
                    f"{q_row['city_id']}_{int(q_row['place_id']):07d}_"
                    f"{int(q_row['year']):04d}_{int(q_row['month']):02d}_"
                    f"{int(q_row['northdeg']):03d}_{q_row['lat']:.7f}_"
                    f"{q_row['lon']:.7f}_{q_row['panoid']}.jpg"
                )
                q_img_path = images_dir / q_row["city_id"] / q_fname
                if q_img_path.exists():
                    query_entries.append({
                        "path": str(q_img_path),
                        "place_id": str(place_id),
                        "lat": float(q_row["lat"]),
                        "lon": float(q_row["lon"]),
                    })

    return db_entries, query_entries


def _load_db_entries_from_csv(
    csv_path: Path,
    images_dir: Path,
) -> list[dict]:
    """Build a database from a GSV-Cities style CSV (one reference per place_id).

    Takes the first row per place_id as the database reference image.
    """
    db_entries, _ = _load_entries_from_csv(csv_path, images_dir)
    return db_entries


# ---------------------------------------------------------------------------
# Recall@K metric
# ---------------------------------------------------------------------------

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    import math
    R = 6371000.0  # meters
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def compute_recall(
    predictions: list[list[dict]],
    query_entries: list[dict],
    k_values: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    """Compute Recall@K for a set of queries using place_id OR GPS distance <= 25m.

    Args:
        predictions: For each query, the ordered list of retrieved results
                     (dicts with keys: place_id, lat, lon).
        query_entries: List of query metadata dicts with keys: place_id, lat, lon.
        k_values: The K values to evaluate.

    Returns:
        Dict mapping "R@1", "R@5", "R@10" (etc.) to recall in [0, 1].
    """
    n = len(query_entries)
    counts = {k: 0 for k in k_values}

    for preds, q in zip(predictions, query_entries):
        q_lat, q_lon = q["lat"], q["lon"]
        q_place_id = q["place_id"]
        
        # Calculate correctness for each retrieved item in preds
        is_correct_top = []
        for r in preds:
            pred_place_id = r["place_id"]
            pred_lat, pred_lon = r["lat"], r["lon"]
            dist = haversine_distance(pred_lat, pred_lon, q_lat, q_lon)
            is_correct_top.append((pred_place_id == q_place_id) or (dist <= 25.0))
            
        for k in k_values:
            if any(is_correct_top[:k]):
                counts[k] += 1

    return {f"R@{k}": counts[k] / n for k in k_values}



# ---------------------------------------------------------------------------
# Sub-command: eval
# ---------------------------------------------------------------------------

def cmd_eval(args: argparse.Namespace) -> None:
    """Evaluate Recall@1/5/10 on a GSV-Cities structured dataset."""
    from retrieval import MegaLocRetriever

    dataset_path = Path(args.dataset_path)
    weights_path = Path(args.weights_path_megaloc)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[eval] Device: {device}")

    # ── Load dataset ──────────────────────────────────────────────────
    print(f"[eval] Scanning dataset: {dataset_path}")
    if getattr(args, "csv_path", None):
        csv_path = Path(args.csv_path)
    else:
        df_dir = dataset_path / "Dataframes"
        if not df_dir.exists():
            raise FileNotFoundError(f"Dataframes folder not found in: {dataset_path}")
        csv_files = list(df_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in: {df_dir}")
        test_files = [f for f in csv_files if f.name.endswith("_test.csv")]
        csv_path = test_files[0] if test_files else csv_files[0]

    print(f"[eval] Using GSV-Cities CSV: {csv_path}")

    images_dir = dataset_path / "Images" if (dataset_path / "Images").exists() else dataset_path
    db_entries, query_entries = _load_entries_from_csv(csv_path, images_dir)
    print(f"[eval] Database: {len(db_entries)} places | Queries: {len(query_entries)}")

    if not query_entries:
        print("[eval] No queries found.")
        sys.exit(1)

    # ── Build retriever + index ───────────────────────────────────────
    retriever = MegaLocRetriever(
        weights_path=weights_path,
        device=device,
        use_fp16=not args.no_fp16,
    )

    index_cache: Optional[Path] = None
    if args.index_cache:
        index_cache = Path(args.index_cache)

    retriever.build_index(db_entries, index_cache=index_cache)

    # ── Run retrieval for every query ─────────────────────────────────
    k_max = max(args.top_k, 10)
    predictions: list[list[dict]] = []

    from tqdm import tqdm
    for q in tqdm(query_entries, desc="Evaluating queries", ncols=80):
        results = retriever.search(Path(q["path"]), k=k_max)
        predictions.append(results)

    # ── Compute and display Recall@K ──────────────────────────────────
    recalls = compute_recall(predictions, query_entries, k_values=(1, 5, 10))

    print("\n" + "=" * 50)
    print("  VPR Evaluation Results")
    print("=" * 50)
    print(f"  Dataset : {dataset_path.name}")
    print(f"  Queries : {len(query_entries)}")
    print(f"  DB size : {len(db_entries)}")
    print("-" * 50)
    for metric, value in recalls.items():
        print(f"  {metric:8s} = {value * 100:.2f} %")
    print("=" * 50)


# ---------------------------------------------------------------------------
# Sub-command: infer
# ---------------------------------------------------------------------------

def cmd_infer(args: argparse.Namespace) -> None:
    """Run the full hybrid pipeline on a single query image."""
    from .retrieval import MegaLocRetriever
    from .ocr import SceneTextExtractor, OllamaGeoFilter
    from .fusion import LateFusionOrchestrator

    image_path = Path(args.image_path)
    dataset_path = Path(args.dataset_path)
    weights_path = Path(args.weights_path_megaloc)

    if not image_path.exists():
        print(f"[infer] Error: image not found: {image_path}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[infer] Device: {device}")

    # ── Build database ────────────────────────────────────────────────
    if args.csv_path:
        csv_path = Path(args.csv_path)
    else:
        df_dir = dataset_path / "Dataframes"
        if not df_dir.exists():
            raise FileNotFoundError(f"Dataframes folder not found in: {dataset_path}")
        csv_files = list(df_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in: {df_dir}")
        train_files = [f for f in csv_files if f.name.endswith("_train.csv")]
        test_files = [f for f in csv_files if f.name.endswith("_test.csv")]
        full_files = [f for f in csv_files if f not in train_files and f not in test_files]
        # A single-photo "where was this taken" query should search the whole known
        # catalogue, not a training-only partition. train/test are a strict split of
        # the full CSV built for evaluation (holding out test data to measure
        # generalisation) — that split is meaningless for inference, and preferring
        # train here made real queries unfindable whenever their true place happened
        # to fall in the held-out test block (fixed after such a case: MegaLoc's own
        # infer_megaloc.py defaults to the test split instead, so the two tools were
        # silently searching two disjoint databases with no overlap between them).
        if full_files:
            csv_path = full_files[0]
        elif train_files:
            csv_path = train_files[0]
            print(f"[infer] Warning: no full (unsplit) CSV found in {df_dir} — falling back to "
                  f"the train split. Places held out in the test split will never be found.")
        elif test_files:
            csv_path = test_files[0]
        else:
            csv_path = csv_files[0]

    print(f"[infer] Using GSV-Cities CSV for database: {csv_path}")

    images_dir = dataset_path / "Images" if (dataset_path / "Images").exists() else dataset_path
    db_entries = _load_db_entries_from_csv(
        csv_path=csv_path,
        images_dir=images_dir,
    )

    print(f"[infer] Database: {len(db_entries)} entries")

    # ── Initialise visual retriever ───────────────────────────────────
    retriever = MegaLocRetriever(
        weights_path=weights_path,
        device=device,
        use_fp16=not args.no_fp16,
    )

    index_cache: Optional[Path] = Path(args.index_cache) if args.index_cache else None
    retriever.build_index(db_entries, index_cache=index_cache)

    # ── Initialise OCR + LLM branches ────────────────────────────────
    text_extractor = SceneTextExtractor(
        device=device,
        weights_path=args.weights_path_textinplace,
    )
    geo_filter = OllamaGeoFilter(
        model=args.ollama_model,
        base_url=args.ollama_url,
        zone_description=args.zone,
    )

    if args.use_ocr and not geo_filter.is_available():
        print(
            f"[infer] Warning: Ollama not reachable at {args.ollama_url}. "
            "Proceeding with visual-only mode (OCR disabled)."
        )
        args.use_ocr = False

    # ── Late fusion ───────────────────────────────────────────────────
    if args.use_ocr:
        orchestrator = LateFusionOrchestrator(
            retriever=retriever,
            text_extractor=text_extractor,
            geo_filter=geo_filter,
            base_alpha=args.alpha,
            top_k_visual=args.top_k,
            top_k_ocr=args.top_k_ocr,
        )
        candidates = orchestrator.run(image_path)
    else:
        # Visual-only: wrap raw retrieval results as RetrievalCandidate objects
        from .fusion import RetrievalCandidate
        raw = retriever.search(image_path, k=args.top_k)
        scores = [r["visual_score"] for r in raw]
        s_min, s_max = min(scores), max(scores)
        s_range = (s_max - s_min) if s_max != s_min else 1.0
        candidates = [
            RetrievalCandidate(
                rank=i,
                place_id=str(r["place_id"]),
                lat=r["lat"],
                lon=r["lon"],
                db_image_path=Path(r["path"]),
                visual_score=r["visual_score"],
                visual_score_norm=(r["visual_score"] - s_min) / s_range,
                fused_score=(r["visual_score"] - s_min) / s_range,
                alpha_used=1.0,
            )
            for i, r in enumerate(raw)
        ]

    # ── Geometric verification (RANSAC — conditional) ─────────────────
    if args.use_ransac and candidates:
        from .geometric_verifier import GeometricVerifier
        verifier = GeometricVerifier(
            device=device,
            logistic_w=args.ransac_logistic_w,
            logistic_b=args.ransac_logistic_b,
            min_confidence=args.ransac_min_confidence,
        )
        geom = verifier.verify(image_path, candidates)
        top = candidates[0]
        top.ransac_inliers = geom.inlier_count
        top.geom_confidence = geom.confidence
        if geom.triggered:
            top.geom_verified = not geom.kill_switch

    # ── Print results ─────────────────────────────────────────────────
    _print_results(image_path, candidates, use_ocr=args.use_ocr)

    if args.use_ransac and candidates and candidates[0].geom_verified is False:
        print(
            "\n[!] KILL-SWITCH activated — P(correct) below threshold.\n"
            "    The system abstains: no GPS coordinate is emitted.\n"
            "    (Reason: local geometric evidence insufficient to trust this prediction.)"
        )


def _print_results(
    query_path: Path,
    candidates: list,
    use_ocr: bool = False,
) -> None:
    """Pretty-print the ranked candidate list to stdout."""
    print("\n" + "=" * 80)
    print("  VPR Inference Results")
    print("=" * 80)
    print(f"  Query : {query_path}")
    if use_ocr and candidates and candidates[0].extracted_texts:
        print(f"  OCR   : {candidates[0].extracted_texts[:5]}")
    print("-" * 80)

    for c in candidates:
        marker = "★" if c.rank == 0 else f"#{c.rank + 1}"
        print(f"  {marker:3s}  Place: {c.place_id}")
        print(f"       Coords       : {c.lat:.6f}, {c.lon:.6f}")
        print(f"       Visual score : {c.visual_score:.4f}  (norm: {c.visual_score_norm:.3f})")
        if use_ocr:
            status = "coherent" if c.text_coherent else "not coherent"
            print(
                f"       Text branch  : {status}  "
                f"(conf={c.text_confidence:.2f}, α={c.alpha_used:.2f})"
            )
        print(f"       Fused score  : {c.fused_score:.4f}")
        if c.geom_verified is not None:
            gstatus = "VALIDATED" if c.geom_verified else "REJECTED [kill-switch]"
            print(
                f"       Geom verify  : {gstatus} "
                f"(inliers={c.ransac_inliers}, P={c.geom_confidence:.3f})"
            )
        print(f"       Maps         : https://maps.google.com/?q={c.lat:.6f},{c.lon:.6f}")
        print(f"       DB image     : {c.db_image_path.name}")
        print("-" * 80)

    print("=" * 80)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vpr_pipeline",
        description="Hybrid VPR pipeline: MegaLoc visual retrieval + OCR late fusion.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── Shared arguments ──────────────────────────────────────────────
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument(
        "--weights-path-megaloc", default="MegaLoc/Results/trainings/megaloc_finetuned_paris.pth",
        help="Path to the MegaLoc .pth checkpoint.",
    )
    shared.add_argument(
        "--weights-path-textinplace", default="TextInPlace/weights/textinplace_finetuned.pth",
        help="Path to the TextInPlace .pth checkpoint.",
    )
    shared.add_argument(
        "--no-fp16", action="store_true",
        help="Disable fp16 on GPU (use fp32 — doubles VRAM usage).",
    )
    shared.add_argument(
        "--cpu", action="store_true",
        help="Force CPU inference (slow, but useful for debugging).",
    )
    shared.add_argument(
        "--index-cache", default=None,
        help="Path to a .faiss cache file. Built on first run, loaded afterwards.",
    )
    shared.add_argument(
        "--top-k", type=int, default=10,
        help="Number of candidates to retrieve from FAISS.",
    )

    # ── eval sub-command ──────────────────────────────────────────────
    eval_p = subparsers.add_parser(
        "eval",
        parents=[shared],
        help="Evaluate Recall@K on a GSV-Cities dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    eval_p.add_argument(
        "--dataset-path", required=True,
        help="Root folder of the GSV-Cities dataset (containing Dataframes/ and Images/).",
    )
    eval_p.add_argument(
        "--csv-path", default=None,
        help="Optional GSV-Cities CSV path for evaluation (defaults to the _test.csv inside Dataframes/).",
    )
    eval_p.set_defaults(func=cmd_eval)

    # ── infer sub-command ─────────────────────────────────────────────
    infer_p = subparsers.add_parser(
        "infer",
        parents=[shared],
        help="Geolocate a single query image with optional OCR late fusion.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    infer_p.add_argument(
        "--image-path", required=True,
        help="Path to the query image.",
    )
    infer_p.add_argument(
        "--dataset-path", required=True,
        help="Root folder of the GSV-Cities dataset (containing Dataframes/ and Images/).",
    )
    infer_p.add_argument(
        "--csv-path", default=None,
        help="Optional GSV-Cities CSV path to build the database (defaults to the _train.csv inside Dataframes/).",
    )
    infer_p.add_argument(
        "--use-ocr", action="store_true",
        help="Enable the OCR + Ollama late-fusion branch.",
    )
    infer_p.add_argument(
        "--alpha", type=float, default=0.7,
        help="Base weight for the visual branch in fusion (0.0–1.0).",
    )
    infer_p.add_argument(
        "--top-k-ocr", type=int, default=5,
        help="Number of visual candidates to send to the LLM for validation.",
    )
    infer_p.add_argument(
        "--ollama-model", default="qwen3:8b",
        help="Ollama model tag for the geo-filter.",
    )
    infer_p.add_argument(
        "--ollama-url", default="http://localhost:11434",
        help="Ollama server base URL.",
    )
    infer_p.add_argument(
        "--zone", default="Paris, France",
        help="Human-readable zone description passed to the LLM prompt.",
    )
    infer_p.add_argument(
        "--use-ransac", action="store_true",
        help=(
            "Enable conditional RANSAC geometric verification (Sferrazza et al., CVPR 2025). "
            "Triggered only when late-fusion confidence is ambiguous. "
            "Never re-ranks candidates — only validates the top-1 prediction."
        ),
    )
    infer_p.add_argument(
        "--ransac-min-confidence", type=float, default=0.5,
        help="Kill-switch threshold: P(correct) below this → abstain (no GPS emitted).",
    )
    infer_p.add_argument(
        "--ransac-logistic-w", type=float, default=0.05,
        help="Logistic slope w in P(correct) = σ(w·i_q + b).",
    )
    infer_p.add_argument(
        "--ransac-logistic-b", type=float, default=-2.0,
        help="Logistic intercept b in P(correct) = σ(w·i_q + b). Default: ~40 inliers → 50%%.",
    )
    infer_p.set_defaults(func=cmd_infer)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
