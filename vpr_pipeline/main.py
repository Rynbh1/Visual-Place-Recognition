#!/usr/bin/env python3
"""
VPR & Text Spotting Orchestrator CLI.
Integrates MegaLoc visual retrieval with TextInPlace text spotting and Qwen late-fusion reranking.
Optimized for laptops with 8GB VRAM (NVIDIA RTX 4060).
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont


# Add project root directory to path to allow absolute imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vpr_pipeline.utils import clear_vram, load_reranker, late_fusion_rerank
from vpr_pipeline.retrieval import load_megaloc_model, extract_descriptors, build_faiss_index, search_index
from vpr_pipeline.ocr import load_ocr_model, spot_text_in_images


def parse_dataset(dataset_path: Path) -> Tuple[List[Path], List[Path], List[np.ndarray], List[str], List[str]]:
    """
    Parses the dataset directory.
    Supports two structures:
      1. Folder-based: Scanning dataset_path/places/place_XXXXXX/ for image splits.
      2. CSV-based: Fallback to reading Dataframes/*_test.csv and matching Images/.
      
    Returns:
        db_paths: paths to database (reference) images
        q_paths: paths to query images
        positives_per_query: list of numpy arrays containing database indices of positive matches for each query
        db_places: place labels for database images
        q_places: place labels for query images
    """
    places_dir = dataset_path / "places"
    
    # 1. Folder-based parsing
    if places_dir.exists() and places_dir.is_dir():
        print(f"[Dataset] Found 'places' folder at {places_dir}. Scanning place directories...")
        db_paths = []
        q_paths = []
        db_places = []
        q_places = []
        
        place_dirs = sorted([d for d in places_dir.iterdir() if d.is_dir() and d.name.startswith("place_")])
        for p_dir in place_dirs:
            place_id = p_dir.name
            images = sorted([img for img in p_dir.iterdir() if img.suffix.lower() in (".jpg", ".jpeg", ".png")])
            if not images:
                continue
            
            # First image goes to database, others to queries
            db_paths.append(images[0])
            db_places.append(place_id)
            for q_img in images[1:]:
                q_paths.append(q_img)
                q_places.append(place_id)
                
        # Map query places to their indices in the database
        positives_per_query = []
        for q_place in q_places:
            pos_indices = [idx for idx, db_place in enumerate(db_places) if db_place == q_place]
            positives_per_query.append(np.array(pos_indices))
            
        print(f"[Dataset] Loaded {len(db_paths)} database locations and {len(q_paths)} queries from directories.")
        return db_paths, q_paths, positives_per_query, db_places, q_places

    # 2. CSV-based parsing (Fallback)
    df_dir = dataset_path / "Dataframes"
    img_dir = dataset_path / "Images"
    if df_dir.exists() and img_dir.exists():
        print(f"[Dataset] places folder not found. Falling back to CSV search under {df_dir}...")
        csv_files = list(df_dir.glob("*_test.csv"))
        if not csv_files:
            csv_files = list(df_dir.glob("*.csv"))
        if csv_files:
            csv_path = csv_files[0]
            print(f"[Dataset] Loading metadata from {csv_path}...")
            df = pd.read_csv(csv_path)
            
            # Map values into dataset paths
            db_rows = []
            q_rows = []
            
            def get_img_name(row):
                city = row['city_id']
                pl_id = f"{row['place_id']:07d}"
                panoid = str(int(row['panoid'])) if isinstance(row['panoid'], float) and row['panoid'].is_integer() else str(row['panoid'])
                year = f"{row['year']:04d}"
                month = f"{row['month']:02d}"
                northdeg = f"{row['northdeg']:03d}"
                lat, lon = f"{row['lat']:.7f}", f"{row['lon']:.7f}"
                return f"{city}_{pl_id}_{year}_{month}_{northdeg}_{lat}_{lon}_{panoid}.jpg"
                
            for pid, group in df.groupby("place_id"):
                if len(group) >= 2:
                    db_rows.append(group.iloc[0])
                    for i in range(1, len(group)):
                        q_rows.append(group.iloc[i])
                else:
                    db_rows.append(group.iloc[0])
                    
            db_paths = [img_dir / r['city_id'] / get_img_name(r) for r in db_rows]
            q_paths = [img_dir / r['city_id'] / get_img_name(r) for r in q_rows]
            
            # Filter non-existent images
            db_exists = [p.exists() for p in db_paths]
            q_exists = [p.exists() for p in q_paths]
            
            db_paths = [p for p, exists in zip(db_paths, db_exists) if exists]
            db_places = [str(r['place_id']) for r, exists in zip(db_rows, db_exists) if exists]
            
            q_paths = [p for p, exists in zip(q_paths, q_exists) if exists]
            q_places = [str(r['place_id']) for r, exists in zip(q_rows, q_exists) if exists]
            
            # Map query places to their indices in the database
            positives_per_query = []
            for q_place in q_places:
                pos_indices = [idx for idx, db_place in enumerate(db_places) if db_place == q_place]
                positives_per_query.append(np.array(pos_indices))
                
            print(f"[Dataset] Loaded {len(db_paths)} database locations and {len(q_paths)} queries from CSV.")
            return db_paths, q_paths, positives_per_query, db_places, q_places
            
    raise FileNotFoundError(f"No place directories or CSV descriptors found at {dataset_path}")


def run_evaluation(args):
    """
    Executes VPR evaluation pipeline: retrieval, ocr text extraction, reranking, metrics.
    """
    dataset_path = Path(args.dataset_path)
    db_paths, q_paths, positives_per_query, db_places, q_places = parse_dataset(dataset_path)
    
    if len(q_paths) == 0:
        print("[Eval] Error: No query images found for evaluation.")
        return
        
    # ==========================================
    # 1. MegaLoc Visual Retrieval
    # ==========================================
    print("\n" + "=" * 60)
    print(" 1. VISUAL RETRIEVAL (MegaLoc)")
    print("=" * 60)
    megaloc_model, device = load_megaloc_model(args.megaloc_weights)
    
    print("[MegaLoc] Extracting database descriptors...")
    db_descriptors = extract_descriptors(megaloc_model, db_paths, device, batch_size=8)
    
    print("[MegaLoc] Extracting query descriptors...")
    q_descriptors = extract_descriptors(megaloc_model, q_paths, device, batch_size=8)
    
    # Unload MegaLoc from VRAM
    clear_vram(megaloc_model)
    print("[MegaLoc] Unloaded. VRAM cleared.")
    
    # Build FAISS index
    index = build_faiss_index(db_descriptors)
    print("[FAISS] Searching visual candidates...")
    distances, predictions = search_index(index, q_descriptors, top_k=args.top_k)
    
    # Calculate recalls before text reranking
    recalls_visual = np.zeros(3)  # Recall@1, Recall@5, Recall@10
    recall_values = [1, 5, 10]
    for query_index, pred in enumerate(predictions):
        for i, n in enumerate(recall_values):
            if np.any(np.in1d(pred[:n], positives_per_query[query_index])):
                recalls_visual[i:] += 1
                break
    recalls_visual = recalls_visual / len(q_paths) * 100
    
    print("\nVisual recalls (MegaLoc Only):")
    print(f"  R@1: {recalls_visual[0]:.2f}% | R@5: {recalls_visual[1]:.2f}% | R@10: {recalls_visual[2]:.2f}%")
    
    if args.disable_reranking:
        print("[Eval] Text-based reranking is disabled. Stopping evaluation.")
        return
        
    # ==========================================
    # 2. Text Spotting (TextInPlace)
    # ==========================================
    print("\n" + "=" * 60)
    print(" 2. TEXT SPOTTING (TextInPlace)")
    print("=" * 60)
    ocr_model, device = load_ocr_model(args.textinplace_config, args.textinplace_weights)
    
    print("[OCR] Extracting text from query images...")
    query_texts = spot_text_in_images(ocr_model, q_paths, device)
    
    # Identify unique database candidates to only run OCR on visual matches (Optimization)
    unique_candidates = sorted(list(set(predictions.flatten())))
    candidate_paths = [db_paths[idx] for idx in unique_candidates]
    
    print(f"[OCR] Extracting text from {len(candidate_paths)} candidate database images (out of {len(db_paths)} total)...")
    candidate_texts_subset = spot_text_in_images(ocr_model, candidate_paths, device)
    
    # Map back subset text list to full database list
    db_texts = [[] for _ in range(len(db_paths))]
    for subset_idx, db_idx in enumerate(unique_candidates):
        db_texts[db_idx] = candidate_texts_subset[subset_idx]
        
    # Unload TextInPlace from VRAM
    clear_vram(ocr_model)
    print("[OCR] Unloaded. VRAM cleared.")
    
    # ==========================================
    # 3. Late Fusion Reranking (Qwen)
    # ==========================================
    print("\n" + "=" * 60)
    print(" 3. LATE FUSION RE-RANKING (Qwen)")
    print("=" * 60)
    reranker = load_reranker(args.qwen_model)
    
    reranked_predictions = []
    print("[Reranker] Reranking query candidate lists...")
    for q_idx, pred in enumerate(predictions):
        reranked = late_fusion_rerank(reranker, pred, query_texts[q_idx], db_texts, top_k=args.top_k)
        reranked_predictions.append(reranked)
        
    reranked_predictions = np.array(reranked_predictions)
    
    # Unload Reranker from VRAM
    clear_vram(reranker)
    print("[Reranker] Unloaded. VRAM cleared.")
    
    # Calculate recalls after text reranking
    recalls_fused = np.zeros(3)  # Recall@1, Recall@5, Recall@10
    for query_index, pred in enumerate(reranked_predictions):
        for i, n in enumerate(recall_values):
            if np.any(np.in1d(pred[:n], positives_per_query[query_index])):
                recalls_fused[i:] += 1
                break
    recalls_fused = recalls_fused / len(q_paths) * 100
    
    print("\n" + "=" * 60)
    print("                 FINAL EVALUATION SUMMARY")
    print("=" * 60)
    print(f" Dataset       : {dataset_path.name}")
    print(f" Queries       : {len(q_paths)}")
    print(f" Database Size : {len(db_paths)}")
    print("-" * 60)
    print(f" MegaLoc Only   - R@1: {recalls_visual[0]:.2f}% | R@5: {recalls_visual[1]:.2f}% | R@10: {recalls_visual[2]:.2f}%")
    print(f" MegaLoc + Qwen - R@1: {recalls_fused[0]:.2f}% | R@5: {recalls_fused[1]:.2f}% | R@10: {recalls_fused[2]:.2f}%")
    print("=" * 60)


def run_inference(args):
    """
    Executes visual place recognition inference on a single query photo.
    """
    query_path = Path(args.image_path)
    if not query_path.exists():
        raise FileNotFoundError(f"Query image not found at: {query_path}")
        
    dataset_path = Path(args.dataset_path)
    db_paths, _, _, db_places, _ = parse_dataset(dataset_path)
    
    # ==========================================
    # 1. MegaLoc Visual Retrieval
    # ==========================================
    print(f"\n[MegaLoc] Running visual descriptor extraction on {query_path.name}...")
    megaloc_model, device = load_megaloc_model(args.megaloc_weights)
    
    q_desc = extract_descriptors(megaloc_model, [query_path], device, batch_size=1)
    
    # Extract reference database descriptors (in real systems, loaded from cache)
    print("[MegaLoc] Extracting database descriptors...")
    db_descriptors = extract_descriptors(megaloc_model, db_paths, device, batch_size=8)
    
    clear_vram(megaloc_model)
    
    # Search candidates
    index = build_faiss_index(db_descriptors)
    distances, predictions = search_index(index, q_desc, top_k=args.top_k)
    prediction = predictions[0]
    distance = distances[0]
    
    # ==========================================
    # 2. Text Spotting (TextInPlace)
    # ==========================================
    print(f"\n[OCR] Spotting text in query {query_path.name}...")
    ocr_model, device = load_ocr_model(args.textinplace_config, args.textinplace_weights)
    
    query_text = spot_text_in_images(ocr_model, [query_path], device)[0]
    
    cand_paths = [db_paths[idx] for idx in prediction]
    print(f"[OCR] Spotting text in the top {args.top_k} visual database candidates...")
    candidate_texts = spot_text_in_images(ocr_model, cand_paths, device)
    
    db_texts = [[] for _ in range(len(db_paths))]
    for subset_idx, db_idx in enumerate(prediction):
        db_texts[db_idx] = candidate_texts[subset_idx]
        
    clear_vram(ocr_model)
    
    # ==========================================
    # 3. Late Fusion Reranking (Qwen)
    # ==========================================
    print(f"\n[Reranker] Query text detected: {query_text}")
    reranker = load_reranker(args.qwen_model)
    reranked = late_fusion_rerank(reranker, prediction, query_text, db_texts, top_k=args.top_k)
    clear_vram(reranker)
    
    print("\n" + "=" * 80)
    print("                 SINGLE IMAGE INFERENCE MATCHES (TOP 5)")
    print("=" * 80)
    print(f" Query Image  : {query_path.name}")
    print(f" Query Text   : {query_text}")
    print("-" * 80)
    
    for rank in range(5):
        if rank >= len(reranked):
            break
        idx = reranked[rank]
        path = db_paths[idx]
        place_id = db_places[idx]
        text_spotted = db_texts[idx]
        
        # Parse metadata from filename
        parts = path.stem.split("_")
        try:
            # Format: {city_id}_{place_id:07d}_{year:04d}_{month:02d}_{north:03d}_{lat:.7f}_{lon:.7f}_{panoid}
            lat = float(parts[-3])
            lon = float(parts[-2])
            coords_str = f"Lat: {lat:.7f}, Lon: {lon:.7f}"
            gmaps = f"https://www.google.com/maps/search/?api=1&query={lat},{lon}"
        except Exception:
            coords_str = "Unknown coordinates"
            gmaps = "N/A"
            
        # Check if index is in original predictions to get visual distance
        visual_rank = np.where(prediction == idx)[0]
        if len(visual_rank) > 0:
            v_dist = distance[visual_rank[0]]
            v_info = f"Visual Dist: {v_dist:.4f} (Rank #{visual_rank[0] + 1})"
        else:
            v_info = "Visual Rank: >20"
            
        print(f" #{rank+1} - Place ID: {place_id} | {path.name}")
        print(f"     {coords_str} | {v_info}")
        print(f"     Text Detected: {text_spotted}")
        print(f"     Google Maps: {gmaps}")
        print("-" * 80)
    
    # Save a comparison comparative matched image panel containing the top 5 candidates
    target_size = (250, 250)
    spacing = 15
    combined_img = Image.new("RGB", (6 * target_size[0] + 7 * spacing, target_size[1] + 130), "white")
    draw = ImageDraw.Draw(combined_img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
        
    def add_panel(img_path, title, x_offset, border_color, subtitle="", text_spotted=[]):
        img_obj = Image.open(img_path).convert("RGB").resize(target_size)
        bordered = Image.new("RGB", (target_size[0] + 6, target_size[1] + 6), border_color)
        bordered.paste(img_obj, (3, 3))
        combined_img.paste(bordered, (x_offset, spacing))
        
        draw.text((x_offset + 5, spacing + target_size[1] + 10), title, fill="black", font=font)
        # Handle newlines in subtitle
        y_cursor = spacing + target_size[1] + 25
        for line in subtitle.split('\n'):
            draw.text((x_offset + 5, y_cursor), line, fill="gray", font=font)
            y_cursor += 15
        draw.text((x_offset + 5, y_cursor + 5), f"Text: {text_spotted}", fill="blue", font=font)

    # Draw query panel
    add_panel(query_path, "QUERY IMAGE", spacing, (0, 102, 204), text_spotted=query_text)
    
    # Draw top 5 panels
    for rank in range(5):
        if rank >= len(reranked):
            break
        idx = reranked[rank]
        path = db_paths[idx]
        place_id = db_places[idx]
        text_spotted = db_texts[idx]
        
        parts = path.stem.split("_")
        try:
            lat = float(parts[-3])
            lon = float(parts[-2])
            subtitle = f"Place ID: {place_id}\nLat: {lat:.5f}, Lon: {lon:.5f}"
        except Exception:
            subtitle = f"Place ID: {place_id}"
            
        x_offset = (rank + 1) * target_size[0] + (rank + 2) * spacing
        # Use green border for the best match, and light grey for others
        border_color = (46, 204, 113) if rank == 0 else (189, 195, 199)
        add_panel(path, f"CANDIDATE #{rank+1}", x_offset, border_color, subtitle=subtitle, text_spotted=text_spotted)
              
    out_path = Path(args.output_dir) / f"{query_path.stem}_vpr_match.png"
    combined_img.save(out_path)
    print(f"[Inference] Comparison panel saved to: {out_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="MegaLoc + TextInPlace + Qwen Late Fusion VPR Pipeline")
    subparsers = parser.add_subparsers(dest="mode", required=True, help="Modes: eval (evaluate dataset) or infer (query single image)")
    
    # Common arguments
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument("--megaloc-weights", type=str, default="MegaLoc/megaloc_finetuned_paris.pth", help="Path to MegaLoc model weights")
    common_parser.add_argument("--textinplace-config", type=str, default="TextInPlace/repo/configs/Bridge/TotalText/R_50_poly.yaml", help="Path to TextInPlace model configuration")
    common_parser.add_argument("--textinplace-weights", type=str, default="TextInPlace/repo/checkpoints/Bridge_tt.pth", help="Path to TextInPlace weights")
    common_parser.add_argument("--qwen-model", type=str, default="Qwen/Qwen3-Reranker-0.6B", help="Model name of local Qwen reranker")
    common_parser.add_argument("--top-k", type=int, default=20, help="Number of database candidates to query and rerank")
    
    # Subparser: eval
    eval_parser = subparsers.add_parser("eval", parents=[common_parser], help="Evaluate visual place recognition recalls on a complete dataset folder")
    eval_parser.add_argument("--dataset-path", type=str, required=True, help="Path to the dataset directory (e.g. datasets/paris_75019)")
    eval_parser.add_argument("--disable-reranking", action="store_true", help="Calculate visual-only recalls without text reranking")
    eval_parser.set_defaults(func=run_evaluation)
    
    # Subparser: infer
    infer_parser = subparsers.add_parser("infer", parents=[common_parser], help="Find best matches for a single query image")
    infer_parser.add_argument("--image-path", type=str, required=True, help="Path to the query image file")
    infer_parser.add_argument("--dataset-path", type=str, required=True, help="Path to the reference database directory")
    infer_parser.add_argument("--output-dir", type=str, default=".", help="Directory to save the match visualization panel")
    infer_parser.set_defaults(func=run_inference)
    
    args = parser.parse_args()
    
    # Explicitly check CUDA availability and log device
    if torch.cuda.is_available():
        print(f"[Device] CUDA initialized. Device name: {torch.cuda.get_device_name(0)}")
        print(f"[Device] VRAM Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    else:
        print("[Device] CUDA is not available. Using CPU instead.")
        
    try:
        args.func(args)
    except Exception as e:
        print(f"[Error] Execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
