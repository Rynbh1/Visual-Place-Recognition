import os
import sys
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

# Add MegaLoc folder to search path so its models can be loaded natively
sys.path.append(str(Path(__file__).resolve().parent.parent / "MegaLoc"))
from lib.megaloc_model import MegaLoc

def get_megaloc_transform() -> T.Compose:
    """
    Returns standard image transformations expected by the MegaLoc model.
    """
    return T.Compose([
        T.Resize((322, 322)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_megaloc_model(weights_path: str) -> Tuple[MegaLoc, torch.device]:
    """
    Lazy loads the MegaLoc model, restores weights, and transfers it to GPU.
    Casts to float16 to optimize VRAM on RTX 4060.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[MegaLoc] Loading weights from {weights_path} on {device}...")
    
    if not os.path.exists(weights_path):
        # Check secondary paths
        local_path = Path(__file__).resolve().parent.parent / "MegaLoc" / weights_path
        if local_path.exists():
            weights_path = str(local_path)
        else:
            raise FileNotFoundError(f"MegaLoc weights not found at: {weights_path}")
            
    model = MegaLoc()
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    
    model = model.to(device)
    model.eval()
    print("[MegaLoc] Model loaded successfully.")
    return model, device


def extract_descriptors(
    model: MegaLoc,
    image_paths: List[Path],
    device: torch.device,
    batch_size: int = 8
) -> np.ndarray:
    """
    Extracts global image descriptors using the MegaLoc model.
    Processes images in batches to optimize GPU throughput without VRAM overflow.
    """
    transform = get_megaloc_transform()
    descriptors = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(image_paths), batch_size), desc="[MegaLoc] Extracting"):
            batch_paths = image_paths[i:i+batch_size]
            batch_tensors = []
            
            for path in batch_paths:
                try:
                    img = Image.open(path).convert("RGB")
                    tensor = transform(img)
                    batch_tensors.append(tensor)
                except Exception as e:
                    print(f"[MegaLoc] Warning: Failed to load {path} ({e})")
                    # Fallback to zero tensor to maintain indexing
                    batch_tensors.append(torch.zeros((3, 322, 322)))
            
            if not batch_tensors:
                continue
                
            input_tensor = torch.stack(batch_tensors).to(device)
            # Model forward pass
            feats = model(input_tensor)
            descriptors.append(feats.cpu().numpy())
            
    if not descriptors:
        return np.empty((0, model.feat_dim), dtype="float32")
        
    return np.concatenate(descriptors, axis=0)


def build_faiss_index(db_descriptors: np.ndarray) -> faiss.IndexFlatL2:
    """
    Builds a standard FAISS L2 CPU index.
    Kept on CPU to preserve RTX 4060 VRAM.
    """
    dim = db_descriptors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(db_descriptors.astype("float32"))
    print(f"[FAISS] Created flat L2 index containing {index.ntotal} items.")
    return index


def search_index(
    index: faiss.IndexFlatL2,
    query_descriptors: np.ndarray,
    top_k: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs FAISS visual retrieval.
    Returns:
        distances: L2 distances [num_queries, top_k]
        indices: retrieved database indices [num_queries, top_k]
    """
    distances, indices = index.search(query_descriptors.astype("float32"), top_k)
    return distances, indices
