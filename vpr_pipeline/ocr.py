import os
import string
import sys
from pathlib import Path
from typing import List, Tuple

import torch
import torchvision.transforms as transforms
from PIL import Image

# Add TextInPlace repo to module search path at index 0 to avoid utils conflict
repo_dir = str(Path(__file__).resolve().parent.parent / "TextInPlace" / "repo")
sys.path.insert(0, repo_dir)

from backbone import setup_cfg
from network import STVGLNet_test
from utils import util

# Remove to clean up search path
sys.path.remove(repo_dir)


# Vocabulary mapping for character decoding
voc = list(string.printable[:-6])


class ArgsNamespace:
    """
    Dummy arguments wrapper matching setup_cfg parameter requirements.
    """
    def __init__(self, config_file: str, weights_path: str):
        self.config_file = config_file
        self.opts = []
        self.confidence_threshold = 0.3
        self.features_dim = 16384
        self.resume = weights_path


def get_ocr_transform() -> transforms.Compose:
    """
    Returns standard image transformations expected by TextInPlace Backbone.
    Normalized according to ImageNet standard parameters.
    """
    IMAGENET_MEAN_STD = {
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    }
    return transforms.Compose([
        transforms.Resize((320, 320), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN_STD['mean'], std=IMAGENET_MEAN_STD['std']),
    ])


def rec_decode(rec) -> str:
    """
    Decodes the model's raw character index sequence into a string.
    """
    s = ''
    for c in rec:
        c = int(c)
        if c < len(voc):
            s += voc[c]
        elif c == len(voc):
            return s
        else:
            s += u''
    return s


def load_ocr_model(config_file: str, weights_path: str) -> Tuple[STVGLNet_test, torch.device]:
    """
    Lazy loads the TextInPlace text spotter model on the device.
    Weights are restored from weights_path.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[OCR] Loading TextInPlace model with weights from {weights_path} on {device}...")
    
    if not os.path.exists(config_file):
        config_path = repo_dir / config_file
        if config_path.exists():
            config_file = str(config_path)
        else:
            raise FileNotFoundError(f"TextInPlace config not found at: {config_file}")
            
    if not os.path.exists(weights_path):
        weights_path_chk = repo_dir / weights_path
        if weights_path_chk.exists():
            weights_path = str(weights_path_chk)
        else:
            raise FileNotFoundError(f"TextInPlace weights not found at: {weights_path}")
            
    args = ArgsNamespace(config_file, weights_path)
    cfg = setup_cfg(args)
    
    model = STVGLNet_test(cfg)
    model = model.to(device)
    model = util.resume_model(args, model)
    model.eval()
    
    print("[OCR] TextInPlace model loaded successfully.")
    return model, device


def spot_text_in_images(model: STVGLNet_test, image_paths: List[Path], device: torch.device) -> List[List[str]]:
    """
    Runs text spotting on a list of image paths.
    Returns:
        List of lists where each element is a list of spotted words for that image.
    """
    transform = get_ocr_transform()
    results = []
    
    # We do sequential processing to minimize peak memory consumption
    with torch.no_grad():
        for path in image_paths:
            try:
                img = Image.open(path).convert("RGB")
                # Add batch dimension [1, 3, H, W]
                x = transform(img).unsqueeze(0).to(device)
                
                # STVGLNet_test forward returns: predictions, frozen_features
                predictions, _ = model(x)
                
                recs = []
                if len(predictions) > 0:
                    for pred in predictions:
                        instances = pred["instances"].to("cpu")
                        # Some versions use recs attribute directly
                        if hasattr(instances, "recs"):
                            for rec in instances.recs:
                                word = rec_decode(rec)
                                if word:
                                    recs.append(word)
                results.append(recs)
            except Exception as e:
                print(f"[OCR] Warning: Failed to process {path.name} ({e})")
                results.append([])
                
    return results
