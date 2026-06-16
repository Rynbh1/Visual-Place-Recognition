import gc
import sys
import types
from typing import List, Tuple, Union

import numpy as np
import torch

# Workaround patches to ensure sentence-transformers/transformers imports
# work correctly on specific older PyTorch versions
if not hasattr(torch, "compiler"):
    compiler = types.ModuleType("torch.compiler")
    compiler.disable = lambda recursive=False: (lambda x: x)
    torch.compiler = compiler
    sys.modules["torch.compiler"] = compiler
if not hasattr(torch, "float8_e4m3fn"):
    torch.float8_e4m3fn = torch.float16
if not hasattr(torch, "float8_e5m2"):
    torch.float8_e5m2 = torch.float16

# Patched load_state_dict implementing assign=True for PyTorch 2.0.1 (required by transformers)
original_load_state_dict = torch.nn.Module.load_state_dict
def patched_load_state_dict(self, state_dict, *args, **kwargs):
    assign = kwargs.pop("assign", False)
    if assign:
        for key, tensor in state_dict.items():
            if key in self._parameters:
                old_param = self._parameters[key]
                requires_grad = old_param.requires_grad if old_param is not None else True
                new_param = torch.nn.Parameter(tensor, requires_grad=requires_grad)
                self._parameters[key] = new_param
            elif key in self._buffers:
                self._buffers[key] = tensor
        kwargs["strict"] = False
        return original_load_state_dict(self, {}, *args, **kwargs)
    return original_load_state_dict(self, state_dict, *args, **kwargs)
torch.nn.Module.load_state_dict = patched_load_state_dict


from sentence_transformers import CrossEncoder

def clear_vram(model=None):
    """
    Forces garbage collection and empties the PyTorch CUDA cache to free up VRAM.
    Optionally deletes references to a specified model object.
    """
    if model is not None:
        try:
            del model
        except NameError:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_reranker(model_name: str = "Qwen/Qwen3-Reranker-0.6B") -> CrossEncoder:
    """
    Lazy loads the Qwen local reranker model using sentence_transformers.
    Casts the model to float16 to conserve GPU memory.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Reranker] Loading local reranker {model_name} on {device}...")
    
    # Load model
    reranker = CrossEncoder(model_name, device=device, trust_remote_code=True)
    
    # Cast to float16 / half precision if on CUDA
    if device == "cuda":
        reranker.model = reranker.model.half()
        
    reranker.tokenizer.pad_token = reranker.tokenizer.eos_token
    reranker.model.config.pad_token_id = reranker.tokenizer.eos_token_id
    
    print("[Reranker] Loaded successfully.")
    return reranker


def simple_text_overlap_score(q_words: List[str], db_words: List[str]) -> float:
    """
    Fallback textual overlap score calculation. Filters digit-containing tokens
    and returns Jaccard-like length ratio.
    """
    if not q_words or not db_words:
        return 0.0
    
    q_digits = [s for s in q_words if any(c.isdigit() for c in s)]
    db_digits = [s for s in db_words if any(c.isdigit() for c in s)]
    
    if not q_digits or not db_digits:
        return 0.0
        
    common = set(q_digits) & set(db_digits)
    numerator = sum(len(s) for s in common)
    denominator = sum(len(s) for s in q_digits)
    
    return numerator / denominator if denominator > 0 else 0.0


def late_fusion_rerank(
    reranker: Union[CrossEncoder, None],
    prediction: np.ndarray,
    query_words: List[str],
    db_words_list: List[List[str]],
    top_k: int = 50
) -> np.ndarray:
    """
    Late Fusion Reranking logic.
    If no text is detected in query, returns original visual candidate ranking.
    If text is detected, queries local Qwen model (if provided) to rerank the candidates,
    using visual similarity ranks as the secondary sorting key (tie-breaker).
    Otherwise, falls back to digit-matching overlap heuristics.
    """
    # Keep only the top K visual candidates to rerank
    cand_indices = prediction[:top_k]
    remaining_indices = prediction[top_k:]
    
    if not query_words:
        return prediction

    q_str = " ".join(query_words)
    
    # Reranking scoring
    if reranker is not None:
        try:
            # Prepare pairs: (query_text, candidate_text)
            pairs = []
            for ref_idx in cand_indices:
                cand_str = " ".join(db_words_list[ref_idx])
                pairs.append((q_str, cand_str))
                
            # Compute cross-encoder similarity scores
            scores = reranker.predict(pairs, show_progress_bar=False)
            
            # Zip and sort: sort by score descending, then by visual rank implicitly (original order)
            preds_with_scores = list(zip(cand_indices, scores))
            # stable sort to maintain relative visual rank in case of score ties
            preds_with_scores.sort(key=lambda x: x[1], reverse=True)
            
            reranked_cands = np.array([x[0] for x in preds_with_scores], dtype=prediction.dtype)
            return np.concatenate([reranked_cands, remaining_indices])
        except Exception as e:
            print(f"[Reranker] Warning: Qwen inference failed ({e}). Falling back to word overlap.")
            
    # Fallback to Jaccard digit matching
    scores = []
    for ref_idx in cand_indices:
        score = simple_text_overlap_score(query_words, db_words_list[ref_idx])
        scores.append(score)
        
    preds_with_scores = list(zip(cand_indices, scores))
    preds_with_scores.sort(key=lambda x: x[1], reverse=True)
    
    reranked_cands = np.array([x[0] for x in preds_with_scores], dtype=prediction.dtype)
    return np.concatenate([reranked_cands, remaining_indices])
