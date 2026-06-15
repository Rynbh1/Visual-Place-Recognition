import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
#          Multi-Similarity Loss (implémentation PyTorch native)
# ---------------------------------------------------------------------------
class MultiSimilarityLoss(nn.Module):
    def __init__(self, alpha=2.0, beta=50.0, base=0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.base = base

    def forward(self, embeddings, labels):
        # embeddings: [B, D] déjà normalisés L2 par MegaLoc
        batch_size = embeddings.size(0)
        
        # Matrice de similarité cosine (produit matriciel car normés L2)
        sim_mat = torch.matmul(embeddings, embeddings.t())
        
        # Masques pour positifs et négatifs
        labels = labels.unsqueeze(1)
        pos_mask = (labels == labels.t()).float()
        neg_mask = (labels != labels.t()).float()
        
        # Retirer l'auto-similarité (la diagonale) du masque positif
        pos_mask = pos_mask - torch.eye(batch_size, device=embeddings.device)
        
        loss = 0.0
        count = 0
        for i in range(batch_size):
            pos_idx = torch.nonzero(pos_mask[i]).flatten()
            neg_idx = torch.nonzero(neg_mask[i]).flatten()
            
            if len(pos_idx) == 0 or len(neg_idx) == 0:
                continue
                
            pos_sims = sim_mat[i, pos_idx]
            neg_sims = sim_mat[i, neg_idx]
            
            # Formule de la Multi-Similarity Loss
            loss_pos = (1.0 / self.alpha) * torch.log(1.0 + torch.sum(torch.exp(-self.alpha * (pos_sims - self.base))))
            loss_neg = (1.0 / self.beta) * torch.log(1.0 + torch.sum(torch.exp(self.beta * (neg_sims - self.base))))
            
            loss += loss_pos + loss_neg
            count += 1
            
        return loss / max(count, 1)
