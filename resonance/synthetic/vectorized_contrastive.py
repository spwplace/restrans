"""Vectorized InfoNCE contrastive loss to replace the slow Python-loop version."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorizedContrastiveLoss(nn.Module):
    """Fast vectorized InfoNCE for proof groups."""
    
    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature
        
    def forward(self, embeddings, positive_mask, negative_mask=None):
        """
        embeddings: [N, D]
        positive_mask: [N, N] bool - True for positive pairs
        negative_mask: [N, N] bool - True for negative pairs
        """
        if negative_mask is None:
            negative_mask = ~positive_mask
            diag = torch.arange(embeddings.size(0), device=embeddings.device)
            negative_mask[diag, diag] = False
            
        # Normalize and compute similarity
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        sim = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        N = sim.size(0)
        
        # Exclude self-similarity
        diag_mask = torch.eye(N, device=sim.device, dtype=torch.bool)
        sim = sim.masked_fill(diag_mask, float('-inf'))
        
        # For each anchor, compute log prob of positives over all valid pairs
        # We want: log(exp(sim[i, pos]) / sum(exp(sim[i, valid])))
        # where valid = all except self
        
        sim_exp = torch.exp(sim)
        
        # Sum of exp similarities for positive pairs
        pos_exp = sim_exp * positive_mask.float()
        pos_sum = pos_exp.sum(dim=1)  # [N]
        
        # Sum of exp similarities for all valid pairs (excluding self)
        valid_mask = ~diag_mask
        valid_sum = (sim_exp * valid_mask.float()).sum(dim=1)  # [N]
        
        # Only compute loss for anchors that have at least one positive and one negative
        has_pos = positive_mask.sum(dim=1) > 0
        has_neg = negative_mask.sum(dim=1) > 0
        valid_anchors = has_pos & has_neg
        
        if valid_anchors.sum() == 0:
            return torch.tensor(0.0, device=sim.device, requires_grad=True)
        
        # Add small epsilon for numerical stability
        loss = -torch.log((pos_sum[valid_anchors] + 1e-8) / (valid_sum[valid_anchors] + 1e-8))
        return loss.mean()
