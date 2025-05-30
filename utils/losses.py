import torch
import torch.nn.functional as F

def info_nce_loss(anchor, positive, negatives, temperature=0.07):
    """
    Compute InfoNCE loss for contrastive learning.

    Args:
        anchor (Tensor): Anchor representations, shape (B, D)
        positive (Tensor): Positive samples for each anchor, shape (B, D)
        negatives (Tensor): Negative samples, shape (B_neg, D)
        temperature (float): Scaling factor

    Returns:
        Tensor: scalar loss value
    """
    # Positive score: cosine sim between anchor and positive
    pos_score = F.cosine_similarity(anchor, positive) / temperature  # (B,)

    # Negative scores: (B, B_neg)
    neg_score = torch.matmul(anchor, negatives.T) / temperature

    # Construct logits: (B, 1 + B_neg)
    logits = torch.cat([pos_score.unsqueeze(1), neg_score], dim=1)

    # Labels: first column (positive is correct)
    labels = torch.zeros(anchor.size(0), dtype=torch.long).to(anchor.device)

    return F.cross_entropy(logits, labels)