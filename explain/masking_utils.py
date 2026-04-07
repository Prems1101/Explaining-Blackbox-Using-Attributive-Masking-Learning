import torch


def get_mask_embedding(model_F, tokenizer, device):
    """
    Returns the embedding for the [MASK] token from model F's embedding table.
    Paper (Section 3.3): m = embedding of <MASK> token for encoder-based models.
    This is fixed and NOT optimized.
    """
    mask_id = tokenizer.mask_token_id
    if mask_id is None:
        mask_id = tokenizer.unk_token_id
    with torch.no_grad():
        mask_emb = model_F.distilbert.embeddings.word_embeddings(
            torch.tensor([mask_id], device=device)
        )  # [1, D]
    return mask_emb.squeeze(0).detach()  # [D]


def soft_mask(embeddings, scores, mask_embedding):
    """
    Equation (3) from the paper:
    M(av, xv) = av[i]*xv_i + (1 - av[i])*m

    embeddings:     [B, T, D]
    scores:         [B, T]        attribution map av ∈ [0,1]
    mask_embedding: [D]

    score=1 → keep original token
    score=0 → fully replaced by mask
    """
    m = mask_embedding.view(1, 1, -1)       # [1, 1, D]
    a = scores.unsqueeze(-1)                 # [B, T, 1]
    return a * embeddings + (1 - a) * m


def inverse_mask(embeddings, scores, mask_embedding):
    """
    Equation (2) from the paper: x''v = M(1-av, xv)
    Applies the complement of the attribution map.
    High-score tokens get masked out → tests comprehensiveness.
    """
    return soft_mask(embeddings, 1.0 - scores, mask_embedding)
