"""
Evaluation metrics from Section 4 / Appendix A.5 of the AML paper.

Metrics:
- Sufficiency  (Suff ↓): keep top-k%, measure how well prediction is preserved
- Comprehensiveness (Comp ↑): remove top-k%, measure drop in prediction
- Log-Odds (LO ↓): log-prob difference after masking top-k%
- Sparsity: fraction of tokens with score > threshold (informational)
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from explain.masking_utils import soft_mask, inverse_mask, get_mask_embedding

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model_F   = None
_tokenizer = None
_mask_emb  = None


def _load_f():
    global _model_F, _tokenizer, _mask_emb
    if _model_F is None:
        mn = "distilbert-base-uncased-finetuned-sst-2-english"
        _model_F   = AutoModelForSequenceClassification.from_pretrained(mn).to(device)
        _model_F.eval()
        _tokenizer = AutoTokenizer.from_pretrained(mn)
        _mask_emb  = get_mask_embedding(_model_F, _tokenizer, device)
    return _model_F, _tokenizer, _mask_emb


def _top_k_mask(scores_1d, k_frac=0.2):
    """Returns a binary mask [T] with top-k% positions = 1."""
    k = max(1, int(len(scores_1d) * k_frac))
    topk_idx = torch.topk(scores_1d, k).indices
    mask = torch.zeros_like(scores_1d)
    mask[topk_idx] = 1.0
    return mask


def sufficiency_score(explainer, text, k=0.20):
    """
    Appendix A.5: Suff = p(y'|x) - p(y'|x^(k))
    where x^(k) keeps only the top-k% tokens (rest masked).
    Lower is better (kept tokens fully explain the prediction).
    """
    model_F, tokenizer, mask_emb = _load_f()
    result = explainer.explain(text)

    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       padding=True, max_length=128)
    inputs = {kk: vv.to(device) for kk, vv in inputs.items()}

    with torch.no_grad():
        orig_probs = torch.softmax(model_F(**inputs).logits, dim=-1)[0]
        y_idx = orig_probs.argmax().item()
        orig_p = orig_probs[y_idx].item()

    # Build score tensor aligned with tokenizer output
    score_map = {w: s for w, s in result["word_importance_original"]}
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    scores = torch.tensor([
        score_map.get(t.replace("##", ""), 0.0) for t in tokens
    ], device=device)

    keep_mask = _top_k_mask(scores, k)   # 1=important, 0=mask out

    with torch.no_grad():
        embs = model_F.distilbert.embeddings.word_embeddings(inputs["input_ids"])
        x_suff = soft_mask(embs, keep_mask.unsqueeze(0), mask_emb)
        suff_p  = torch.softmax(
            model_F(inputs_embeds=x_suff,
                    attention_mask=inputs["attention_mask"]).logits, dim=-1
        )[0, y_idx].item()

    return orig_p - suff_p  # lower = better (ideally 0)


def comprehensiveness_score(explainer, text, k=0.20):
    """
    Appendix A.5: Comp = p(y'|x) - p(y'|x^(k))
    where x^(k) REMOVES the top-k% tokens (replaces with mask).
    Higher is better.
    """
    model_F, tokenizer, mask_emb = _load_f()
    result = explainer.explain(text)

    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       padding=True, max_length=128)
    inputs = {kk: vv.to(device) for kk, vv in inputs.items()}

    with torch.no_grad():
        orig_probs = torch.softmax(model_F(**inputs).logits, dim=-1)[0]
        y_idx  = orig_probs.argmax().item()
        orig_p = orig_probs[y_idx].item()

    score_map = {w: s for w, s in result["word_importance_original"]}
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    scores = torch.tensor([
        score_map.get(t.replace("##", ""), 0.0) for t in tokens
    ], device=device)

    # Comprehensiveness: REMOVE top-k (mask them → use inverse of keep_mask)
    keep_mask = _top_k_mask(scores, k)
    remove_mask = 1.0 - keep_mask   # kept tokens = 1, important tokens = 0

    with torch.no_grad():
        embs = model_F.distilbert.embeddings.word_embeddings(inputs["input_ids"])
        x_comp = soft_mask(embs, remove_mask.unsqueeze(0), mask_emb)
        comp_p  = torch.softmax(
            model_F(inputs_embeds=x_comp,
                    attention_mask=inputs["attention_mask"]).logits, dim=-1
        )[0, y_idx].item()

    return orig_p - comp_p   # higher = better


def sparsity_score(result, threshold=0.3):
    """Number of words with attribution score above threshold."""
    return sum(1 for _, s in result["word_importance_sorted"] if s > threshold)
