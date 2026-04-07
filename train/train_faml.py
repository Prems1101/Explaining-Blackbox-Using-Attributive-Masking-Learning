"""
fAML — Instance-Specific Finetuning at Inference Time (Section 3.1).

Key corrections vs original code:
1. fAML is NOT a separate training from scratch — it STARTS from the pretrained
   pAML checkpoint and does a small number of gradient steps on a SINGLE example.
2. Model F remains COMPLETELY FROZEN — only Gθ is finetuned.
3. Hyperparameters λp, λa, λinv are fixed from pretraining (not re-tuned here).
4. The best attribution map is selected by monitoring the metric of interest
   across finetuning steps (Section 3.1: "selecting the attribution map that
   performs the best on the metric at hand during the finetuning phase").
"""

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import copy
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from models.attribution_model import AttributionModel
from explain.masking_utils import soft_mask, inverse_mask, get_mask_embedding

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Hyperparameters (same as pAML pretraining) ───────────────────────────────
LAMBDA_P   = 1.0
LAMBDA_INV = 5.0
LAMBDA_A   = 2.0
FAML_LR    = 1e-4   # slightly higher LR for instance-specific finetuning
FAML_STEPS = 20     # paper: "very small number of finetuning steps"
MAX_LEN    = 128


def faml_explain(text, paml_checkpoint="paml_best.pt"):
    """
    Given a single text, fine-tune a copy of the pretrained Gθ for a few
    steps on that specific instance, then return the best attribution map.

    Returns:
        scores:     [T] per-token attribution scores
        tokens:     list of token strings
        label:      predicted label string
        confidence: prediction confidence
    """

    # ── Load model F (frozen) ────────────────────────────────────────────────
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    model_F = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model_F.eval()
    for p in model_F.parameters():
        p.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    mask_emb = get_mask_embedding(model_F, tokenizer, device)

    # ── Load pretrained Gθ and clone it for this instance ───────────────────
    model_G = AttributionModel(backbone_name="distilbert-base-uncased",
                               num_classes=2, hidden_dim=768).to(device)
    model_G.load_state_dict(torch.load(paml_checkpoint, map_location=device))

    # Clone so the original checkpoint is not modified
    model_Gv = copy.deepcopy(model_G).to(device)
    optimizer = torch.optim.AdamW(model_Gv.parameters(), lr=FAML_LR)

    # ── Tokenize the single instance ─────────────────────────────────────────
    inputs = tokenizer(text, return_tensors="pt", padding=True,
                       truncation=True, max_length=MAX_LEN)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Original prediction yv (fixed — F is frozen)
    with torch.no_grad():
        logits_orig = model_F(**inputs).logits
        yv = torch.softmax(logits_orig, dim=-1)               # [1, C]
        token_embs = model_F.distilbert.embeddings.word_embeddings(
            inputs["input_ids"])                              # [1, T, D]

    label      = "POSITIVE" if yv[0, 1] > yv[0, 0] else "NEGATIVE"
    confidence = yv.max().item()

    # ── Instance-specific finetuning ─────────────────────────────────────────
    best_comp  = -float("inf")
    best_scores = None

    for step in range(FAML_STEPS):
        model_Gv.train()
        scores = model_Gv(inputs["input_ids"], inputs["attention_mask"], yv)
        scores = torch.clamp(scores, 1e-4, 1 - 1e-4)

        x_masked = soft_mask(token_embs, scores, mask_emb)
        x_inv    = inverse_mask(token_embs, scores, mask_emb)

        lm  = model_F(inputs_embeds=x_masked,
                      attention_mask=inputs["attention_mask"]).logits
        li  = model_F(inputs_embeds=x_inv,
                      attention_mask=inputs["attention_mask"]).logits

        Lp   = F.cross_entropy(lm, yv.argmax(dim=-1))
        p_yi = torch.softmax(li, dim=-1).gather(1, yv.argmax(dim=-1, keepdim=True)).squeeze()
        Linv = -torch.log(1.0 - p_yi + 1e-8)
        La   = -torch.mean(torch.log(1.0 - scores + 1e-8))

        loss = LAMBDA_P * Lp + LAMBDA_INV * Linv + LAMBDA_A * La

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_Gv.parameters(), 1.0)
        optimizer.step()

        # Monitor comprehensiveness — keep best attribution map (Section 3.1)
        with torch.no_grad():
            model_Gv.eval()
            s = model_Gv(inputs["input_ids"], inputs["attention_mask"], yv)
            s = torch.clamp(s, 1e-4, 1 - 1e-4)
            orig_p = yv[0, yv.argmax(dim=-1)[0]].item()
            xi = inverse_mask(token_embs, s, mask_emb)
            li2 = model_F(inputs_embeds=xi,
                          attention_mask=inputs["attention_mask"]).logits
            inv_p = torch.softmax(li2, dim=-1)[0, yv.argmax(dim=-1)[0]].item()
            comp = orig_p - inv_p  # comprehensiveness proxy

        if comp > best_comp:
            best_comp = comp
            best_scores = s.squeeze(0).detach()

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    return best_scores, tokens, label, confidence

if __name__ == "__main__":
    import time
    
    # Path to your successfully trained pAML model
    checkpoint_path = "paml_best.pt"
    
    text_to_test = "This movie is absolutely magnificent!"
    print(f"Running fAML instance-specific finetuning for: '{text_to_test}'")
    start_time = time.time()
    
    # Run the fAML explanation
    scores, tokens, label, confidence = faml_explain(text_to_test, paml_checkpoint=checkpoint_path)
    
    print(f"\nfAML Finetuning completed in {time.time() - start_time:.2f} seconds.")
    print(f"Prediction Label: {label} (Confidence: {confidence:.4f})\n")
    print("Tokens and Attributions:")
    print("-" * 30)
    for token, score in zip(tokens, scores):
        if token not in ['[CLS]', '[SEP]', '[PAD]']:
            print(f"{token:15}: {score.item():.4f}")

