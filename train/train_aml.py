"""
pAML Pretraining — Section 3.1 of the AML paper.

Key design decisions (matching the paper):
1. Model F is COMPLETELY FROZEN throughout — only Gθ is trained.
2. Mask token m = embedding of [MASK] from model F (fixed, not learned).
3. yv = softmax output of F on original input (not one-hot).
4. Loss = λp*Lp + λa*La + λinv*Linv  (Equations 4-7).
   - Lp: cross-entropy between masked prediction and yv (preserve).
   - Linv: -log(1 - F(x'')[y]) (destroy with inverse mask).
   - La: BCE sparsity prior on scores.
5. Attribution model Gθ uses backbone + MLP head (not a flat 2-layer MLP).
6. Best model selected by monitoring metric of interest on validation set.
"""

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from models.attribution_model import AttributionModel
from explain.masking_utils import soft_mask, inverse_mask, get_mask_embedding

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Hyperparameters (tuned on validation set per paper Section 3.3) ─────────
LAMBDA_P   = 1.0   # preserve masked prediction
LAMBDA_INV = 5.0   # destroy inverse-masked prediction
LAMBDA_A   = 2.0   # sparsity regularisation
LR         = 2e-5
EPOCHS     = 5
BATCH_SIZE = 8
MAX_LEN    = 128
TRAIN_SIZE = 68000

# ── Load black-box model F (FROZEN) ─────────────────────────────────────────
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model_F = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
model_F.eval()
for p in model_F.parameters():
    p.requires_grad = False   # Paper: "parameters of F remain fixed throughout"

tokenizer = AutoTokenizer.from_pretrained(model_name)

# ── Mask embedding m (fixed, from F's embedding table) ──────────────────────
mask_emb = get_mask_embedding(model_F, tokenizer, device)  # [D], detached

# ── Attribution model Gθ ────────────────────────────────────────────────────
model_G = AttributionModel(backbone_name="distilbert-base-uncased",
                           num_classes=2, hidden_dim=768).to(device)
optimizer = torch.optim.AdamW(model_G.parameters(), lr=LR, weight_decay=1e-2)

# ── Data ─────────────────────────────────────────────────────────────────────
dataset = load_dataset("sst2")
train_texts = dataset["train"]["sentence"][:TRAIN_SIZE]
val_texts   = dataset["validation"]["sentence"][:67000]

train_loader = DataLoader(train_texts, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_texts,   batch_size=BATCH_SIZE, shuffle=False)


# ── Loss helpers ─────────────────────────────────────────────────────────────
def aml_loss(logits_masked, logits_inv, scores, yv):
    """
    Equation (4):  L = λp*Lp + λa*La + λinv*Linv
    """
    # Lp: cross-entropy between masked prediction and yv (Eq. 5)
    Lp = F.cross_entropy(logits_masked, yv.argmax(dim=-1))

    # Linv: -log(1 - p_y(x''))  (Eq. 6)
    y_idx = yv.argmax(dim=-1)
    inv_probs = torch.softmax(logits_inv, dim=-1)
    p_y_inv = inv_probs.gather(1, y_idx.unsqueeze(1)).squeeze(1)
    Linv = -torch.mean(torch.log(1.0 - p_y_inv + 1e-8))

    # La: BCE sparsity  -mean(log(1 - av[j]))  (Eq. 7)
    La = -torch.mean(torch.log(1.0 - scores + 1e-8))

    return LAMBDA_P * Lp + LAMBDA_INV * Linv + LAMBDA_A * La, Lp, Linv, La


def run_step(text_batch, train=True):
    inputs = tokenizer(text_batch, return_tensors="pt", padding=True,
                       truncation=True, max_length=MAX_LEN)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Step 1 — original prediction yv (no grad needed, F is frozen)
    with torch.no_grad():
        logits_orig = model_F(**inputs).logits
        yv = torch.softmax(logits_orig, dim=-1)          # [B, C]

    # Step 2 — attribution scores from Gθ
    if train:
        scores = model_G(inputs["input_ids"], inputs["attention_mask"], yv)
    else:
        with torch.no_grad():
            scores = model_G(inputs["input_ids"], inputs["attention_mask"], yv)
    scores = torch.clamp(scores, 1e-4, 1 - 1e-4)

    # Step 3 — get raw token embeddings (before positional encoding for simplicity)
    with torch.no_grad():
        token_embs = model_F.distilbert.embeddings.word_embeddings(inputs["input_ids"])

    # Step 4 — masked and inverse-masked forward passes through F
    x_masked = soft_mask(token_embs, scores.detach() if not train else scores, mask_emb)
    x_inv    = inverse_mask(token_embs, scores.detach() if not train else scores, mask_emb)

    logits_masked = model_F(inputs_embeds=x_masked,
                            attention_mask=inputs["attention_mask"]).logits
    logits_inv    = model_F(inputs_embeds=x_inv,
                            attention_mask=inputs["attention_mask"]).logits

    loss, Lp, Linv, La = aml_loss(logits_masked, logits_inv, scores, yv)
    return loss, Lp, Linv, La


def evaluate(loader):
    model_G.eval()
    total = 0.0
    with torch.no_grad():
        for batch in loader:
            loss, *_ = run_step(batch, train=False)
            total += loss.item()
    return total / len(loader)


# ── Training loop ─────────────────────────────────────────────────────────────
print("Starting pAML pretraining…")
best_val_loss = float("inf")

for epoch in range(EPOCHS):
    model_G.train()
    total_loss = 0.0

    for i, batch in enumerate(train_loader):
        loss, Lp, Linv, La = run_step(batch, train=True)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_G.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        if i % 50 == 0:
            print(f"  E{epoch+1} step {i:4d} | loss={loss.item():.4f} "
                  f"Lp={Lp.item():.4f} Linv={Linv.item():.4f} La={La.item():.4f}")

    avg_train = total_loss / len(train_loader)
    val_loss  = evaluate(val_loader)
    print(f"Epoch {epoch+1} | train={avg_train:.4f} | val={val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model_G.state_dict(), "paml_best.pt")
        print("  ✅ Saved best pAML model")

print("pAML training complete.")
