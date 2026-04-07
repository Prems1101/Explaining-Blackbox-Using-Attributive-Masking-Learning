import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from models.attribution_model import AttributionModel
from explain.masking_utils import get_mask_embedding

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AMLExplainer:
    """
    pAML explainer: single forward pass through pretrained Gθ.
    Paper Section 3.1: "an attribution map can be generated for any input
    through a single forward pass of the input and its prediction via the
    pretrained attribution model."
    """

    def __init__(self, model_path="paml_best.pt"):
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"

        # Model F — frozen black-box
        self.model_F = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        self.model_F.eval()
        for p in self.model_F.parameters():
            p.requires_grad = False

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Pretrained attribution model Gθ
        self.model_G = AttributionModel(backbone_name="distilbert-base-uncased",
                                        num_classes=2, hidden_dim=768).to(device)
        self.model_G.load_state_dict(torch.load(model_path, map_location=device))
        self.model_G.eval()

    def explain(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True,
                                padding=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model_F(**inputs).logits
            probs  = torch.softmax(logits, dim=-1)[0]          # [C]

        label      = "POSITIVE" if probs[1] > probs[0] else "NEGATIVE"
        confidence = probs.max().item()

        with torch.no_grad():
            scores = self.model_G(
                inputs["input_ids"],
                inputs["attention_mask"],
                probs.unsqueeze(0)
            )[0]   # [T]

        return self._format(inputs, scores, probs, label, confidence, "pAML")

    def _format(self, inputs, scores, probs, label, confidence, mode):
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        scores_np = scores.cpu().float().numpy()

        pairs = [
            (tok.replace("##", ""), float(s))
            for tok, s in zip(tokens, scores_np)
            if tok not in ["[CLS]", "[SEP]", "[PAD]", "<s>", "</s>"]
        ]

        # Normalize to [0, 1]
        max_s = max(s for _, s in pairs) + 1e-8
        pairs = [(w, s / max_s) for w, s in pairs]

        return {
            "mode": mode,
            "prediction": {"label": label, "score": confidence},
            "word_importance_original": pairs,
            "word_importance_sorted":   sorted(pairs, key=lambda x: x[1], reverse=True),
        }
