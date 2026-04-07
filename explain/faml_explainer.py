import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transformers import AutoTokenizer
from train.train_faml import faml_explain

device_str = "cpu"


class FAMLExplainer:
    """
    fAML explainer: instance-specific finetuning of the pretrained Gθ.
    Paper Section 3.1: "continuously finetuning the pretrained attribution
    model on a specific instance w.r.t. the metric of interest."
    """

    def __init__(self, model_path="paml_best.pt"):
        self.paml_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            "distilbert-base-uncased-finetuned-sst-2-english"
        )

    def explain(self, text):
        scores, tokens, label, confidence = faml_explain(text, self.paml_path)

        pairs = [
            (tok.replace("##", ""), float(s))
            for tok, s in zip(tokens, scores.cpu().float().numpy())
            if tok not in ["[CLS]", "[SEP]", "[PAD]", "<s>", "</s>"]
        ]

        max_s = max(s for _, s in pairs) + 1e-8
        pairs = [(w, s / max_s) for w, s in pairs]

        return {
            "mode": "fAML",
            "prediction": {"label": label, "score": confidence},
            "word_importance_original": pairs,
            "word_importance_sorted":   sorted(pairs, key=lambda x: x[1], reverse=True),
        }
