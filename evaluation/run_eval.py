from explain.aml_explainer import AMLExplainer
from explain.faml_explainer import FAMLExplainer
from evaluation.metrics import *
from datasets import load_dataset

# ===== LOAD MODELS =====
paml = AMLExplainer()
faml = FAMLExplainer()

# ===== LOAD DATA =====
dataset = load_dataset("imdb")

# 🔥 take 50 samples
texts = dataset["test"]["text"][:50]

# ===== FUNCTION =====
def evaluate_model(model, texts, name):

    total_suff = 0
    total_comp = 0
    total_spars = 0

    count = 0

    for i, text in enumerate(texts):

        try:
            result = model.explain(text)

            suff = sufficiency_score(model, text)
            comp = comprehensiveness_score(model, text)
            spars = sparsity_score(result)

            total_suff += suff
            total_comp += comp
            total_spars += spars

            count += 1

            if i % 10 == 0:
                print(f"{name} → processed {i} samples")

        except Exception as e:
            print(f"Skipping sample {i}: {e}")
            continue

    return {
        "suff": total_suff / count,
        "comp": total_comp / count,
        "spars": total_spars / count
    }


# ===== RUN =====
print("\n🔍 Evaluating pAML...")
paml_scores = evaluate_model(paml, texts, "pAML")

print("\n🔍 Evaluating fAML...")
faml_scores = evaluate_model(faml, texts, "fAML")


# ===== FINAL OUTPUT =====
print("\n📊 FINAL RESULTS (Average over 50 samples)\n")

print(f"pAML  → Suff: {paml_scores['suff']:.4f}, "
      f"Comp: {paml_scores['comp']:.4f}, "
      f"Sparsity: {paml_scores['spars']:.2f}")

print(f"fAML → Suff: {faml_scores['suff']:.4f}, "
      f"Comp: {faml_scores['comp']:.4f}, "
      f"Sparsity: {faml_scores['spars']:.2f}")