# 🔍 AML Explainability: Attributive Masking Learning

This repository hosts an implementation of **Attributive Masking Learning (AML)** for explaining black-box NLP classification models (such as DistilBERT), fine-tuned on the SST-2 sentiment dataset. It features an interactive Streamlit UI that allows users to instantly visualize which words heavily influenced the model's prediction.

## 🚀 Two Modes of Explanation

Our system supports two distinct methods for attribution extraction:
1. **pAML (Predictive Attribution Model):** A globally trained attribution model that produces highly accurate, sparse highlights. It learned general rules over 67,000 sentences.
2. **fAML (Instance-Specific Finetuning):** Takes the pAML baseline and performs "on-the-fly" gradient optimizations for a few steps specifically on the sentence you provided.

## 🛠️ Setup & Local Usage

1. Create a virtual environment and load the requirements:
```bash
pip install -r requirements.txt
```

2. Play with the interactive Explainability App:
```bash
streamlit run app/app.py
```

## 🧠 Training & Hyperparameters

We utilize a custom sparsity regularization penalty (`LAMBDA_A`) to prevent the model from blindly highlighting useless stop-words (like "is" or "the"). The model is trained aggressively against the SST-2 dataset to maximize the comprehensiveness and sufficiency of its explanations.

To retrain the base model yourself:
```bash
python -m train.train_aml
```

## ☁️ Deployment Note

Because the `.pt` model weights exceed 100MB, they are tracked via **Git Large File Storage (LFS)**. If you are cloning this repository, make sure you have Git LFS installed or the model weights will not download correctly!