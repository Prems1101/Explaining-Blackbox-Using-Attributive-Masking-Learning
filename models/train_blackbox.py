from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import accuracy_score

# Load dataset
dataset = load_dataset("imdb")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length")

dataset = dataset.map(tokenize, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=2
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {"accuracy": accuracy_score(labels, preds)}

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,  # 🔥 KEY
    metric_for_best_model="accuracy",
    greater_is_better=True,
    num_train_epochs=2,  # keep small
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"].shuffle(seed=42).select(range(5000)),
    eval_dataset=dataset["test"].select(range(1000)),
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

# SAVE BEST MODEL
trainer.save_model("best_model")
tokenizer.save_pretrained("best_model")