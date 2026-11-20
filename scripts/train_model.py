"""
SmartShield — TinyBERT Fine-Tuning Script
Fine-tunes TinyBERT on Indian SMS scam dataset and exports to ONNX (INT8).

Usage:
    python scripts/train_model.py
    
Output:
    models/smartshield_tinybert.onnx  (INT8 quantized)
    models/tokenizer/                  (saved tokenizer)
"""

import os
import sys
import csv
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Config
MODEL_NAME = "huawei-noah/TinyBERT_General_6L_768D"
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 2e-5
LABEL_MAP = {"legitimate": 0, "scam": 1}


class SMSDataset(Dataset):
    """SMS text classification dataset."""

    def __init__(self, filepath, tokenizer, max_length=MAX_LENGTH):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = []
        self.labels = []

        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.texts.append(row["text"])
                self.labels.append(LABEL_MAP[row["label"]])

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }


def compute_metrics(eval_pred):
    """Compute evaluation metrics for Trainer."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", pos_label=1
    )
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def train():
    """Fine-tune TinyBERT and export to ONNX."""
    print("=" * 60)
    print("SmartShield — TinyBERT Fine-Tuning")
    print("=" * 60)

    # 1. Load tokenizer and model
    print(f"\n[1/6] Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    )

    # 2. Load datasets
    print("[2/6] Loading datasets...")
    train_dataset = SMSDataset(os.path.join(DATA_DIR, "train.csv"), tokenizer)
    val_dataset = SMSDataset(os.path.join(DATA_DIR, "val.csv"), tokenizer)
    test_dataset = SMSDataset(os.path.join(DATA_DIR, "test.csv"), tokenizer)

    print(f"  Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    # 3. Training
    print("[3/6] Training...")
    
    # Class weights for imbalance
    n_scam = sum(1 for l in train_dataset.labels if l == 1)
    n_legit = sum(1 for l in train_dataset.labels if l == 0)
    weight_legit = n_scam / (n_scam + n_legit)
    weight_scam = n_legit / (n_scam + n_legit)
    class_weights = torch.tensor([weight_legit, weight_scam], dtype=torch.float32)
    print(f"  Class weights: legit={weight_legit:.3f}, scam={weight_scam:.3f}")

    training_args = TrainingArguments(
        output_dir=os.path.join(MODEL_DIR, "checkpoints"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=10,
        save_total_limit=2,
        report_to="none",
    )

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(logits.device))
            loss = loss_fn(logits, labels)
            return (loss, outputs) if return_outputs else loss

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # 4. Evaluate on test set
    print("\n[4/6] Evaluating on test set...")
    test_results = trainer.evaluate(test_dataset)
    print(f"  Test Accuracy: {test_results['eval_accuracy']:.4f}")
    print(f"  Test Precision: {test_results['eval_precision']:.4f}")
    print(f"  Test Recall: {test_results['eval_recall']:.4f}")
    print(f"  Test F1: {test_results['eval_f1']:.4f}")

    # Confusion matrix
    test_preds = trainer.predict(test_dataset)
    preds = np.argmax(test_preds.predictions, axis=-1)
    labels = test_preds.label_ids
    cm = confusion_matrix(labels, preds)
    print(f"\n  Confusion Matrix:")
    print(f"              Pred Legit  Pred Scam")
    print(f"  Act Legit     {cm[0][0]:>5}      {cm[0][1]:>5}")
    print(f"  Act Scam      {cm[1][0]:>5}      {cm[1][1]:>5}")
    
    fpr = cm[0][1] / (cm[0][0] + cm[0][1]) if (cm[0][0] + cm[0][1]) > 0 else 0
    print(f"  False Positive Rate: {fpr:.2%}")
    
    print(f"\n  Full Classification Report:")
    print(classification_report(labels, preds, target_names=["legitimate", "scam"]))

    # 5. Save tokenizer
    print("[5/6] Saving tokenizer...")
    tokenizer_path = os.path.join(MODEL_DIR, "tokenizer")
    tokenizer.save_pretrained(tokenizer_path)

    # 6. Export to ONNX
    print("[6/6] Exporting to ONNX (INT8)...")
    onnx_path = os.path.join(MODEL_DIR, "smartshield_tinybert.onnx")
    
    dummy_input = tokenizer(
        "test message", max_length=MAX_LENGTH, padding="max_length",
        truncation=True, return_tensors="pt"
    )
    
    model.eval()
    torch.onnx.export(
        model,
        (dummy_input["input_ids"], dummy_input["attention_mask"]),
        onnx_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size"},
            "attention_mask": {0: "batch_size"},
            "logits": {0: "batch_size"}
        },
        opset_version=14,
        do_constant_folding=True,
    )
    
    # INT8 quantization
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        quantized_path = os.path.join(MODEL_DIR, "smartshield_tinybert_int8.onnx")
        quantize_dynamic(
            onnx_path, quantized_path,
            weight_type=QuantType.QInt8
        )
        final_size = os.path.getsize(quantized_path) / (1024 * 1024)
        print(f"  INT8 model saved: {quantized_path} ({final_size:.1f} MB)")
    except ImportError:
        print("  WARNING: onnxruntime.quantization not available. Saving FP32 model.")
        final_size = os.path.getsize(onnx_path) / (1024 * 1024)
        print(f"  FP32 model saved: {onnx_path} ({final_size:.1f} MB)")

    # Save evaluation results
    results_path = os.path.join(PROJECT_ROOT, "benchmarks", "training_results.json")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    
    import json
    results_data = {
        "model": MODEL_NAME,
        "epochs": EPOCHS,
        "max_length": MAX_LENGTH,
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "test_size": len(test_dataset),
        "test_accuracy": test_results["eval_accuracy"],
        "test_precision": test_results["eval_precision"],
        "test_recall": test_results["eval_recall"],
        "test_f1": test_results["eval_f1"],
        "false_positive_rate": fpr,
        "confusion_matrix": cm.tolist(),
    }
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"  Results saved: {results_path}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    train()
