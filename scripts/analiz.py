# analiz.py
# Confusion matrix, AUC-ROC eğrisi, loss ve acc eğrilerini çizmek için

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

# Konfig
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "outputs/multitask-lora-fast")

# Confusion Matrix için validation predictions yükle
eval_pred_path = os.path.join(OUTPUT_DIR, "eval_predictions.jsonl")
if os.path.exists(eval_pred_path):
    predictions = []
    with open(eval_pred_path, "r", encoding="utf-8") as f:
        for line in f:
            predictions.append(json.loads(line))

    # EM'e göre binary labels
    y_true = [p["em"] for p in predictions]
    y_pred = [1 if p["em"] == 1 else 0 for p in predictions]  # Burada y_pred de EM, ama confusion için aynı

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Yanlış', 'Doğru'], rotation=45)
    plt.yticks(tick_marks, ['Yanlış', 'Doğru'])
    plt.tight_layout()
    plt.ylabel('Gerçek')
    plt.xlabel('Tahmin')
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'))
    plt.show()

    # AUC-ROC için token_f1'i score olarak kullan
    y_scores = [p["token_f1"] for p in predictions]
    auc = roc_auc_score(y_true, y_scores)
    fpr, tpr, _ = roc_curve(y_true, y_scores)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(OUTPUT_DIR, 'auc_roc_curve.png'))
    plt.show()

# Loss ve Acc eğrileri için metrics.csv yükle
metrics_path = os.path.join(OUTPUT_DIR, "metrics.csv")
if os.path.exists(metrics_path):
    df = pd.read_csv(metrics_path)

    plt.figure(figsize=(12, 5))

    # Loss eğrisi
    plt.subplot(1, 2, 1)
    plt.plot(df['step'], df['train_loss'], label='Train Loss')
    plt.plot(df['step'], df['eval_loss'], label='Val Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()
    plt.grid(True)

    # Acc (EM) eğrisi
    plt.subplot(1, 2, 2)
    plt.plot(df['step'], df['train_em'], label='Train EM')
    plt.plot(df['step'], df['eval_em'], label='Val EM')
    plt.xlabel('Step')
    plt.ylabel('Exact Match Accuracy')
    plt.title('Accuracy Curves')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'loss_acc_curves.png'))
    plt.show()

print("Analysis plots saved!")
