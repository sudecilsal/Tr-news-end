# save_tokenized_data.py
# Tokenized verileri kaydetmek için script

import os
import numpy as np
from datasets import load_from_disk
from transformers import AutoTokenizer

# Konfig
MODEL_NAME = os.environ.get("BASE_MODEL", "google/mt5-small")
DATA_DIR = os.environ.get("DATA_DIR", "data2/processed/multitask_text2text")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "outputs/multitask-lora-fast")
MAX_SOURCE_LEN = int(os.environ.get("MAX_SOURCE_LEN", 384))
MAX_TARGET_LEN = int(os.environ.get("MAX_TARGET_LEN", 64))
MAX_TRAIN_SAMPLES = int(os.environ.get("MAX_TRAIN_SAMPLES", 3000))
MAX_EVAL_SAMPLES = int(os.environ.get("MAX_EVAL_SAMPLES", 600))
MAX_TEST_SAMPLES = int(os.environ.get("MAX_TEST_SAMPLES", 600))

# Tokenizer yükle
print(f"Loading tokenizer: {MODEL_NAME}")
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# Dataset yükle
assert os.path.exists(DATA_DIR), f"Dataset not found: {DATA_DIR}"
ds = load_from_disk(DATA_DIR)

def subset(d, n):
    return d.select(range(min(len(d), n))) if n and len(d) > n else d

train_raw = subset(ds["train"], MAX_TRAIN_SAMPLES)
val_raw = subset(ds["validation"], MAX_EVAL_SAMPLES)
test_raw = subset(ds["test"], MAX_TEST_SAMPLES)

print(f"Loaded dataset: train={len(train_raw)}, val={len(val_raw)}, test={len(test_raw)}")

# Tokenization
def preprocess(batch):
    X = tok(batch["source"], max_length=MAX_SOURCE_LEN, truncation=True)
    Y = tok(text_target=batch["target"], max_length=MAX_TARGET_LEN, truncation=True)
    X["labels"] = Y["input_ids"]
    return X

remove_cols = list(train_raw.features)
train_tok = train_raw.map(preprocess, batched=True, remove_columns=remove_cols, desc="Tokenize train")
val_tok = val_raw.map(preprocess, batched=True, remove_columns=remove_cols, desc="Tokenize val")
test_tok = test_raw.map(preprocess, batched=True, remove_columns=remove_cols, desc="Tokenize test")

# Verileri numpy array'e çevir ve kaydet
def save_tokenized_data(dataset, name, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    input_ids = np.array(dataset["input_ids"])
    attention_mask = np.array(dataset["attention_mask"])
    labels = np.array(dataset["labels"])

    np.save(os.path.join(output_dir, f"{name}_x.npy"), input_ids)
    np.save(os.path.join(output_dir, f"{name}_attention_mask.npy"), attention_mask)
    np.save(os.path.join(output_dir, f"{name}_y.npy"), labels)
    print(f"Saved {name} data to {output_dir}")

save_tokenized_data(train_tok, "train", OUTPUT_DIR)
save_tokenized_data(val_tok, "val", OUTPUT_DIR)
save_tokenized_data(test_tok, "test", OUTPUT_DIR)

print("All tokenized data saved!")
