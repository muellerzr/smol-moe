from datasets import load_dataset
from transformers import AutoTokenizer
import torch
from contextlib import nullcontext

# parameters
seq_len = 4096
output_path = "tokenized_tinystories"
model_name = "Qwen/Qwen3-0.6B"  # or your actual tokenizer name

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
raw_dataset = load_dataset("roneneldan/TinyStories", split="train[:10%]")

def tokenize_function(examples):
    tokenized_batch = tokenizer(
        examples["text"],
        padding=False,
        truncation=True,
        max_length=seq_len,
    )
    tokenized_batch["labels"] = tokenized_batch["input_ids"].copy()
    return tokenized_batch

# 1️⃣ Tokenize text
tokenized_dataset = raw_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
    num_proc=8,
)

# 2️⃣ Pack sequences into fixed length
def create_packed_sequences(examples):
    all_tokens = []
    for input_ids in examples["input_ids"]:
        all_tokens.extend(input_ids)

    num_sequences = len(all_tokens) // (seq_len + 1)
    packed_input_ids, packed_labels, packed_position_ids = [], [], []

    for i in range(num_sequences):
        start_idx = i * (seq_len + 1)
        end_idx = start_idx + (seq_len + 1)
        full_seq = all_tokens[start_idx:end_idx]
        packed_input_ids.append(full_seq[:-1])
        packed_labels.append(full_seq[1:])
        packed_position_ids.append(list(range(seq_len)))

    return {
        "input_ids": packed_input_ids,
        "labels": packed_labels,
        "shift_labels": packed_labels,
        "position_ids": packed_position_ids,
    }

packed_dataset = tokenized_dataset.map(
    create_packed_sequences,
    batched=True,
    remove_columns=tokenized_dataset.column_names,
    batch_size=1000,
    num_proc=4,
)

# 3️⃣ Save to disk in Arrow format
packed_dataset.save_to_disk(output_path)
print(f"✅ Saved pretokenized TinyStories dataset to {output_path}")
