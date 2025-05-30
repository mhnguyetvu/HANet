import pytorch_lightning as pl
from torch.utils.data import DataLoader
from models.hanet_model import HANetModel
from utils.data_loader import MAVENDataset
from utils.memory_utils import load_memory, save_memory, build_memory_set, prototypical_augment
from utils.losses import info_nce_loss
import torch
import os
import json
import argparse

# Function to extract label2id mapping from incremental task
def extract_incremental_label_map(json_path, base_labels):
    new_labels = set()
    with open(json_path, 'r') as f:
        for line in f:
            doc = json.loads(line)
            for event in doc.get("events", []):
                if event["type"] not in base_labels:
                    new_labels.add(event["type"])
    full_labels = sorted(base_labels.union(new_labels))
    return {label: idx for idx, label in enumerate(full_labels)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, help="Path to incremental_task_*.jsonl")
    parser.add_argument("--base_ckpt", type=str, default="hanet_base.ckpt", help="Base model checkpoint")
    parser.add_argument("--output", type=str, default="hanet_incremental.ckpt", help="Output model checkpoint")
    args = parser.parse_args()

    # Config
    model_name = "bert-base-uncased"
    batch_size = 4
    max_epochs = 3
    data_dir = "data"
    memory_path = os.path.join(data_dir, "memory.pt")

    # Load base label map
    with open(os.path.join(data_dir, "label2id.json"), 'r') as f:
        base_label2id = json.load(f)

    # Extend label map
    label2id = extract_incremental_label_map(args.task, set(base_label2id.keys()))

    # Load incremental dataset
    inc_set = MAVENDataset(jsonl_path=args.task, tokenizer_name=model_name, label2id=label2id)

    # Load memory and merge
    memory_data = load_memory(memory_path)
    if memory_data:
        memory_examples = [ex for ex_list in memory_data.values() for ex in ex_list]
        inc_set.samples.extend(memory_examples)

    # Reload DataLoader
    inc_loader = DataLoader(inc_set, batch_size=batch_size, shuffle=True)

    # Load model
    model = HANetModel.load_from_checkpoint(args.base_ckpt, model_name=model_name, num_labels=len(label2id))

    # Update classifier head if needed
    if model.classifier.out_features != len(label2id):
        model.classifier = torch.nn.Linear(model.bert.config.hidden_size, len(label2id))
        model.hparams.num_labels = len(label2id)

    # Trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        log_every_n_steps=10,
        default_root_dir="lightning_logs_inc"
    )

    # Train
    trainer.fit(model, inc_loader)

    # Save model and updated memory
    trainer.save_checkpoint(args.output)
    print(f"✅ Incremental training complete. Model saved to {args.output}")

    # Rebuild memory using updated model (optional: k=1)
    new_memory = build_memory_set(inc_set, label2id, k=1)
    save_memory(new_memory, memory_path)
    print(f"🧠 Memory updated and saved to {memory_path}")