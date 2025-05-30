import pytorch_lightning as pl
from torch.utils.data import DataLoader
from model.hanet_model import HANetModel
from data_loader import MAVENDataset
import os
import json

# Function to extract label2id mapping dynamically from dataset
def extract_label_map(json_path):
    event_types = set()
    with open(json_path, 'r') as f:
        for line in f:
            doc = json.loads(line)
            for event in doc.get("events", []):
                event_types.add(event["type"])
    event_types.add("None")
    return {label: idx for idx, label in enumerate(sorted(event_types))}

# Configurations
model_name = "bert-base-uncased"
batch_size = 4
max_epochs = 5
data_dir = "data"
train_path = os.path.join(data_dir, "train.jsonl")
valid_path = os.path.join(data_dir, "valid.jsonl")

# Build label2id from training data
label2id = extract_label_map(train_path)

# Load datasets
train_set = MAVENDataset(jsonl_path=train_path, tokenizer_name=model_name, label2id=label2id)
val_set = MAVENDataset(jsonl_path=valid_path, tokenizer_name=model_name, label2id=label2id)

# DataLoaders
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size)

# Instantiate model
model = HANetModel(model_name=model_name, num_labels=len(label2id))

# Trainer setup
trainer = pl.Trainer(
    max_epochs=max_epochs,
    accelerator="auto",
    log_every_n_steps=10,
    default_root_dir="lightning_logs"
)

# Train
trainer.fit(model, train_loader, val_loader)

# Save model
trainer.save_checkpoint("hanet_base.ckpt")
print("✅ Base training complete. Model saved to hanet_base.ckpt")
