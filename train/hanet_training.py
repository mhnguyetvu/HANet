# hanet_training.py with augmentation & replay (HANet-style)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import json
from collections import defaultdict
import random

# ======= Config =======
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 64
BATCH_SIZE = 16
LR = 2e-5
EPOCHS = 3
REPLAY_PER_CLASS = 1
SIGMA = 0.1  # For prototypical augmentation

# ======= Tokenizer =======
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# ======= Label Setup =======
with open("data/event_types.txt") as f:
    all_event_types = [line.strip() for line in f]
label2id = {etype: i for i, etype in enumerate(all_event_types)}
id2label = {i: t for t, i in label2id.items()}

# ======= Dataset =======
class EventDataset(Dataset):
    def __init__(self, jsonl_file, label2id):
        self.samples = []
        with open(jsonl_file) as f:
            for line in f:
                item = json.loads(line)
                words = item["words"]
                for evt in item.get("gold_evt_links", []):
                    self.samples.append((words, evt["event_type"]))
        self.label2id = label2id

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        words, label = self.samples[idx]
        sent = " ".join(words)
        inputs = tokenizer(sent, padding='max_length', truncation=True, max_length=MAX_LEN, return_tensors='pt')
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "label": torch.tensor(self.label2id[label])
        }

# ======= Model =======
class HANetSimple(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.pooler_output
        return self.classifier(pooled), pooled

# ======= Memory Buffer for Replay =======
def build_exemplar_memory(dataset, per_class=1):
    memory = defaultdict(list)
    for input in dataset:
        label = input["label"].item()
        if len(memory[label]) < per_class:
            memory[label].append(input)
    return memory

# ======= Prototypical Augmentation =======
def augment_feature(rep, num_aug=5, sigma=0.1):
    return [rep + torch.randn_like(rep) * sigma for _ in range(num_aug)]

# ======= Training / Evaluation =======
def train_one_epoch(model, data, optimizer, loss_fn):
    model.train()
    for batch in data:
        input_ids = batch["input_ids"].unsqueeze(0).to(DEVICE)
        attention_mask = batch["attention_mask"].unsqueeze(0).to(DEVICE)
        label = batch["label"].unsqueeze(0).to(DEVICE)

        logits, _ = model(input_ids, attention_mask)
        loss = loss_fn(logits, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def evaluate(model, dataloader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].unsqueeze(0).to(DEVICE)
            attention_mask = batch["attention_mask"].unsqueeze(0).to(DEVICE)
            label = batch["label"].unsqueeze(0).to(DEVICE)

            logits, _ = model(input_ids, attention_mask)
            pred = torch.argmax(logits, dim=1)
            correct += (pred == label).sum().item()
            total += label.size(0)
    return correct / total

# ======= Main Training Script =======
def run_training():
    # Base Task
    base_dataset = EventDataset("data/base_task.jsonl", label2id)
    base_loader = DataLoader(base_dataset, batch_size=1, shuffle=True)

    model = HANetSimple(num_classes=len(label2id)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    print("\n🚀 Training Base Task...")
    for epoch in range(EPOCHS):
        for batch in base_loader:
            train_one_epoch(model, [batch], optimizer, loss_fn)
        acc = evaluate(model, base_loader)
        print(f"Epoch {epoch+1} - Base Task Accuracy: {acc:.4f}")

    # Build memory
    memory = build_exemplar_memory(base_loader.dataset, per_class=REPLAY_PER_CLASS)

    # Incremental Tasks
    for task_id in range(1, 6):
        print(f"\n🧩 Incremental Task {task_id}")
        few_dataset = EventDataset(f"data/incremental_task_{task_id}.jsonl", label2id)
        few_loader = DataLoader(few_dataset, batch_size=1, shuffle=True)

        train_batch = []
        for few in few_loader:
            input_ids = few["input_ids"].unsqueeze(0).to(DEVICE)
            attention_mask = few["attention_mask"].unsqueeze(0).to(DEVICE)
            label = few["label"].unsqueeze(0).to(DEVICE)

            # Get prototype feature
            _, rep = model(input_ids, attention_mask)
            reps = augment_feature(rep.squeeze(0), num_aug=5, sigma=SIGMA)

            # Create synthetic inputs
            for r in reps:
                train_batch.append({
                    "input_ids": torch.zeros_like(input_ids[0]),  # dummy input
                    "attention_mask": torch.ones_like(attention_mask[0]),
                    "label": label.squeeze(0),
                    "aug_feature": r.detach()
                })

        # Add replay
        for cls, samples in memory.items():
            for ex in samples:
                train_batch.append(ex)

        # Replace model forward to accept direct features for augmented ones
        original_forward = model.forward
        def mix_forward(input_ids, attention_mask):
            if input_ids.sum() == 0:
                return model.classifier(attention_mask.unsqueeze(0)), attention_mask.unsqueeze(0)
            else:
                return original_forward(input_ids, attention_mask)
        model.forward = mix_forward

        for epoch in range(3):
            for b in train_batch:
                input_ids = b.get("input_ids", torch.zeros(MAX_LEN)).unsqueeze(0).to(DEVICE)
                attention_mask = b.get("attention_mask", torch.ones(MAX_LEN)).unsqueeze(0).to(DEVICE)
                label = b["label"].unsqueeze(0).to(DEVICE)
                logits, _ = model(input_ids, attention_mask)
                loss = loss_fn(logits, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        acc = evaluate(model, few_loader)
        print(f"🎯 Task {task_id} Accuracy: {acc:.4f}")

if __name__ == "__main__":
    run_training()