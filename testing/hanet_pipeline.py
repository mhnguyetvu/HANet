import os
import json
import torch
import random
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import torch.nn as nn

# ======= Config =======
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 64
BATCH_SIZE = 16
LR = 2e-5
EPOCHS = 3
SIGMA = 0.1

# ======= Tokenizer =======
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# ======= Label Setup =======
event_types = ["Life:Marry", "Movement:Transport", "Conflict:Attack"]
label2id = {etype: i for i, etype in enumerate(event_types)}
id2label = {i: t for t, i in label2id.items()}

# ======= Dataset =======
class EventDataset(Dataset):
    def __init__(self, jsonl_file, label2id, tokenizer, max_len=64):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        with open(jsonl_file) as f:
            for line in f:
                item = json.loads(line)
                words = item["words"]
                for evt in item.get("gold_evt_links", []):
                    label = evt["event_type"]
                    if label in label2id:
                        self.samples.append((words, label))
        self.label2id = label2id

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        words, label = self.samples[idx]
        sent = " ".join(words)
        inputs = self.tokenizer(sent, padding='max_length', truncation=True, max_length=self.max_len, return_tensors='pt')
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "label": torch.tensor(self.label2id[label], dtype=torch.long)
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

# ======= Evaluate =======
def evaluate(model, dataloader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            logits, _ = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0.0

# ======= Augment + Replay =======
def augment_feature(rep, num_aug=5, sigma=0.1):
    return [rep + torch.randn_like(rep) * sigma for _ in range(num_aug)]

def build_exemplar_memory(dataset, per_class=1):
    memory = defaultdict(list)
    for item in dataset:
        label = item["label"].item()
        if len(memory[label]) < per_class:
            memory[label].append(item)
    return memory

# ======= Train base =======
def train_base(base_path):
    print("\n🚀 Training Base Task...")
    dataset = EventDataset(base_path, label2id, tokenizer, MAX_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = HANetSimple(num_classes=len(label2id)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        for batch in dataloader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            logits, _ = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        acc = evaluate(model, dataloader)
        print(f"Epoch {epoch+1} - Accuracy: {acc:.4f}")
    return model, dataset

# ======= Train incremental =======
def train_incremental(model, fewshot_path, memory):
    print(f"\n🧩 Incremental Task from {os.path.basename(fewshot_path)}")
    few_dataset = EventDataset(fewshot_path, label2id, tokenizer, MAX_LEN)

    if len(few_dataset) == 0:
        print(f"⚠️ No data found in {fewshot_path}, skipping this task.")
        return model

    few_loader = DataLoader(few_dataset, batch_size=1, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    train_batch = []
    for few in few_loader:
        input_ids = few["input_ids"].to(DEVICE)
        attention_mask = few["attention_mask"].to(DEVICE)
        label = few["label"].to(DEVICE)
        _, rep = model(input_ids, attention_mask)
        reps = augment_feature(rep.squeeze(0), num_aug=5, sigma=SIGMA)
        for r in reps:
            train_batch.append({"label": label.item(), "aug_feature": r.detach()})
    for cls, samples in memory.items():
        for ex in samples:
            train_batch.append(ex)

    for epoch in range(EPOCHS):
        model.train()
        for b in train_batch:
            label = b["label"]
            if isinstance(label, torch.Tensor):
                label = label.clone().detach().item()
            label = torch.tensor(label, dtype=torch.long).unsqueeze(0).to(DEVICE)

            if "aug_feature" in b:
                feature = b["aug_feature"].unsqueeze(0).float().to(DEVICE)
                logits = model.classifier(feature)
            else:
                input_ids = b["input_ids"].unsqueeze(0).to(DEVICE)
                attention_mask = b["attention_mask"].unsqueeze(0).to(DEVICE)
                logits, _ = model(input_ids, attention_mask)

            loss = loss_fn(logits, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    acc = evaluate(model, few_loader)
    print(f"🎯 Task Accuracy: {acc:.4f}")
    return model


# Step 1: Train base task
model, base_dataset = train_base("/workspaces/HANet/datasets/hanet_minimal/base_task.jsonl")

# Step 2: Build memory replay buffer
memory = build_exemplar_memory(base_dataset, per_class=1)

# Step 3: Train incremental tasks
for task_id in range(1, 4):
    path = f"//workspaces/HANet/datasets/hanet_minimal/incremental_task_{task_id}.jsonl"
    model = train_incremental(model, path, memory)