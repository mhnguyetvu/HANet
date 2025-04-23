# === train_incremental.py ===
import torch
import torch.nn as nn
from transformers import BertTokenizer, AdamW
from models.hanet_model import HANetTriggerEncoder
from utils.data_loader import MAVENDataset
from torch.utils.data import DataLoader
import config
import os
import json
import warnings
from utils.prototypical_augmenter import PrototypicalAugmenter
from utils.constrastive_augmenter import ContrastiveAugmenter

warnings.filterwarnings("ignore")

# Thiết lập thiết bị
config.DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Tokenizer
tokenizer = BertTokenizer.from_pretrained(config.BASE_MODEL)

# Load mô hình từ checkpoint đã huấn luyện base task
model = HANetTriggerEncoder(config.BASE_MODEL, config.NUM_LABELS).to(config.DEVICE)
model.load_state_dict(torch.load("/data/AITeam/nguyetnvm/Hanet/checkpoints/hanet_best_base.pt"))
print("✅ Loaded base model checkpoint")

# Gắn hàm extract_trigger_embedding vào model nếu chưa có
if not hasattr(model, 'extract_trigger_embedding'):
    def extract_trigger_embedding(input_ids, attention_mask, trigger_pos):
        outputs = model.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        start_emb = hidden_states[torch.arange(input_ids.size(0)), trigger_pos[:, 0]]
        end_emb = hidden_states[torch.arange(input_ids.size(0)), trigger_pos[:, 1] - 1]
        return torch.cat([start_emb, end_emb], dim=-1)
    model.extract_trigger_embedding = extract_trigger_embedding

augmenter = PrototypicalAugmenter(model, model.extract_trigger_embedding, config.DEVICE)
contrastive_aug = ContrastiveAugmenter(tokenizer, model, model.extract_trigger_embedding, config.DEVICE)

# Danh sách các incremental task
incremental_tasks = ["incr_1_task.jsonl", "incr_2_task.jsonl", "incr_3_task.jsonl"]

label_map = {}
def get_label_id(event_type):
    if event_type not in label_map:
        label_map[event_type] = len(label_map)
    return label_map[event_type]

memory_bank = {}
MAX_MEMORY_PER_CLASS = 1
def select_exemplars(class_id, candidates):
    return candidates[:MAX_MEMORY_PER_CLASS]

for task_file in incremental_tasks:
    print(f"\n🚀 Training on {task_file}...")
    raw_data = []

    task_path = f"/data/AITeam/nguyetnvm/Hanet/data/hanet_benchmark/{task_file}"
    with open(task_path, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line.strip())
            content = doc["content"]
            for event in doc.get("events", []):
                event_type = event["type"]
                label_id = get_label_id(event_type)

                for mention in event["mention"]:
                    sent_id = mention["sent_id"]
                    offset = mention["offset"]
                    tokens = content[sent_id]["tokens"]
                    sentence = content[sent_id]["sentence"]
                    encoded = tokenizer(" ".join(tokens), return_tensors="pt", truncation=True, padding="max_length", max_length=512)

                    example = {
                        "input_ids": encoded["input_ids"].squeeze(),
                        "attention_mask": encoded["attention_mask"].squeeze(),
                        "trigger_pos": torch.tensor(offset),
                        "label": torch.tensor(label_id),
                        "sentence": sentence,
                        "offset": offset
                    }
                    raw_data.append(example)
                    memory_bank.setdefault(label_id, []).append(example)

    num_classes = len(label_map)
    old_num_classes = model.classifier.out_features
    if old_num_classes < num_classes:
        print(f"⚙️ Expanding classifier: {old_num_classes} → {num_classes}")
        old_classifier = model.classifier
        new_classifier = nn.Linear(old_classifier.in_features, num_classes).to(config.DEVICE)
        with torch.no_grad():
            new_classifier.weight[:old_num_classes] = old_classifier.weight
            new_classifier.bias[:old_num_classes] = old_classifier.bias
        model.classifier = new_classifier

    replay_data = []
    for cls_id, exemplars in memory_bank.items():
        replay_data.extend(select_exemplars(cls_id, exemplars))

    proto_augments = []
    for cls_id, exemplars in memory_bank.items():
        proto_augments.extend(augmenter.generate_augments(exemplars, cls_id))

    all_data = raw_data + replay_data

    class InlineDataset(torch.utils.data.Dataset):
        def __init__(self, data):
            self.data = data
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            return self.data[idx]

    dataloader = DataLoader(InlineDataset(all_data), batch_size=config.BATCH_SIZE, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=config.LR)
    model.train()

    for epoch in range(config.MAX_EPOCHS):
        for batch in dataloader:
            input_ids = batch['input_ids'].to(config.DEVICE)
            attention_mask = batch['attention_mask'].to(config.DEVICE)
            trigger_pos = batch['trigger_pos'].to(config.DEVICE)
            labels = batch['label'].to(config.DEVICE)

            logits = model(input_ids, attention_mask, trigger_pos)
            loss = nn.CrossEntropyLoss()(logits, labels)

            for proto in proto_augments:
                pred = model.classifier(proto['embedding'].unsqueeze(0))
                loss += 0.1 * nn.CrossEntropyLoss()(pred, proto['label'].unsqueeze(0).to(config.DEVICE))

            # === Contrastive Loss ===
            if 'sentence' in batch and 'trigger_pos' in batch:
                try:
                    sentence = batch['sentence'][0]  # first item in batch
                    offset = batch['trigger_pos'][0].tolist()
                    z1, z2 = contrastive_aug.generate_contrastive_pairs(sentence, offset)
                    loss += 0.1 * contrastive_aug.contrastive_loss(z1, z2)
                except Exception as e:
                    print("⚠️ Contrastive loss error:", e)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")

    ckpt_path = f"/data/AITeam/nguyetnvm/Hanet/checkpoints/hanet_{task_file.replace('.jsonl', '')}.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"💾 Saved model after {task_file} to {ckpt_path}")

with open("/data/AITeam/nguyetnvm/Hanet/checkpoints/label_map.json", "w") as f:
    json.dump(label_map, f)
print("✅ Saved label map.")