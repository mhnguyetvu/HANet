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
warnings.filterwarnings("ignore")

# Thiết lập thiết bị
config.DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Khởi tạo tokenizer
tokenizer = BertTokenizer.from_pretrained(config.BASE_MODEL)

# Load mô hình từ checkpoint đã huấn luyện base_task
model = HANetTriggerEncoder(config.BASE_MODEL, config.NUM_LABELS).to(config.DEVICE)
model.load_state_dict(torch.load("/data/AITeam/nguyetnvm/Hanet/checkpoints/hanet_best_base.pt"))
print("✅ Loaded base model checkpoint")

# Định nghĩa danh sách các incremental task
incremental_tasks = ["incr_1_task.jsonl", "incr_2_task.jsonl"]

# ==== Tạo label_map động để ánh xạ event type → index ====
label_map = {}
def get_label_id(event_type):
    if event_type not in label_map:
        label_map[event_type] = len(label_map)
    return label_map[event_type]

# ==== Cấu trúc Memory Bank ==== 
memory_bank = {}  # class_id → list of exemplar dict
MAX_MEMORY_PER_CLASS = 1

# ==== Hàm chọn exemplar tốt nhất (hiện tại chọn ngẫu nhiên) ====
def select_exemplars(class_id, candidates):
    return candidates[:MAX_MEMORY_PER_CLASS]  # simple truncation

# Huấn luyện cho từng incremental task
for task_file in incremental_tasks:
    print(f"\n🚀 Training on {task_file}...")

    # Load dataset và ánh xạ nhãn động
    raw_data = []
    with open(f"/data/AITeam/nguyetnvm/Hanet/data/small_split/{task_file}", "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line.strip())
            for event in doc["events"]:
                mention = event["mention"][0]
                sentence = doc["content"][mention["sent_id"]]
                input_text = sentence["sentence"]
                tokens = tokenizer(input_text, truncation=True, padding="max_length", return_tensors="pt")
                label_id = get_label_id(event["type"])
                example = {
                    "input_ids": tokens["input_ids"].squeeze(),
                    "attention_mask": tokens["attention_mask"].squeeze(),
                    "trigger_pos": torch.tensor(mention["offset"]),
                    "label": torch.tensor(label_id)
                }
                raw_data.append(example)
                # Lưu vào memory candidate
                memory_bank.setdefault(label_id, []).append(example)

    # === Thêm exemplar từ memory vào dữ liệu ===
    replay_data = []
    for cls_id, exemplars in memory_bank.items():
        replay_data.extend(select_exemplars(cls_id, exemplars))

    all_data = raw_data + replay_data

    # Dataloader từ list
    class InlineDataset(torch.utils.data.Dataset):
        def __init__(self, data):
            self.data = data
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            return self.data[idx]

    dataloader = DataLoader(InlineDataset(raw_data), batch_size=config.BATCH_SIZE, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=config.LR)
    model.train()

    for epoch in range(config.MAX_EPOCHS):
        for batch in dataloader:
            input_ids = batch['input_ids'].to(config.DEVICE)
            attention_mask = batch['attention_mask'].to(config.DEVICE)
            labels = batch['label'].to(config.DEVICE)
            trigger_pos = batch['trigger_pos'].to(config.DEVICE)

            logits = model(input_ids, attention_mask, trigger_pos)
            loss = nn.CrossEntropyLoss()(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")

    # Lưu checkpoint sau mỗi incremental task
    checkpoint_path = f"/data/AITeam/nguyetnvm/Hanet/checkpoints/hanet_{task_file.replace('.jsonl', '')}.pt"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"💾 Saved model after {task_file} to {checkpoint_path}")

# === Lưu label_map sau toàn bộ training ===
label_map_path = "/data/AITeam/nguyetnvm/Hanet/checkpoints/label_map.json"
with open(label_map_path, "w") as f:
    json.dump(label_map, f)
print(f"💾 Saved label map to {label_map_path}")
