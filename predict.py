# === predict.py ===
import torch
from transformers import BertTokenizer
from models.hanet_model import HANetTriggerEncoder
import json
import os
from tqdm import tqdm
import config

# === Load label map ===
with open("/data/AITeam/nguyetnvm/Hanet/checkpoints/label_map.json", "r") as f:
    label_map = json.load(f)
id2label = {v: k for k, v in label_map.items()}

# === Load mô hình ===
model = HANetTriggerEncoder(config.BASE_MODEL, num_labels=len(label_map)).to(config.DEVICE)
model.load_state_dict(torch.load("/data/AITeam/nguyetnvm/Hanet/checkpoints/hanet_incr_3_task.pt"))
model.eval()

# === Load tokenizer ===
tokenizer = BertTokenizer.from_pretrained(config.BASE_MODEL)

# === Load dữ liệu valid
input_path = "/data/AITeam/nguyetnvm/Hanet/data/valid.jsonl"
output_path = "/data/AITeam/nguyetnvm/Hanet/predictions/predict_valid.jsonl"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

results = []

with open(input_path, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="🔍 Predicting"):
        doc = json.loads(line.strip())
        content = doc["content"]

        for event in doc.get("events", []):
            for mention in event["mention"]:
                sent_id = mention["sent_id"]
                offset = mention["offset"]
                tokens = content[sent_id]["tokens"]

                encoded = tokenizer(" ".join(tokens), return_tensors="pt", truncation=True, padding="max_length", max_length=512)
                input_ids = encoded["input_ids"].to(config.DEVICE)
                attention_mask = encoded["attention_mask"].to(config.DEVICE)
                trigger_pos = torch.tensor([offset], device=config.DEVICE)

                with torch.no_grad():
                    logits = model(input_ids, attention_mask, trigger_pos)
                    pred_id = logits.argmax(dim=-1).item()
                    pred_label = id2label[pred_id]

                results.append({
                    "doc_id": doc["id"],
                    "trigger_word": mention["trigger_word"],
                    "pred_event_type": pred_label,
                    "true_event_type": event["type"],
                    "offset": offset,
                    "sent_id": sent_id
                })

# === Lưu kết quả
with open(output_path, "w", encoding="utf-8") as f:
    for item in results:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"✅ Saved predictions to: {output_path}")


