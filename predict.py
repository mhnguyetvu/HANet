# === predict.py ===
import torch
from transformers import BertTokenizer
from models.hanet_model import HANetTriggerEncoder
import config
import json 

# Load label_map
with open("/data/AITeam/nguyetnvm/Hanet/checkpoints/label_map.json") as f:
    label_map = json.load(f)
reverse_label_map = {v: k for k, v in label_map.items()}

# Khởi tạo thiết bị và mô hình
config.DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
tokenizer = BertTokenizer.from_pretrained(config.BASE_MODEL)

model = HANetTriggerEncoder(config.BASE_MODEL, config.NUM_LABELS).to(config.DEVICE)
model.load_state_dict(torch.load("/data/AITeam/nguyetnvm/Hanet/checkpoints/hanet_incr_2_task.pt"))
model.eval()

# ==== Ví dụ câu đầu vào ====
text = "The rebels organized a large protest in the capital."
trigger_offset = [2, 3]  # ví dụ từ 'organized'

# Tokenize
inputs = tokenizer(text, truncation=True, padding="max_length", return_tensors="pt")
input_ids = inputs["input_ids"].to(config.DEVICE)
attention_mask = inputs["attention_mask"].to(config.DEVICE)
trigger_pos = torch.tensor([trigger_offset]).to(config.DEVICE)

# Dự đoán
with torch.no_grad():
    logits = model(input_ids, attention_mask, trigger_pos)
    pred = torch.argmax(logits, dim=-1).item()

# Ánh xạ ngược label_map (bạn phải lưu label_map từ training nếu chưa có)
reverse_label_map = {v: k for k, v in label_map.items()}
event_type = reverse_label_map.get(pred, "UNKNOWN")

print(f"🔍 Trigger: '{text.split()[trigger_offset[0]]}' → Predicted event type: {event_type}")
