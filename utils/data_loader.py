import json
import torch
from torch.utils.data import Dataset

class MAVENDataset(Dataset):
    def __init__(self, jsonl_file, tokenizer):
        self.data = []
        with open(jsonl_file, 'r') as f:
            for line in f:
                doc = json.loads(line)
                for event in doc["events"]:
                    mention = event["mention"][0]
                    sentence = doc["content"][mention["sent_id"]]
                    input_text = sentence["sentence"]
                    tokens = tokenizer(input_text, truncation=True, padding="max_length", return_tensors="pt")
                    self.data.append({
                        "input_ids": tokens["input_ids"].squeeze(),
                        "attention_mask": tokens["attention_mask"].squeeze(),
                        "trigger_pos": torch.tensor(mention["offset"]),  # Chuyển sẵn offset thành tensor
                        "label": torch.tensor(event["type_id"])  # Đảm bảo label cũng là tensor
                    })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]