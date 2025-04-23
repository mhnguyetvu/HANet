import json
import torch
from torch.utils.data import Dataset

class MAVENDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            self.data = [json.loads(line.strip()) for line in f]

        self.tokenizer = tokenizer

        # Load 10 base classes
        with open("/data/AITeam/nguyetnvm/Hanet/data/hanet_benchmark/event_type.txt", "r") as f:
            all_classes = [line.strip() for line in f.readlines()]
        self.base_classes = all_classes[:10]
        self.label2id = {et: idx for idx, et in enumerate(self.base_classes)}

        self.samples = []
        for doc in self.data:
            content = doc["content"]  # list of sentences
            for event in doc.get("events", []):
                event_type = event["type"]
                if event_type not in self.label2id:
                    continue
                label = self.label2id[event_type]

                for mention in event["mention"]:
                    sent_id = mention["sent_id"]
                    offset = mention["offset"]
                    tokens = content[sent_id]["tokens"]

                    # Tokenize toàn bộ câu
                    encoded = tokenizer(" ".join(tokens), return_tensors="pt", truncation=True, padding="max_length", max_length=512)
                    input_ids = encoded["input_ids"].squeeze()
                    attention_mask = encoded["attention_mask"].squeeze()

                    self.samples.append({
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "trigger_pos": torch.tensor(offset),
                        "label": torch.tensor(label)
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
