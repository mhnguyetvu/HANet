import json
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import torch

class MAVENDataset(Dataset):
    """
    PyTorch Dataset for MAVEN-style event detection JSONL files.
    Parses annotated events and generates input for training the HANet model.
    """
    def __init__(self, jsonl_path, tokenizer_name, label2id):
        """
        Args:
            jsonl_path (str): Path to the JSONL file
            tokenizer_name (str): HuggingFace tokenizer name (e.g., 'bert-base-uncased')
            label2id (dict): Mapping from event type string to integer label
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.label2id = label2id
        self.samples = []

        with open(jsonl_path, 'r') as f:
            for line in f:
                doc = json.loads(line)
                content = doc["content"]

                for event in doc.get("events", []):
                    label = self.label2id[event["type"]]
                    for m in event["mention"]:
                        tokens = content[m["sent_id"]]["tokens"]
                        offset = m["offset"]
                        trigger_mask = [0] * len(tokens)
                        for i in range(offset[0], offset[1]):
                            trigger_mask[i] = 1

                        encoding = self.tokenizer(tokens, is_split_into_words=True, return_tensors="pt", padding='max_length', truncation=True, max_length=128)

                        self.samples.append({
                            'input_ids': encoding['input_ids'].squeeze(),
                            'attention_mask': encoding['attention_mask'].squeeze(),
                            'trigger_mask': torch.tensor(trigger_mask + [0] * (128 - len(trigger_mask)))[:128],
                            'labels': torch.tensor(label)
                        })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]