import json
import argparse
import torch
from models.hanet_model import HANetModel
from transformers import AutoTokenizer
from tqdm import tqdm
import os

def load_test_candidates(path):
    with open(path, 'r') as f:
        return [json.loads(line) for line in f]


def predict(model, tokenizer, data, label2id, device):
    id2label = {v: k for k, v in label2id.items()}
    model.eval()
    results = []

    for doc in tqdm(data):
        content = doc["content"]
        candidates = doc["candidates"]
        pred_list = []

        for mention in candidates:
            sent = content[mention['sent_id']]
            tokens = sent['tokens']
            offset = mention['offset']

            encoding = tokenizer(tokens, is_split_into_words=True, return_tensors="pt", padding='max_length', truncation=True, max_length=128)
            trigger_mask = torch.zeros(128)
            for i in range(offset[0], offset[1]):
                if i < 128:
                    trigger_mask[i] = 1

            with torch.no_grad():
                logits = model(
                    input_ids=encoding['input_ids'].to(device),
                    attention_mask=encoding['attention_mask'].to(device),
                    trigger_mask=trigger_mask.unsqueeze(0).to(device)
                )
                pred_label = torch.argmax(logits, dim=-1).item()
                pred_list.append({"id": mention['id'], "type_id": pred_label})

        results.append({"id": doc['id'], "predictions": pred_list})

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--label_map", type=str, default="data/label2id.json")
    parser.add_argument("--test", type=str, required=True)
    parser.add_argument("--output", type=str, default="res/results.jsonl")
    args = parser.parse_args()

    # Load label map
    with open(args.label_map, 'r') as f:
        label2id = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HANetModel.load_from_checkpoint(args.model, model_name="bert-base-uncased", num_labels=len(label2id)).to(device)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    test_data = load_test_candidates(args.test)
    preds = predict(model, tokenizer, test_data, label2id, device)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        for item in preds:
            f.write(json.dumps(item) + "\n")

    print(f"✅ Saved predictions to {args.output}")
