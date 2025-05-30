import json
import random
from collections import defaultdict
from pathlib import Path

def split_base_and_incremental(train_path, save_dir, base_classes=4, shot=5, seed=42):
    """
    Split train.jsonl into base task and multiple incremental tasks as in HANet.
    Args:
        train_path (str): Path to the full train.jsonl
        save_dir (str): Where to save base_task.jsonl and incremental_task_*.jsonl
        base_classes (int): Number of event types for base task
        shot (int): Number of samples per class for incremental
        seed (int): Random seed for reproducibility
    """
    random.seed(seed)
    with open(train_path, 'r') as f:
        docs = [json.loads(line) for line in f]

    # Group all mentions by event type
    type_to_mentions = defaultdict(list)
    for doc in docs:
        for ev in doc.get("events", []):
            type_to_mentions[ev["type"]].append((doc, ev))

    all_types = sorted(type_to_mentions.keys())
    base_types = all_types[:base_classes]
    inc_types = all_types[base_classes:]

    # Build base task
    base_docs = []
    for doc in docs:
        base_events = [ev for ev in doc.get("events", []) if ev["type"] in base_types]
        if base_events:
            base_docs.append({
                "id": doc["id"],
                "title": doc["title"],
                "content": doc["content"],
                "events": base_events,
                "negative_triggers": doc.get("negative_triggers", [])
            })

    # Build incremental tasks
    inc_tasks = []
    per_task = 3  # 3 event types per incremental task like in HANet
    for i in range(0, len(inc_types), per_task):
        types = inc_types[i:i+per_task]
        inc_task = []
        for t in types:
            samples = random.sample(type_to_mentions[t], min(shot, len(type_to_mentions[t])))
            for doc, ev in samples:
                inc_task.append({
                    "id": doc["id"],
                    "title": doc["title"],
                    "content": doc["content"],
                    "events": [ev],
                    "negative_triggers": doc.get("negative_triggers", [])
                })
        inc_tasks.append(inc_task)

    # Save to disk
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{save_dir}/base_task.jsonl", "w") as f:
        for ex in base_docs:
            f.write(json.dumps(ex) + "\n")

    for i, task in enumerate(inc_tasks):
        with open(f"{save_dir}/incremental_task_{i+1}.jsonl", "w") as f:
            for ex in task:
                f.write(json.dumps(ex) + "\n")

    print(f"✅ Created base_task.jsonl with {len(base_docs)} examples")
    print(f"✅ Created {len(inc_tasks)} incremental tasks")
if __name__ == "__main__":
    split_base_and_incremental(
        train_path="data/train.jsonl",
        save_dir="data",
        base_classes=4,
        shot=5,
        seed=42
    )
