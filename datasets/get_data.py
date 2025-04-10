import json
from collections import defaultdict
import os

# Step 1: Load hand-crafted train.jsonl (3 documents)
train_data = [
    {
        "content": [
            {"tokens": ["Alice", "and", "Bob", "got", "married", "in", "Hawaii"]},
            {"tokens": ["They", "met", "during", "a", "conference", "in", "Paris"]}
        ],
        "events": [
            {
                "type": "Life:Marry",
                "mention": [
                    {"trigger_word": "married", "sent_id": 0, "offset": 4}
                ]
            }
        ]
    },
    {
        "content": [
            {"tokens": ["John", "traveled", "to", "Germany", "by", "plane"]},
            {"tokens": ["He", "visited", "Berlin", "and", "Munich"]}
        ],
        "events": [
            {
                "type": "Movement:Transport",
                "mention": [
                    {"trigger_word": "traveled", "sent_id": 0, "offset": 1}
                ]
            }
        ]
    },
    {
        "content": [
            {"tokens": ["An", "explosion", "occurred", "in", "Baghdad"]},
            {"tokens": ["Several", "people", "were", "injured", "in", "the", "attack"]}
        ],
        "events": [
            {
                "type": "Conflict:Attack",
                "mention": [
                    {"trigger_word": "explosion", "sent_id": 0, "offset": 1},
                    {"trigger_word": "attack", "sent_id": 1, "offset": 6}
                ]
            }
        ]
    }
]

# Step 2: Flatten sentences and associate triggers (to EventDataset format)
flattened = []
for doc in train_data:
    content = doc["content"]
    events = doc["events"]

    sent2events = defaultdict(list)
    for event in events:
        for mention in event["mention"]:
            sent_id = mention["sent_id"]
            sent2events[sent_id].append({
                "event_type": event["type"],
                "trigger": mention["offset"]
            })

    for sent_id, sent in enumerate(content):
        words = sent["tokens"]
        evt_links = []
        for evt in sent2events.get(sent_id, []):
            evt_links.append({
                "event_type": evt["event_type"],
                "trigger": evt["trigger"]
            })
        if evt_links:
            flattened.append({
                "words": words,
                "gold_evt_links": evt_links
            })

# Step 3: Group by event type
event2examples = defaultdict(list)
for item in flattened:
    for evt in item["gold_evt_links"]:
        event2examples[evt["event_type"]].append(item)

# Step 4: Split to base (2 types) and incremental (1 type, 2-shot)
all_event_types = list(event2examples.keys())
base_types = all_event_types[:2]
incre_type = all_event_types[2]

# Base task
base_data = []
for t in base_types:
    base_data.extend(event2examples[t])

# Incremental task 1: only 2-shot from the remaining type
incre_data = event2examples[incre_type][:2]

# Save all files
output_dir = "/workspaces/HANet/datasets/hanet_minimal"
os.makedirs(output_dir, exist_ok=True)

with open(os.path.join(output_dir, "train.jsonl"), "w") as f:
    for doc in train_data:
        f.write(json.dumps(doc) + "\n")

with open(os.path.join(output_dir, "base_task.jsonl"), "w") as f:
    for item in base_data:
        f.write(json.dumps(item) + "\n")

with open(os.path.join(output_dir, "incremental_task_1.jsonl"), "w") as f:
    for item in incre_data:
        f.write(json.dumps(item) + "\n")

with open(os.path.join(output_dir, "event_types.txt"), "w") as f:
    for etype in all_event_types:
        f.write(etype + "\n")

output_dir  # Return path to zip or files
