import json
from collections import defaultdict
import os

# Load existing train.jsonl (the one with 3 events)
with open("/workspaces/HANet/datasets/hanet_minimal/train.jsonl") as f:
    documents = [json.loads(line) for line in f]

# Flatten into EventDataset format
flattened = []
for doc in documents:
    content = doc["content"]
    events = doc.get("events", [])
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

# Group examples by event type
event2examples = defaultdict(list)
for item in flattened:
    for evt in item["gold_evt_links"]:
        event2examples[evt["event_type"]].append(item)

# Choose 1 example for task 2 and 1 for task 3
task_2_type = "Life:Marry"
task_3_type = "Movement:Transport"

task_2_data = event2examples[task_2_type][1:]  # avoid duplicate of base
task_3_data = event2examples[task_3_type][1:]

# Save to JSONL
output_dir = "/workspaces/HANet/datasets/hanet_minimal"
task_2_path = os.path.join(output_dir, "incremental_task_2.jsonl")
task_3_path = os.path.join(output_dir, "incremental_task_3.jsonl")

with open(task_2_path, "w") as f2:
    for item in task_2_data:
        f2.write(json.dumps(item) + "\n")

with open(task_3_path, "w") as f3:
    for item in task_3_data:
        f3.write(json.dumps(item) + "\n")

task_2_path, task_3_path
