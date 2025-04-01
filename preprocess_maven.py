import json
import os
import random
from collections import defaultdict

# === Cấu hình ===
INPUT_PATH = "/workspaces/HANet/datasets/train.jsonl"  # ← đường dẫn đến file gốc
OUTPUT_DIR = "data"  # ← nơi sẽ lưu file output
BASE_TYPE_COUNT = 50
INCREMENTAL_TASKS = 5
SHOT_PER_TYPE = 5


random.seed(42)
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(INPUT_PATH, "r") as f:
    documents = [json.loads(line) for line in f]

flattened = []

for doc in documents:
    content = doc["content"]
    events = doc.get("events", [])

    # Map sentence_id to list of events
    sent2events = defaultdict(list)
    for event in events:
        for mention in event["mention"]:
            sent_id = mention["sent_id"]
            sent2events[sent_id].append({
                "event_type": event["type"],
                "trigger": mention["trigger_word"],
                "offset": mention["offset"]
            })

    for sent_id, sent in enumerate(content):
        words = sent["tokens"]
        evt_links = []

        for evt in sent2events.get(sent_id, []):
            evt_links.append({
                "event_type": evt["event_type"],
                "trigger": evt["offset"]
            })

        if evt_links:
            flattened.append({
                "words": words,
                "gold_evt_links": evt_links
            })

# Step 2: group by event type
event2examples = defaultdict(list)
for item in flattened:
    for evt in item["gold_evt_links"]:
        event2examples[evt["event_type"]].append(item)

all_event_types = sorted(event2examples.keys())
random.shuffle(all_event_types)

with open(os.path.join(OUTPUT_DIR, "event_types.txt"), "w") as f:
    for etype in all_event_types:
        f.write(etype + "\n")

# Step 3: split base & incremental
base_types = all_event_types[:BASE_TYPE_COUNT]
incre_groups = [all_event_types[i:i+10] for i in range(BASE_TYPE_COUNT, BASE_TYPE_COUNT + 10*INCREMENTAL_TASKS, 10)]

base_data = []
for t in base_types:
    base_data.extend(event2examples[t])
with open(os.path.join(OUTPUT_DIR, "base_task.jsonl"), "w") as f:
    for item in base_data:
        f.write(json.dumps(item) + "\n")
print(f"✅ Base task: {len(base_data)} examples from {len(base_types)} types.")

for i, group in enumerate(incre_groups):
    few_data = []
    for t in group:
        few_data.extend(event2examples[t][:SHOT_PER_TYPE])
    with open(os.path.join(OUTPUT_DIR, f"incremental_task_{i+1}.jsonl"), "w") as f:
        for item in few_data:
            f.write(json.dumps(item) + "\n")
    print(f"✅ Incremental task {i+1}: {len(few_data)} examples from {len(group)} types.")