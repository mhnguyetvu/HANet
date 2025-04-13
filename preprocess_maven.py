import json
import os
import random
from collections import defaultdict

# 🧷 Config
INPUT_PATH = "/workspaces/HANet/data/base_task_random_1000.jsonl"
OUTPUT_DIR = "/workspaces/HANet/data"
NUM_BASE_CLASSES = 10
NUM_INC_TASKS = 5

random.seed(42)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Step 1: Load all data
print("📥 Loading data...")
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    all_data = [json.loads(line) for line in f]

# Step 2: Group by event_type
event_types_set = set()
event2examples = defaultdict(list)

for example in all_data:
    added_types = set()
    for evt in example.get("gold_evt_links", []):
        event_type = evt["event_type"]
        event_types_set.add(event_type)  # ✅ FIXED: Add to set for saving later
        if event_type not in added_types:
            event2examples[event_type].append(example)
            added_types.add(event_type)
# Sort for consistency
event_types = sorted(event_types_set)
random.shuffle(event_types)

# Step 3: Split classes
base_classes = event_types[:NUM_BASE_CLASSES]
inc_classes = event_types[NUM_BASE_CLASSES:]
classes_per_task = len(inc_classes) // NUM_INC_TASKS

# Step 4: Create base task
base_data = []
for cls in base_classes:
    base_data.extend(event2examples[cls])

base_path = os.path.join(OUTPUT_DIR, "base_task.jsonl")
with open(base_path, "w", encoding="utf-8") as f:
    for item in base_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
print(f"✅ Base task saved: {base_path} ({len(base_data)} examples)")

# Step 5: Create incremental tasks
for i in range(NUM_INC_TASKS):
    start = i * classes_per_task
    end = (i + 1) * classes_per_task
    inc_classes_i = inc_classes[start:end]

    task_data = []
    for cls in inc_classes_i:
        task_data.extend(event2examples[cls])

    task_path = os.path.join(OUTPUT_DIR, f"incremental_task_{i+1}.jsonl")
    with open(task_path, "w", encoding="utf-8") as f:
        for item in task_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"🧩 Incremental task {i+1} saved: {task_path} ({len(task_data)} examples)")
# Step 6: Save all event types
event_types_set = set()

# Extract from all_data which you loaded earlier
event2examples = defaultdict(list)

for example in all_data:
    added_types = set()
    for evt in example.get("gold_evt_links", []):
        event_type = evt["event_type"]
        event_types_set.add(event_type)  # ✅ FIXED: Add to set for saving later
        if event_type not in added_types:
            event2examples[event_type].append(example)
            added_types.add(event_type)

# Sort for consistency
event_types = sorted(event_types_set)

# Save to file
event_type_path = os.path.join(OUTPUT_DIR, "event_type.txt")
with open(event_type_path, "w", encoding="utf-8") as f:
    for evt in event_types:
        f.write(evt + "\n")

print(f"📄 Event types saved to: {event_type_path} ({len(event_types)} classes)")
