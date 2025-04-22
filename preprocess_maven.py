import json
import os
import random
from collections import defaultdict
# === CONFIG ===
INPUT_PATH = "/workspaces/HANet/data/train.jsonl"   # input gốc từ MAVEN
OUTPUT_DIR = "/workspaces/HANet/data/small_split"  # nơi lưu output
os.makedirs(OUTPUT_DIR, exist_ok=True)

TOTAL_CLASSES = 10                # tổng số lớp muốn chọn
BASE_CLASSES = 4                 # số lớp cho base task
INC_TASKS = 2                    # số incremental tasks
EX_BASE = 20                     # ví dụ/lớp cho base
EX_INC = 5                       # ví dụ/lớp cho incremental

# === STEP 1: Load data ===
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    all_data = [json.loads(line.strip()) for line in f]

# === STEP 2: Group documents by event type ===
event2examples = defaultdict(list)
for doc in all_data:
    types = {ev["type"] for ev in doc.get("events", [])}
    if types:
        main_type = sorted(types)[0]  # dùng event type đầu tiên
        event2examples[main_type].append(doc)

# === STEP 3: Chọn lớp đa dạng nhưng phổ biến nhất ===
event_counts = {etype: len(exs) for etype, exs in event2examples.items()}
sorted_event_types = sorted(event_counts.items(), key=lambda x: -x[1])
selected_classes = [etype for etype, count in sorted_event_types if count >= EX_BASE][:TOTAL_CLASSES]

print(f"🎯 Selected {len(selected_classes)} event types:", selected_classes)

# === STEP 4: Chia thành các task ===
base_classes = selected_classes[:BASE_CLASSES]
inc_chunks = [
    selected_classes[BASE_CLASSES + i*3 : BASE_CLASSES + (i+1)*3]
    for i in range(INC_TASKS)
]

task_data = {"base": []}
for i in range(INC_TASKS):
    task_data[f"incr_{i+1}"] = []

# Helper để lấy mẫu theo lớp
def sample_examples(classes, k):
    examples = []
    used_ids = set()
    for cls in classes:
        pool = [ex for ex in event2examples[cls] if ex["id"] not in used_ids]
        sampled = random.sample(pool, min(len(pool), k))
        examples.extend(sampled)
        used_ids.update(ex["id"] for ex in sampled)
    return examples

# Gán example vào từng task
task_data["base"] = sample_examples(base_classes, EX_BASE)
for i, cls in enumerate(inc_chunks):
    task_data[f"incr_{i+1}"] = sample_examples(cls, EX_INC)

# === STEP 5: Save .jsonl files ===
def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

for task, examples in task_data.items():
    path = os.path.join(OUTPUT_DIR, f"{task}_task.jsonl")
    save_jsonl(examples, path)
    print(f"✅ Saved {task} ({len(examples)} samples) to {path}")
