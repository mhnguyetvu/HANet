import json
import os
import random
from collections import defaultdict

# === CONFIG ===
INPUT_PATH = "/data/AITeam/nguyetnvm/Hanet/data/train.jsonl"
OUTPUT_DIR = "/data/AITeam/nguyetnvm/Hanet/data/hanet_benchmark"
TOTAL_CLASSES = 40        # tổng số class sử dụng
BASE_CLASSES = 10         # số class trong base task
INC_TASKS = 5             # số incremental tasks
INC_CLASSES_PER_TASK = 6  # mỗi task incremental có m lớp
BASE_SHOT = 100           # số ví dụ mỗi lớp trong base
INC_SHOT = 5              # số ví dụ mỗi lớp trong incremental

os.makedirs(OUTPUT_DIR, exist_ok=True)
random.seed(42)

# === Load MAVEN data
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    all_data = [json.loads(line.strip()) for line in f]

event2doc_ids = defaultdict(list)
id2doc = {}

for doc in all_data:
    doc_id = doc["id"]
    id2doc[doc_id] = doc
    event_types = {ev["type"] for ev in doc.get("events", [])}
    for et in event_types:
        event2doc_ids[et].append(doc_id)

# === Chọn top-N phổ biến classes
event_counts = {et: len(ids) for et, ids in event2doc_ids.items()}
sorted_event_types = sorted(event_counts.items(), key=lambda x: -x[1])
selected_classes = [et for et, count in sorted_event_types if count >= BASE_SHOT][:TOTAL_CLASSES]

assert len(selected_classes) >= BASE_CLASSES + INC_TASKS * INC_CLASSES_PER_TASK

# === Chia class theo task
base_classes = selected_classes[:BASE_CLASSES]
incremental_classes = selected_classes[BASE_CLASSES:BASE_CLASSES + INC_TASKS * INC_CLASSES_PER_TASK]
incr_class_chunks = [incremental_classes[i * INC_CLASSES_PER_TASK: (i + 1) * INC_CLASSES_PER_TASK] for i in range(INC_TASKS)]

# === Gán document vào task, tránh trùng lặp
used_doc_ids = set()
task_to_doc_ids = {}

def sample_docs(cls_list, k, used_set):
    selected = set()
    for cls in cls_list:
        candidates = [doc_id for doc_id in event2doc_ids[cls] if doc_id not in used_set]
        if len(candidates) < k:
            print(f"⚠️ {cls} only has {len(candidates)} available docs.")
        sampled = random.sample(candidates, min(k, len(candidates)))
        selected.update(sampled)
        used_set.update(sampled)
    return selected

# Base task
task_to_doc_ids["base"] = sample_docs(base_classes, BASE_SHOT, used_doc_ids)

# Incremental tasks
for i, class_chunk in enumerate(incr_class_chunks):
    task_key = f"incr_{i+1}"
    task_to_doc_ids[task_key] = sample_docs(class_chunk, INC_SHOT, used_doc_ids)

# === Save tasks
def save_jsonl(docs, path):
    with open(path, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

for task, doc_ids in task_to_doc_ids.items():
    docs = [id2doc[did] for did in doc_ids]
    save_jsonl(docs, os.path.join(OUTPUT_DIR, f"{task}_task.jsonl"))
    print(f"✅ Saved {task} with {len(docs)} documents")

# === Save class info
with open(os.path.join(OUTPUT_DIR, "event_type.txt"), "w", encoding="utf-8") as f:
    for et in selected_classes:
        f.write(et + "\n")
