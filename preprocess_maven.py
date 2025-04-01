import json
import os
import random
from collections import defaultdict

# === Cấu hình ===
INPUT_PATH = "/Users/mhnguyetvu/workspace/HANet/datasets/MAVEN Event Detection/train.jsonl"  # ← đường dẫn đến file gốc
OUTPUT_DIR = "data"  # ← nơi sẽ lưu file output
BASE_TYPE_COUNT = 50
INCREMENTAL_TASKS = 5
SHOT_PER_TYPE = 5

random.seed(42)

# === Tạo thư mục output nếu chưa có ===
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Load dữ liệu gốc ===
with open(INPUT_PATH, "r") as f:
    data = [json.loads(line) for line in f]

# === Gom ví dụ theo loại sự kiện ===
event2examples = defaultdict(list)
for item in data:
    for evt in item.get("gold_evt_links", []):
        etype = evt["event_type"]
        event2examples[etype].append(item)

# === Lấy danh sách loại sự kiện và shuffle ===
all_event_types = sorted(event2examples.keys())
random.shuffle(all_event_types)

# === Ghi danh sách vào event_types.txt ===
with open(os.path.join(OUTPUT_DIR, "event_types.txt"), "w") as f:
    for t in all_event_types:
        f.write(t + "\n")

# === Chia base + incremental ===
base_types = all_event_types[:BASE_TYPE_COUNT]
incremental_groups = [
    all_event_types[i:i+10]
    for i in range(BASE_TYPE_COUNT, BASE_TYPE_COUNT + INCREMENTAL_TASKS * 10, 10)
]

# === Ghi base_task.jsonl ===
base_data = []
for t in base_types:
    base_data.extend(event2examples[t])
with open(os.path.join(OUTPUT_DIR, "base_task.jsonl"), "w") as f:
    for ex in base_data:
        f.write(json.dumps(ex) + "\n")
print(f"✅ Base task: {len(base_data)} examples from {len(base_types)} types.")

# === Ghi từng incremental task ===
for i, group in enumerate(incremental_groups):
    few_data = []
    for t in group:
        few = event2examples[t][:SHOT_PER_TYPE]
        few_data.extend(few)
    with open(os.path.join(OUTPUT_DIR, f"incremental_task_{i+1}.jsonl"), "w") as f:
        for ex in few_data:
            f.write(json.dumps(ex) + "\n")
    print(f"✅ Incremental task {i+1}: {len(few_data)} examples from {len(group)} types.")
