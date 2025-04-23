import json
from sklearn.metrics import classification_report, accuracy_score, f1_score

input_path = "/data/AITeam/nguyetnvm/Hanet/predictions/predict_valid.jsonl"

y_true = []
y_pred = []

with open(input_path, "r", encoding="utf-8") as f:
    for line in f:
        entry = json.loads(line.strip())
        y_true.append(entry["true_event_type"])
        y_pred.append(entry["pred_event_type"])

print("📊 Classification Report:")
print(classification_report(y_true, y_pred, digits=4))

acc = accuracy_score(y_true, y_pred)
print(f"\n✅ Accuracy: {acc:.4f}")
# F1-score
f1_micro = f1_score(y_true, y_pred, average='micro')
f1_macro = f1_score(y_true, y_pred, average='macro')
print(f"✅ F1-micro: {f1_micro:.4f}")
print(f"✅ F1-macro: {f1_macro:.4f}")
