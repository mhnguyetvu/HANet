## DATASET
https://drive.google.com/drive/folders/19Q0lqJE6A98OLnRqQVhbX3e6rG4BVGn8

📚 Reference
HANet paper: Continual Few-shot Event Detection via Hierarchical Augmentation Networks (EMNLP 2022)

MAVEN dataset: https://github.com/THU-KEG/MAVEN-dataset

# 🧠 HANet Training Pipeline – Flow Explanation

This training script implements a continual few-shot event detection model using **HANet-style training**, with memory replay and prototypical augmentation.

---

## 📦 1. Configuration

```python
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 64
BATCH_SIZE = 16
LR = 2e-5
EPOCHS = 3
REPLAY_PER_CLASS = 1
SIGMA = 0.1  # Gaussian noise for augmentation
```

---

## 🔤 2. Tokenizer and Label Mapping

- Use `BertTokenizer` from Huggingface
- Read `event_types.txt` and map event type → label index

---

## 📚 3. Dataset Preparation

Each line in the `.jsonl` file contains:
- `"words"`: tokenized sentence
- `"gold_evt_links"`: list of events in the sentence

The custom `Dataset` returns:
- `input_ids`, `attention_mask` (for BERT)
- `label` (event type ID)

---

## 🧠 4. Model: BERT + Classifier

```python
BERT ➞ [CLS] vector ➞ Linear layer ➞ softmax
```

Also returns the `[CLS]` embedding for prototypical augmentation.

---

## 📂 5. Replay & Prototypical Augmentation

- **Replay Memory**: stores representative samples from base classes
- **Prototypical Augmentation**:
  - Take the prototype embedding from few-shot sample
  - Add Gaussian noise to simulate new instances
  - Use synthetic embedding directly

---

## 🏋️‍♂️ 6. Train Base Task

- Load `base_task.jsonl`
- Train normally on all base classes
- Save 1 exemplar/class to memory

---

## ➕ 7. Incremental Learning Loop

For each `incremental_task_i.jsonl`:

1. Extract embedding from few-shot examples
2. Apply augmentation (5x per sample)
3. Add replayed base exemplars
4. Train on synthetic + real samples
5. Evaluate on few-shot task

---

## 🧹 8. Forward Override for Embedding-Only Samples

To support synthetic inputs that contain no tokens:
```python
if input_ids.sum() == 0:
    return classifier(embedding)
```

---

## ✅ 9. Evaluation

Evaluate accuracy for:
- Base task
- Each incremental task

---

## 📊 Flow Summary

```text
        base_task.jsonl               incremental_task_*.jsonl
              │                               │
              ▼                               ▼
   ┌──────────────────────────┐            ┌─────────────────────────────────┐
   │  Tokenize + Map  │            │ Few-shot + Augment + Replay │
   └──────────────────────────┘            └─────────────────────────────────┘
              │                               │
              ▼                               ▼
     Train base model                Fine-tune (new + memory)
              │                               │
              ▼                               ▼
       Evaluate base                  Evaluate per-task
```

---

## 🔍 Optional Enhancements

- [ ] Add `F1-score` computation
- [ ] Log per-task results to file
- [ ] Add contrastive loss (optional in HANet)
- [ ] Save/load checkpoints

