# HANet: Continual Few-Shot Event Detection

This project implements **HANet** (Hierarchical Augmentation Network) for continual few-shot event detection using the MAVEN dataset.

---
## DATASET
https://drive.google.com/drive/folders/19Q0lqJE6A98OLnRqQVhbX3e6rG4BVGn8
## 📂 Folder Structure

```
Hanet/
├── data/                    # MAVEN data and processed splits
├── checkpoints/             # Saved model checkpoints and label map
├── models/                  # HANet model definition
├── utils/                   # Data loading utility
├── preprocessing_maven.py  # Split MAVEN into base & incremental tasks
├── config.py                # Training config
├── train.py                 # Train on base task
├── train_incremental.py     # Train on incremental tasks with memory replay
└── predict.py               # Predict event type for new sentence
```

---

## 🧪 1. Installation

```bash
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

---

## 🧠 2. Train base model

```bash
python train.py
```

- Trains on `base_task.jsonl`
- Saves best checkpoint to `checkpoints/hanet_best_base.pt`

---

## 🔁 3. Continual training on incremental tasks

```bash
python train_incremental.py
```

- Trains sequentially on `incr_*.jsonl`
- Uses memory replay (1 exemplar per class)
- Saves checkpoints to `checkpoints/hanet_incr_X_task.pt`
- Saves label map to `checkpoints/label_map.json`

---

## 🔍 4. Predict a new sentence

```bash
python predict.py
```

Example:
```python
text = "The rebels organized a large protest in the capital."
trigger_offset = [2, 3]  # token offset for 'organized'
```
Output:
```
🔍 Trigger: 'organized' → Predicted event type: Arranging
```

---

## ⚙️ Config (config.py)

| Parameter      | Description                        |
|----------------|------------------------------------|
| `BASE_MODEL`   | Pretrained BERT model              |
| `NUM_LABELS`   | Max number of event types          |
| `MAX_EPOCHS`   | Epochs per task                    |
| `BATCH_SIZE`   | Training batch size                |
| `DEVICE`       | CUDA or CPU                        |

---

## 📌 Notes

- `label_map.json` is automatically generated at the end of incremental training.
- Replay memory uses simple exemplar selection (e.g., longest trigger). Can be enhanced with embeddings or prototype matching.

---

**Author:** nguyetnvm  

