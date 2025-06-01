# HANet: Hierarchical Augmentation for Continual Few-shot Event Detection

This repository contains a PyTorch Lightning implementation of HANet, designed for Continual Few-shot Event Detection (CFED) tasks using the MAVEN dataset.

---

## 📁 Folder Structure
```
HANet/
├── models/                  # HANet model implementation
├── utils/                   # Data loader, losses, memory, and augment utils
├── data/                    # Contains base_task.jsonl, incremental tasks, and label2id.json
├── res/                     # Output predictions (e.g. results.jsonl for evaluation)
├── train_base.py            # Train HANet on the base classes
├── train_incremental.py     # Train HANet incrementally
├── split_cf_ed_tasks.py     # Split MAVEN into base/incremental CFED tasks
├── run_incremental_loop.py  # Loop through all incremental tasks sequentially
├── predict.py               # Generate predictions for submission
├── requirements.txt         # Python dependencies
└── .gitignore               # Files to exclude from git
```

---

## 🚀 Quickstart

### 1. Create environment and install dependencies
```bash
conda create -n hanet python=3.10 -y
conda activate hanet
pip install torch==2.2.2 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 2. Prepare MAVEN dataset
Download:
- `train.jsonl`, `valid.jsonl`, `test.jsonl` from [MAVEN-Dataset](https://github.com/THU-KEG/MAVEN-dataset)

Then run:
```bash
python split_cf_ed_tasks.py  # creates base_task.jsonl and incremental_task_*.jsonl
```

### 3. Train base model
```bash
python train_base.py
```

### 4. Train incrementally (full sequence):
```bash
python run_incremental_loop.py
```

### 5. Predict on test set
```bash
python predict.py --model hanet_inc_final.ckpt --test data/test.jsonl --output res/results.jsonl
```

### 6. Evaluate
Use Codalab submission with zipped `res/results.jsonl`
---

## 📊 Evaluation
- Submit predictions to [CodaLab leaderboard](https://codalab.lisn.upsaclay.fr/competitions/3480) for full evaluation

---

## 📌 Citation
If you use this code, please cite the HANet paper:
```
@inproceedings{zhou2023hanet,
  title={HANet: Hierarchical Augmentation for Continual Few-shot Event Detection},
  author={Zhou, Wenhui and colleagues},
  booktitle={Findings of ACL 2023},
  year={2023}
}
```

---

## 🙌 Acknowledgements
- [MAVEN Dataset](https://github.com/THU-KEG/MAVEN-dataset)
- Original HANet Paper: https://arxiv.org/abs/2305.13091
