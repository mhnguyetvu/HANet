import torch
BASE_MODEL = "bert-base-uncased"
NUM_LABELS = 168
BATCH_SIZE = 4
LR = 2e-5
MAX_EPOCHS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
