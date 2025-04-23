# === config.py ===

# Mô hình BERT nền tảng
BASE_MODEL = "bert-base-uncased"  # Hoặc "vinai/phobert-base" nếu dùng tiếng Việt

# Số lớp cần phân loại (Base task có 10 lớp)
NUM_LABELS = 10

# Học rate cho optimizer AdamW
LR = 2e-5

# Batch size khi huấn luyện
BATCH_SIZE = 8

# Số epoch
MAX_EPOCHS = 5

# Thiết bị tính toán: "cuda:0" sẽ được gán tự động trong train_base.py
DEVICE = "cuda"
