import torch  # Thư viện deep learning chính
import torch.nn as nn  # Module loss, layers
from transformers import BertTokenizer, AdamW  # BERT tokenizer và optimizer AdamW
from models.hanet_model import HANetTriggerEncoder  # Import mô hình HANet đã định nghĩa
from utils.data_loader import MAVENDataset  # Dataset reader cho file jsonl
from torch.utils.data import DataLoader  # Dataloader để tạo batch
import config  # Cấu hình: model, batch size, learning rate, device
import psutil, os  # Để đo lượng RAM sử dụng
import warnings
warnings.filterwarnings('ignore')

# Ép sử dụng GPU cụ thể (A100 gán vào cuda:0)
config.DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Khởi tạo tokenizer BERT
tokenizer = BertTokenizer.from_pretrained(config.BASE_MODEL)

# Khởi tạo mô hình và chuyển sang GPU nếu có
model = HANetTriggerEncoder(config.BASE_MODEL, config.NUM_LABELS).to(config.DEVICE)

# Khởi tạo optimizer AdamW
optimizer = AdamW(model.parameters(), lr=config.LR)

# Tải dữ liệu từ base task
data = MAVENDataset("/data/AITeam/nguyetnvm/Hanet/data/small_split/base_task.jsonl", tokenizer)

# Tạo thư mục lưu checkpoint nếu chưa có
os.makedirs("/data/AITeam/nguyetnvm/Hanet/checkpoints", exist_ok=True)

# Tạo DataLoader để chia batch
dataloader = DataLoader(data, batch_size=config.BATCH_SIZE, shuffle=True)

# Bắt đầu huấn luyện
model.train()
best_loss = float("inf")  # Theo dõi loss tốt nhất để lưu mô hình

for epoch in range(config.MAX_EPOCHS):  # Lặp qua các epoch
    for batch in dataloader:  # Lặp qua từng batch
        input_ids = batch['input_ids'].to(config.DEVICE)  # Tensor input ids
        attention_mask = batch['attention_mask'].to(config.DEVICE)  # Attention mask
        labels = batch['label'].to(config.DEVICE)  # Nhãn event type
        trigger_pos = batch['trigger_pos'].to(config.DEVICE)  # Vị trí trigger [start, end]

        logits = model(input_ids, attention_mask, trigger_pos)  # Forward pass
        loss = nn.CrossEntropyLoss()(logits, labels)  # Tính loss phân loại

        optimizer.zero_grad()  # Reset gradient
        loss.backward()  # Lan truyền gradient
        optimizer.step()  # Cập nhật trọng số

    # Lưu mô hình nếu loss tốt hơn
    if loss.item() < best_loss:
        best_loss = loss.item()
        torch.save(model.state_dict(), "/data/AITeam/nguyetnvm/Hanet/checkpoints/hanet_best_base.pt")
        print(f"💾 Saved best model at epoch {epoch+1} with loss {best_loss:.4f}")

    # In thông tin RAM sử dụng sau mỗi epoch
    process = psutil.Process(os.getpid())
    ram_usage = process.memory_info().rss / 1024 ** 2  # RAM tính bằng MB
    print(f"Epoch {epoch+1} | Loss: {loss.item():.4f} | 🧠 RAM: {ram_usage:.2f} MB")

    # Nếu dùng GPU thì in lượng VRAM và tên thiết bị
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.memory_allocated() / 1024 ** 2  # VRAM tính bằng MB
        device_name = torch.cuda.get_device_name(torch.cuda.current_device())
        print(f"🔥 GPU memory: {gpu_mem:.2f} MB | 📟 GPU: {device_name}")

