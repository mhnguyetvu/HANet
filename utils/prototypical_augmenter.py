import torch
import torch.nn.functional as F
"""
Lấy ra các exemplar đã lưu trong memory bank.

Tính mean và std của embedding.

Sinh thêm num_augments embedding giả bằng Gaussian noise quanh prototype.

Trả về các embedding với nhãn tương ứng.
"""
class PrototypicalAugmenter:
    def __init__(self, model, feature_extractor, device, std_ratio=0.1, num_augments=5):
        """
        Args:
            model: The classification model (e.g., HANet)
            feature_extractor: Function to extract embedding from model
            device: CUDA or CPU
            std_ratio: Ratio of std to use for Gaussian noise
            num_augments: Number of augmented examples to generate per class
        """
        self.model = model
        self.feature_extractor = feature_extractor
        self.device = device
        self.std_ratio = std_ratio
        self.num_augments = num_augments

    def generate_augments(self, exemplars, label_id):
        """
        Args:
            exemplars: List of dicts {input_ids, attention_mask, trigger_pos, label}
            label_id: Class label of exemplar
        Returns:
            List of augmented feature dicts
        """
        self.model.eval()

        with torch.no_grad():
            all_embeddings = []
            for ex in exemplars:
                input_ids = ex['input_ids'].unsqueeze(0).to(self.device)
                attention_mask = ex['attention_mask'].unsqueeze(0).to(self.device)
                trigger_pos = ex['trigger_pos'].unsqueeze(0).to(self.device)

                embed = self.feature_extractor(input_ids, attention_mask, trigger_pos)  # [1, H]
                all_embeddings.append(embed.squeeze(0))

            all_embeddings = torch.stack(all_embeddings)  # [N, H]
            mean = all_embeddings.mean(dim=0)
            std = all_embeddings.std(dim=0)

            # Ensure no NaNs and negative stds
            std = torch.where(torch.isnan(std), torch.full_like(std, 1e-6), std)
            std = torch.clamp(std * self.std_ratio, min=1e-6)

            augments = []
            for _ in range(self.num_augments):
                noise = torch.normal(mean=torch.zeros_like(std), std=std).to(self.device)
                new_embed = mean + noise
                augments.append({
                    'embedding': new_embed.detach(),
                    'label': torch.tensor(label_id)
                })

            return augments
