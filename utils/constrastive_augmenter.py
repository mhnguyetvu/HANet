import torch
import torch.nn.functional as F
import random
"""
Augmentation trên câu gốc bằng:

- Drop token ngẫu nhiên

- Thay token ngẫu nhiên bằng [MASK]

Sinh cặp contrastive embedding (z1, z2) từ hai bản augment của cùng một câu chứa trigger

Tính contrastive loss bằng cosine similarity + temperature scaling
"""
class ContrastiveAugmenter:
    def __init__(self, tokenizer, model, feature_extractor, device, temperature=0.1):
        """
        Args:
            tokenizer: BERT tokenizer
            model: classification model
            feature_extractor: function to extract embeddings
            device: CUDA or CPU
            temperature: temperature scaling for contrastive loss
        """
        self.tokenizer = tokenizer
        self.model = model
        self.feature_extractor = feature_extractor
        self.device = device
        self.temperature = temperature

    def random_token_drop(self, tokens, drop_prob=0.1):
        return [t for t in tokens if random.random() > drop_prob or t == '[CLS]' or t == '[SEP]']

    def random_token_replace(self, tokens, replace_prob=0.1):
        return [t if random.random() > replace_prob else '[MASK]' for t in tokens]

    def augment_text(self, tokens):
        choice = random.choice(['drop', 'replace'])
        return self.random_token_drop(tokens) if choice == 'drop' else self.random_token_replace(tokens)

    def generate_contrastive_pairs(self, sentence, trigger_pos):
        tokens = sentence.split()
        aug1 = self.augment_text(tokens)
        aug2 = self.augment_text(tokens)

        enc1 = self.tokenizer(" ".join(aug1), return_tensors="pt", padding="max_length", truncation=True, max_length=512)
        enc2 = self.tokenizer(" ".join(aug2), return_tensors="pt", padding="max_length", truncation=True, max_length=512)

        input_ids1 = enc1["input_ids"].to(self.device)
        attn_mask1 = enc1["attention_mask"].to(self.device)
        input_ids2 = enc2["input_ids"].to(self.device)
        attn_mask2 = enc2["attention_mask"].to(self.device)

        trig_pos = torch.tensor([trigger_pos], device=self.device)
        z1 = self.feature_extractor(input_ids1, attn_mask1, trig_pos)
        z2 = self.feature_extractor(input_ids2, attn_mask2, trig_pos)
        return z1, z2

    def contrastive_loss(self, z1, z2):
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        representations = torch.cat([z1, z2], dim=0)  # [2, H]

        similarity_matrix = torch.matmul(representations, representations.T) / self.temperature
        labels = torch.tensor([1, 0], dtype=torch.long, device=self.device)

        loss = F.cross_entropy(similarity_matrix, labels)
        return loss
