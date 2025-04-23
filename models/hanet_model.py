import torch
import torch.nn as nn
from transformers import BertModel

class HANetTriggerEncoder(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size * 2, num_labels)

    def forward(self, input_ids, attention_mask, trigger_pos):
        # BERT encoding
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # [B, L, H]

        # === Extract trigger embeddings ===
        batch_size = input_ids.size(0)
        device = input_ids.device

        # Sanity check: trigger_pos.shape = [B, 2]
        assert trigger_pos.size(1) == 2, "trigger_pos must have shape [B, 2]"

        # Clamp trigger indices to avoid out-of-bound
        trigger_start = torch.clamp(trigger_pos[:, 0], 0, hidden_states.size(1) - 1)
        trigger_end = torch.clamp(trigger_pos[:, 1] - 1, 0, hidden_states.size(1) - 1)

        # Get embeddings at trigger start & end
        start_emb = hidden_states[torch.arange(batch_size, device=device), trigger_start]  # [B, H]
        end_emb = hidden_states[torch.arange(batch_size, device=device), trigger_end]      # [B, H]

        # Concatenate trigger representation
        trig_repr = torch.cat([start_emb, end_emb], dim=-1)  # [B, 2H]

        # Classify
        logits = self.classifier(trig_repr)  # [B, num_labels]
        return logits
