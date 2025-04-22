import torch
import torch.nn as nn
from transformers import BertModel

class HANetTriggerEncoder(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size * 2, num_labels)

    def forward(self, input_ids, attention_mask, trigger_pos):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        start_emb = hidden_states[range(len(trigger_pos)), trigger_pos[:, 0]]
        end_emb = hidden_states[range(len(trigger_pos)), trigger_pos[:, 1] - 1]
        trig_repr = torch.cat([start_emb, end_emb], dim=-1)
        return self.classifier(trig_repr)