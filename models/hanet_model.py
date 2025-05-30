import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import AutoModel
from typing import Any, Dict
from utils.losses import info_nce_loss

class HANetModel(pl.LightningModule):
    def __init__(self, model_name: str, num_labels: int, lr: float = 2e-5, lambda_re=1.0, lambda_cls=1.0):
        """
        HANet with memory replay and contrastive augmentation support.

        Args:
            model_name (str): Name of the base BERT model.
            num_labels (int): Number of event type classes.
            lr (float): Learning rate.
            lambda_re (float): Weight for replay loss.
            lambda_cls (float): Weight for contrastive loss.
        """
        super().__init__()
        self.save_hyperparameters()

        self.bert = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, trigger_mask, return_repr=False):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # (B, T, H)

        # Get trigger token representation
        masked_output = sequence_output * trigger_mask.unsqueeze(-1)
        trigger_repr = masked_output.sum(1) / trigger_mask.sum(1, keepdim=True)

        logits = self.classifier(trigger_repr)  # (B, C)
        return (logits, trigger_repr) if return_repr else logits

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        # Forward and primary loss
        logits, trigger_repr = self(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            trigger_mask=batch['trigger_mask'],
            return_repr=True
        )
        loss = self.loss_fn(logits, batch['labels'])
        self.log("train_ce_loss", loss)

        # Replay loss from augmented prototypes
        if 'aug_repr' in batch:
            loss_re = torch.mean((trigger_repr - batch['aug_repr'])**2)
            loss += self.hparams.lambda_re * loss_re
            self.log("loss_replay", loss_re)

        # Contrastive loss using positive/negative trigger pairs
        if 'pos_repr' in batch and 'neg_repr' in batch:
            loss_cls = info_nce_loss(trigger_repr, batch['pos_repr'], batch['neg_repr'])
            loss += self.hparams.lambda_cls * loss_cls
            self.log("loss_contrastive", loss_cls)

        self.log("train_total_loss", loss)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        logits = self(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], trigger_mask=batch['trigger_mask'])
        preds = torch.argmax(logits, dim=-1)
        acc = (preds == batch['labels']).float().mean()
        self.log("val_acc", acc)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)