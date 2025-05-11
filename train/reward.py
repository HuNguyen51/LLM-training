from torch import nn
from transformers.modeling_outputs import SequenceClassifierOutput

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LMBackbone(nn.Module):
    def __init__(self, model):
        super(LMBackbone, self).__init__()
        self.backbone = model.base_model.model.model
        self.base_model_prefix = 'backbone'

    def forward(self, input_ids, attention_mask=None, **kwargs):
        x = self.backbone(input_ids, attention_mask=attention_mask, **kwargs)
        x.hidden_states = torch.stack([x.last_hidden_state])
        return x

class RewardHead(nn.Module):
    def __init__(self, hidden_size, dtype, train_ppo=False, dropout_rate=0.1):
        super(RewardHead, self).__init__()
        self.train_ppo = train_ppo

        self.pooler = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, dtype=dtype),
            nn.Dropout(dropout_rate)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 1, dtype=dtype),
            nn.Dropout(dropout_rate)
        )

    def forward(self, last_hidden_state, attention_mask=None, **kwargs):
        seq_len = last_hidden_state.size(1)
        pooling = self.pooler(last_hidden_state)
        # Áp dụng attention mask để loại bỏ các padding token
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(-1)
            pooling = pooling * attention_mask
            sum_hidden = torch.sum(pooling, dim=1)
            token_count = torch.sum(attention_mask, dim=1)
            # Tránh chia cho 0
            token_count = torch.clamp(token_count, min=1e-9)

            mean_pooled = sum_hidden / token_count
        else:
            mean_pooled = torch.mean(pooling, dim=1)

        x = self.classifier(mean_pooled)
        if self.train_ppo:
            x = x.repeat(1, seq_len)
        return x

class RewardModel(torch.nn.Module):
    def __init__(self, model, train_ppo=False):
        super(RewardModel, self).__init__()

        self.backbone = LMBackbone(model)
        self.base_model_prefix = 'backbone'
        self.hidden_size = model.config.hidden_size
        self.torch_dtype = model.config.torch_dtype

        self.reward_head = RewardHead(self.hidden_size, self.torch_dtype, train_ppo)

        self.to(device)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        x = self.backbone(input_ids, attention_mask=attention_mask, **kwargs)
        outputs = self.reward_head(x.last_hidden_state)
        return SequenceClassifierOutput(
            loss=None,
            logits=outputs,
            hidden_states=outputs.hidden_states if hasattr(outputs, "hidden_states") else None,
            attentions=outputs.attentions if hasattr(outputs, "attentions") else None,
        )

    def score(self, last_hidden_state):
        reward_score = self.reward_head(last_hidden_state)
        return reward_score