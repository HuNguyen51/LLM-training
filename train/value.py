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

class ValueHead(nn.Module):
    def __init__(self, hidden_size, dtype, dropout_rate=0.1):
        super(ValueHead, self).__init__()

        # Kiến trúc đơn giản: 2 lớp fully connected với activation ở giữa
        self.dropout = nn.Dropout(dropout_rate)
        self.summary = nn.Linear(hidden_size, 1, dtype=dtype)
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)

    def forward(self, hidden_states):
        # Forward pass qua các lớp
        x = self.dropout(hidden_states)
        x = self.summary(x)
        x = self.flatten(x)
        return x

class ValueModel(torch.nn.Module):
    def __init__(self, model):
        super(ValueModel, self).__init__()

        self.backbone = LMBackbone(model)
        self.base_model_prefix = 'backbone'
        self.hidden_size = model.config.hidden_size
        self.torch_dtype = model.config.torch_dtype

        self.value_head = ValueHead(self.hidden_size, self.torch_dtype)

        self.to(device)

    def score(self, last_hidden_state):
        value = self.value_head(last_hidden_state)
        return value