"""V_phi: 2-layer MLP head + sigmoid output, attached to a shared LLM backbone.

Paper Section 3.3 / Appendix D:
    Hidden = d_model (3584 for 7B Qwen2.5, 2048 for 3B Qwen2.5).
    GELU activation, sigmoid -> [0,1], zero-init final layer.
    Head params: 12.85M (7B) / 4.20M (3B).
    Backbone trainable with LR 10x smaller than head. Value-head WD = 0.
"""
import torch
import torch.nn as nn
from transformers import AutoModel


class ValueHead(nn.Module):
    def __init__(self, hidden_size: int, mlp_hidden: int | None = None):
        # Paper: hidden = d_model. Default to that if caller doesn't override.
        super().__init__()
        mlp_hidden = mlp_hidden or hidden_size
        self.fc1 = nn.Linear(hidden_size, mlp_hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(mlp_hidden, 1)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, h):  # h: (B, H) - last-token hidden state
        return torch.sigmoid(self.fc2(self.act(self.fc1(h)))).squeeze(-1)


class ValueModel(nn.Module):
    """Shared backbone + value head. Reads last-real-token hidden state."""

    def __init__(self, model_name_or_path: str, mlp_hidden: int | None = None,
                 dtype: torch.dtype = torch.bfloat16, attn_impl: str = "eager"):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            model_name_or_path, torch_dtype=dtype, attn_implementation=attn_impl,
        )
        self.head = ValueHead(self.backbone.config.hidden_size, mlp_hidden=mlp_hidden)

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state                # (B, T, H)
        # last real token via right-most attention=1 position
        idx = (attention_mask.sum(dim=1) - 1).clamp(min=0)
        h_last = h[torch.arange(h.size(0), device=h.device), idx]
        return self.head(h_last.float())          # cast to fp32 for stable sigmoid
