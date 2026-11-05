import torch
from torch import nn


class Sampler(nn.Module):
    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        values = logits.float()
        greedy = values.argmax(dim=-1)
        safe_temperatures = temperatures.clamp_min(1e-10).unsqueeze(1)
        probs = torch.softmax(values / safe_temperatures, dim=-1)
        sampled = (
            probs / torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
        ).argmax(dim=-1)
        return torch.where(temperatures == 0, greedy, sampled)
