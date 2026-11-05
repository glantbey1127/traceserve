import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @torch.compile
    def rms_forward(self, x: torch.Tensor) -> torch.Tensor:
        original_dtype = x.dtype
        values = x.float()
        variance = values.pow(2).mean(dim=-1, keepdim=True)
        # Out-of-place operations preserve the input even when x already is float32.
        normalized = values * torch.rsqrt(variance + self.eps)
        return normalized.to(original_dtype) * self.weight

    @torch.compile
    def add_rms_forward(self, x: torch.Tensor, residual: torch.Tensor):
        original_dtype = x.dtype
        values = x.float() + residual.float()
        updated_residual = values.to(original_dtype)
        variance = values.pow(2).mean(dim=-1, keepdim=True)
        normalized = values * torch.rsqrt(variance + self.eps)
        return normalized.to(original_dtype) * self.weight, updated_residual

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None):
        if residual is None:
            return self.rms_forward(x)
        return self.add_rms_forward(x, residual)
