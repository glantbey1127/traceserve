import math
from dataclasses import dataclass


@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False

    def __post_init__(self):
        if not isinstance(self.max_tokens, int) or isinstance(self.max_tokens, bool) or self.max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")
        self.temperature = float(self.temperature)
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be finite and nonnegative; zero enables greedy decoding")
        if not isinstance(self.ignore_eos, bool):
            raise ValueError("ignore_eos must be a boolean")
