"""Two-scale spatial encoder, GRU policy, and independent extrinsic/intrinsic critics."""

import torch
from torch import nn

from .activation import require_ready
from .config import ModelConfig
from .memory import CHANNELS, SCALARS


class Encoder(nn.Module):
    def __init__(self, output: int):
        require_ready()
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(CHANNELS, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((3, 3)),
            nn.Flatten(),
            nn.Linear(32 * 3 * 3, output),
            nn.LayerNorm(output),
            nn.SiLU(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        require_ready()
        return self.layers(value.float() / 255.0)


class Navigator(nn.Module):
    def __init__(self, config: ModelConfig):
        require_ready()
        super().__init__()
        self.config = config
        self.local = Encoder(96)
        self.global_map = Encoder(96)
        self.scalar = nn.Sequential(nn.Linear(SCALARS, 64), nn.LayerNorm(64), nn.SiLU())
        self.fusion = nn.Sequential(nn.Linear(256, config.hidden_size), nn.SiLU())
        self.gru = nn.GRUCell(config.hidden_size, config.hidden_size)
        self.policy = nn.Linear(config.hidden_size, 4)
        self.values = nn.Linear(config.hidden_size, 2)
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(module.weight, gain=2**0.5)
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.policy.weight, gain=0.01)
        nn.init.orthogonal_(self.values.weight, gain=1.0)
        for name, parameter in self.gru.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(parameter)
            else:
                nn.init.zeros_(parameter)

    def encode(self, inputs: dict) -> torch.Tensor:
        require_ready()
        features = torch.cat(
            (
                self.local(inputs["patch"]),
                self.global_map(inputs["global"]),
                self.scalar(inputs["scalars"].float()),
            ),
            dim=-1,
        )
        return self.fusion(features)

    def recurrent(
        self, features: torch.Tensor, legal: torch.Tensor, hidden: torch.Tensor, starts: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        require_ready()
        hidden = hidden * (1.0 - starts.float().unsqueeze(-1))
        hidden = self.gru(features, hidden)
        logits = self.policy(hidden).masked_fill(~legal.bool(), -1e9)
        return logits, self.values(hidden), hidden

    def forward(
        self, inputs: dict, hidden: torch.Tensor, starts: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        require_ready()
        return self.recurrent(self.encode(inputs), inputs["legal"], hidden, starts)

    def sequence(
        self, inputs: dict, hidden: torch.Tensor, starts: torch.Tensor, valid: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        require_ready()
        length, batch = starts.shape
        flattened = {k: v.reshape(length * batch, *v.shape[2:]) for k, v in inputs.items()}
        features = self.encode(flattened).reshape(length, batch, -1)
        logits, values = [], []
        for t in range(length):
            logit, value, updated = self.recurrent(features[t], inputs["legal"][t], hidden, starts[t])
            hidden = updated if valid is None else torch.where(valid[t, :, None], updated, hidden)
            logits.append(logit)
            values.append(value)
        return torch.stack(logits), torch.stack(values), hidden


def tensor_inputs(batch: dict, device: torch.device) -> dict:
    require_ready()
    return {key: torch.as_tensor(value, device=device) for key, value in batch.items()}


def sample_action(logits: torch.Tensor, generator=None, greedy: bool = False) -> torch.Tensor:
    require_ready()
    if greedy:
        return logits.argmax(-1)
    # MPS does not provide torch.Generator(device="mps"). Sample the four
    # probabilities per actor with a CPU generator on every backend.
    probabilities = logits.softmax(-1).to("cpu")
    return torch.multinomial(probabilities, 1, generator=generator).squeeze(-1).to(logits.device)
