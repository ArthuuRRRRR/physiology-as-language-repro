from __future__ import annotations

import torch
import torch.nn as nn


class ArousalWakeLinear(nn.Module):
    """
    Keondo-style two-head linear probe.

    Input:
        one 30-s EEG spectrogram feature vector per epoch
        (256 frequency bins for PaSL)

    Outputs:
        logits[..., 0] = arousal
        logits[..., 1] = wake
    """

    def __init__(self, input_dim: int = 256) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.classifier = nn.Linear(self.input_dim, 2)

    def model_config(self) -> dict:
        return {
            "input_dim": self.input_dim,
            "outputs": ["arousal", "wake"],
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            if x.shape[-1] != self.input_dim:
                raise ValueError(
                    f"Expected features (N,{self.input_dim}), got {tuple(x.shape)}"
                )
            return self.classifier(x)

        # Convenience path for PaSL spectrograms stored as (B, F, T).
        if x.ndim == 3:
            if x.shape[1] != self.input_dim:
                raise ValueError(
                    f"Expected spectrogram (B,{self.input_dim},T), got {tuple(x.shape)}"
                )
            return self.classifier(x.transpose(1, 2))

        raise ValueError(
            f"Expected (N,{self.input_dim}) or (B,{self.input_dim},T), "
            f"got {tuple(x.shape)}"
        )


def count_parameters(model: nn.Module) -> int:
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )
