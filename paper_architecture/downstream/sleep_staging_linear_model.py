from __future__ import annotations

import torch
import torch.nn as nn


class SleepStageLinear(nn.Module):
    """
    Per-epoch linear probe.

    Input:
        eeg: (B, 256, T)

    For every 30-s epoch:
        256 spectral bins -> 4 sleep stages

    Output:
        logits: (B, T, 4)
    """

    def __init__(
        self,
        freq_bins: int = 256,
        num_classes: int = 4,
    ) -> None:
        super().__init__()

        self.freq_bins = int(freq_bins)
        self.num_classes = int(num_classes)

        self.classifier = nn.Linear(
            self.freq_bins,
            self.num_classes,
        )

    def model_config(self) -> dict:
        return {
            "freq_bins": self.freq_bins,
            "num_classes": self.num_classes,
        }

    def forward(
        self,
        eeg: torch.Tensor,
    ) -> torch.Tensor:

        # Accept optional single-channel dimension:
        # (B,1,F,T) -> (B,F,T)

        if eeg.ndim == 4:
            if eeg.shape[1] != 1:
                raise ValueError(
                    f"Expected one EEG channel, got {tuple(eeg.shape)}"
                )

            eeg = eeg[:, 0]

        if eeg.ndim != 3:
            raise ValueError(
                f"Expected EEG shape (B,F,T), got {tuple(eeg.shape)}"
            )

        _, f, _ = eeg.shape

        if f != self.freq_bins:
            raise ValueError(
                f"Expected {self.freq_bins} frequency bins, got {f}"
            )

        # (B,F,T) -> (B,T,F)
        eeg = eeg.transpose(1, 2)

        # (B,T,256) -> (B,T,4)
        return self.classifier(eeg)


def count_parameters(
    model: nn.Module,
) -> int:

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )
