from __future__ import annotations

import math
import torch
import torch.nn as nn


class ConvFreqTimeBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        freq_kernel: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(freq_kernel, 3),
                stride=(2, 1),
                padding=(freq_kernel // 2, 1),
                bias=False,
            ),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(3, 3),
                padding=(1, 1),
                bias=False,
            ),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),

            nn.Dropout2d(dropout),
        )

    def forward(self, x):
        return self.block(x)


class TemporalResidualBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        dilation: int,
        kernel_size: int = 5,
        dropout: float = 0.15,
    ):
        super().__init__()

        padding = dilation * (kernel_size - 1) // 2

        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            bias=False,
        )
        self.norm1 = nn.GroupNorm(8, channels)

        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            bias=False,
        )
        self.norm2 = nn.GroupNorm(8, channels)

        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act(x)
        x = self.dropout(x)

        return (x + residual) / math.sqrt(2.0)


class SleepStageCNNTCN(nn.Module):
    """
    Input:
        (B, 1, 256, T)
        or (B, 256, T)

    Output:
        (B, T, 4)
    """

    def __init__(
        self,
        freq_bins: int = 256,
        num_classes: int = 4,
        channels: int = 128,
        cnn_dropout: float = 0.10,
        tcn_dropout: float = 0.15,
    ):
        super().__init__()

        self.freq_bins = int(freq_bins)
        self.num_classes = int(num_classes)
        self.channels = int(channels)
        self.cnn_dropout = float(cnn_dropout)
        self.tcn_dropout = float(tcn_dropout)

        self.cnn = nn.Sequential(
            ConvFreqTimeBlock(
                1, 32,
                freq_kernel=7,
                dropout=cnn_dropout,
            ),
            ConvFreqTimeBlock(
                32, 64,
                freq_kernel=5,
                dropout=cnn_dropout,
            ),
            ConvFreqTimeBlock(
                64, 96,
                freq_kernel=5,
                dropout=cnn_dropout,
            ),
            ConvFreqTimeBlock(
                96, channels,
                freq_kernel=3,
                dropout=cnn_dropout,
            ),
        )

        # After CNN:
        # (B, channels, ~16 freq bins, T)
        #
        # Frequency pooling only.
        # Time dimension T is never pooled.

        self.temporal_input = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(8, channels),
            nn.GELU(),
        )

        self.tcn = nn.Sequential(
            TemporalResidualBlock(
                channels,
                dilation=1,
                dropout=tcn_dropout,
            ),
            TemporalResidualBlock(
                channels,
                dilation=2,
                dropout=tcn_dropout,
            ),
            TemporalResidualBlock(
                channels,
                dilation=4,
                dropout=tcn_dropout,
            ),
            TemporalResidualBlock(
                channels,
                dilation=8,
                dropout=tcn_dropout,
            ),
            TemporalResidualBlock(
                channels,
                dilation=16,
                dropout=tcn_dropout,
            ),
        )

        self.classifier = nn.Conv1d(
            channels,
            num_classes,
            kernel_size=1,
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(
            module,
            (nn.Conv1d, nn.Conv2d, nn.Linear),
        ):
            nn.init.trunc_normal_(
                module.weight,
                std=0.02,
            )
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def model_config(self):
        return {
            "freq_bins": self.freq_bins,
            "num_classes": self.num_classes,
            "channels": self.channels,
            "cnn_dropout": self.cnn_dropout,
            "tcn_dropout": self.tcn_dropout,
        }

    def forward(self, eeg):
        if eeg.ndim == 3:
            eeg = eeg.unsqueeze(1)

        if eeg.ndim != 4:
            raise ValueError(
                f"Expected (B,1,F,T), got {tuple(eeg.shape)}"
            )

        if eeg.shape[1] != 1:
            raise ValueError(
                f"Expected one EEG channel, got {tuple(eeg.shape)}"
            )

        if eeg.shape[2] != self.freq_bins:
            raise ValueError(
                f"Expected {self.freq_bins} freq bins, "
                f"got {eeg.shape[2]}"
            )

        # Local frequency/time representation
        x = self.cnn(eeg)

        # Preserve time, aggregate frequency
        x = x.mean(dim=2)

        # (B,C,T)
        x = self.temporal_input(x)

        # Long temporal context
        x = self.tcn(x)

        # (B,4,T)
        logits = self.classifier(x)

        # (B,T,4)
        return logits.transpose(1, 2)


def count_parameters(model):
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )
