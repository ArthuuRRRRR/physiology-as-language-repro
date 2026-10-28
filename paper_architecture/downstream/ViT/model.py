from __future__ import annotations

import torch
import torch.nn as nn


class SleepStageViT(nn.Module):
    """
    ViT-Tiny-style sleep staging model.

    Input
    -----
    x : Tensor
        Shape (B, 1, 256, 512)

        256 = frequency bins
        512 = consecutive 30-second epochs

    Output
    ------
    logits : Tensor
        Shape (B, 512, 4)

        One prediction per 30-second epoch:
            0 = Wake
            1 = Light
            2 = Deep
            3 = REM

    Architecture
    ------------
    1. Spectral patch embedding:
       16 frequency bins x 1 temporal epoch.

    2. Spectral Transformer:
       Models the frequency structure independently
       for every 30-second epoch.

    3. Temporal Transformer:
       Models sleep-stage transitions across the
       complete 256-minute window.

    4. Linear classifier:
       192 -> 4 for every epoch.
    """

    def __init__(
        self,
        *,
        freq_bins: int = 256,
        num_epochs: int = 512,
        freq_patch_size: int = 16,
        embed_dim: int = 192,
        num_heads: int = 3,
        spectral_depth: int = 4,
        temporal_depth: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        num_classes: int = 4,
    ) -> None:

        super().__init__()

        if freq_bins % freq_patch_size != 0:
            raise ValueError(
                f"freq_bins={freq_bins} must be divisible by "
                f"freq_patch_size={freq_patch_size}"
            )

        self.freq_bins = int(freq_bins)
        self.num_epochs = int(num_epochs)
        self.freq_patch_size = int(freq_patch_size)

        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)

        self.spectral_depth = int(spectral_depth)
        self.temporal_depth = int(temporal_depth)

        self.num_classes = int(num_classes)

        self.num_freq_patches = (
            self.freq_bins
            // self.freq_patch_size
        )

        hidden_dim = int(
            self.embed_dim
            * mlp_ratio
        )

        # ====================================================
        # Spectral patch embedding
        #
        # (B, 1, 256, 512)
        # ->
        # (B, 192, 16, 512)
        #
        # Temporal resolution is preserved exactly.
        # ====================================================

        self.patch_embed = nn.Conv2d(
            in_channels=1,
            out_channels=self.embed_dim,
            kernel_size=(
                self.freq_patch_size,
                1,
            ),
            stride=(
                self.freq_patch_size,
                1,
            ),
            bias=True,
        )

        self.patch_norm = nn.LayerNorm(
            self.embed_dim
        )

        # ====================================================
        # Spectral CLS token
        #
        # One learned CLS token is used to aggregate the
        # frequency information of each 30-second epoch.
        # ====================================================

        self.spectral_cls_token = nn.Parameter(
            torch.zeros(
                1,
                1,
                self.embed_dim,
            )
        )

        # +1 for CLS token
        self.spectral_pos_embed = nn.Parameter(
            torch.zeros(
                1,
                self.num_freq_patches + 1,
                self.embed_dim,
            )
        )

        self.spectral_pos_drop = nn.Dropout(
            dropout
        )

        # ====================================================
        # Spectral Transformer
        # ====================================================

        spectral_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.spectral_transformer = (
            nn.TransformerEncoder(
                spectral_layer,
                num_layers=self.spectral_depth,
            )
        )

        self.spectral_norm = nn.LayerNorm(
            self.embed_dim
        )

        # ====================================================
        # Temporal positional embedding
        #
        # One position per 30-second epoch.
        # ====================================================

        self.temporal_pos_embed = nn.Parameter(
            torch.zeros(
                1,
                self.num_epochs,
                self.embed_dim,
            )
        )

        self.temporal_pos_drop = nn.Dropout(
            dropout
        )

        # ====================================================
        # Temporal Transformer
        # ====================================================

        temporal_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.temporal_transformer = (
            nn.TransformerEncoder(
                temporal_layer,
                num_layers=self.temporal_depth,
            )
        )

        self.temporal_norm = nn.LayerNorm(
            self.embed_dim
        )

        # ====================================================
        # Sleep-stage classifier
        # ====================================================

        self.classifier = nn.Linear(
            self.embed_dim,
            self.num_classes,
        )

        self._init_weights()

    # ========================================================
    # Initialization
    # ========================================================

    def _init_weights(
        self,
    ) -> None:

        nn.init.trunc_normal_(
            self.spectral_cls_token,
            std=0.02,
        )

        nn.init.trunc_normal_(
            self.spectral_pos_embed,
            std=0.02,
        )

        nn.init.trunc_normal_(
            self.temporal_pos_embed,
            std=0.02,
        )

        for module in self.modules():

            if isinstance(
                module,
                nn.Linear,
            ):

                nn.init.trunc_normal_(
                    module.weight,
                    std=0.02,
                )

                if module.bias is not None:
                    nn.init.zeros_(
                        module.bias
                    )

            elif isinstance(
                module,
                nn.Conv2d,
            ):

                nn.init.trunc_normal_(
                    module.weight,
                    std=0.02,
                )

                if module.bias is not None:
                    nn.init.zeros_(
                        module.bias
                    )

    # ========================================================
    # Spectral encoder
    # ========================================================

    def encode_spectral(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        # ----------------------------------------------------
        # Input:
        # (B, 1, 256, 512)
        # ----------------------------------------------------

        x = self.patch_embed(
            x
        )

        # ----------------------------------------------------
        # x:
        # (B, D, F, T)
        #
        # D = 192
        # F = 16 spectral patches
        # T = 512 epochs
        # ----------------------------------------------------

        batch_size = x.shape[0]
        embed_dim = x.shape[1]
        freq_tokens = x.shape[2]
        time_steps = x.shape[3]

        if freq_tokens != self.num_freq_patches:

            raise RuntimeError(
                f"Expected {self.num_freq_patches} "
                f"frequency patches, got {freq_tokens}"
            )

        if time_steps != self.num_epochs:

            raise RuntimeError(
                f"Expected {self.num_epochs} epochs, "
                f"got {time_steps}"
            )

        # ----------------------------------------------------
        # Rearrange:
        #
        # (B, D, F, T)
        # ->
        # (B, T, F, D)
        # ->
        # (B*T, F, D)
        #
        # Each temporal epoch becomes an independent
        # spectral token sequence.
        # ----------------------------------------------------

        x = x.permute(
            0,
            3,
            2,
            1,
        ).contiguous()

        x = x.view(
            batch_size * time_steps,
            freq_tokens,
            embed_dim,
        )

        x = self.patch_norm(
            x
        )

        # ----------------------------------------------------
        # Add spectral CLS token
        # ----------------------------------------------------

        cls_token = (
            self.spectral_cls_token.expand(
                batch_size * time_steps,
                -1,
                -1,
            )
        )

        x = torch.cat(
            (
                cls_token,
                x,
            ),
            dim=1,
        )

        x = (
            x
            + self.spectral_pos_embed
        )

        x = self.spectral_pos_drop(
            x
        )

        # ----------------------------------------------------
        # Spectral Transformer
        # ----------------------------------------------------

        x = self.spectral_transformer(
            x
        )

        x = self.spectral_norm(
            x
        )

        # ----------------------------------------------------
        # Keep CLS representation
        #
        # (B*T, 17, D)
        # ->
        # (B*T, D)
        # ----------------------------------------------------

        x = x[:, 0]

        # ----------------------------------------------------
        # Restore temporal sequence
        #
        # (B*T, D)
        # ->
        # (B, T, D)
        # ----------------------------------------------------

        x = x.view(
            batch_size,
            time_steps,
            embed_dim,
        )

        return x

    # ========================================================
    # Forward
    # ========================================================

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        if x.ndim != 4:

            raise ValueError(
                "Expected input shape "
                "(B, 1, 256, 512), "
                f"got {tuple(x.shape)}"
            )

        if x.shape[1] != 1:

            raise ValueError(
                "Expected one EEG channel, "
                f"got {x.shape[1]}"
            )

        # ----------------------------------------------------
        # Spectral representation
        #
        # (B, 1, 256, 512)
        # ->
        # (B, 512, 192)
        # ----------------------------------------------------

        x = self.encode_spectral(
            x
        )

        # ----------------------------------------------------
        # Temporal positional embeddings
        # ----------------------------------------------------

        x = (
            x
            + self.temporal_pos_embed
        )

        x = self.temporal_pos_drop(
            x
        )

        # ----------------------------------------------------
        # Temporal Transformer
        # ----------------------------------------------------

        x = self.temporal_transformer(
            x
        )

        x = self.temporal_norm(
            x
        )

        # ----------------------------------------------------
        # Per-epoch sleep-stage prediction
        #
        # (B, 512, 192)
        # ->
        # (B, 512, 4)
        # ----------------------------------------------------

        logits = self.classifier(
            x
        )

        return logits

    # ========================================================
    # Config
    # ========================================================

    def model_config(
        self,
    ) -> dict:

        return {
            "freq_bins": self.freq_bins,
            "num_epochs": self.num_epochs,
            "freq_patch_size": self.freq_patch_size,
            "num_freq_patches": self.num_freq_patches,
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "spectral_depth": self.spectral_depth,
            "temporal_depth": self.temporal_depth,
            "num_classes": self.num_classes,
        }
