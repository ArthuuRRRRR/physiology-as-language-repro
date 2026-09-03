"""Self-attention decoder for the Physiology-as-Language reproduction."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class MAGEStyleMLMHead(nn.Module):
    """MAGE-style prediction head tied to the EEG token embeddings."""

    def __init__(
        self,
        embedding_dim: int,
        codebook_size: int,
        layer_norm_eps: float,
    ) -> None:
        super().__init__()
        self.dense = nn.Linear(embedding_dim, embedding_dim)
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(embedding_dim, eps=layer_norm_eps)
        self.bias = nn.Parameter(torch.zeros(codebook_size))

    def forward(self, features: Tensor, token_embedding_weight: Tensor) -> Tensor:
        features = self.norm(self.activation(self.dense(features)))
        return F.linear(features, token_embedding_weight, self.bias)


class PaperEEGTokenDecoder(nn.Module):
    def __init__(
        self,
        num_respiration_tokens: int = 64,
        eeg_grid_height: int = 8,
        eeg_grid_width: int = 64,
        codebook_size: int = 8192,
        embedding_dim: int = 768,
        num_heads: int = 8,
        num_layers: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()

        self.num_respiration_tokens = num_respiration_tokens
        self.eeg_grid_height = eeg_grid_height
        self.eeg_grid_width = eeg_grid_width
        self.num_eeg_tokens = eeg_grid_height * eeg_grid_width
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim

        self.encoder_to_decoder = nn.Linear(embedding_dim, embedding_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))

        # Separate learnable decoder positions follow the MAGE transition.
        self.respiration_position = nn.Parameter(
            torch.zeros(1, num_respiration_tokens, embedding_dim)
        )
        self.eeg_frequency_position = nn.Parameter(
            torch.zeros(1, eeg_grid_height, 1, embedding_dim)
        )
        self.eeg_time_position = nn.Parameter(
            torch.zeros(1, 1, eeg_grid_width, embedding_dim)
        )

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=int(embedding_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
            layer_norm_eps=layer_norm_eps,
        )
        # This is deliberately a self-attention stack, not a conventional
        # cross-attention TransformerDecoder.
        self.blocks = nn.TransformerEncoder(
            decoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(embedding_dim, eps=layer_norm_eps),
            enable_nested_tensor=False,
        )
        self.mlm_head = MAGEStyleMLMHead(
            embedding_dim=embedding_dim,
            codebook_size=codebook_size,
            layer_norm_eps=layer_norm_eps,
        )

        self.apply(self._initialize_module)
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.respiration_position, std=0.02)
        nn.init.normal_(self.eeg_frequency_position, std=0.02)
        nn.init.normal_(self.eeg_time_position, std=0.02)

    @staticmethod
    def _initialize_module(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _eeg_positions(self) -> Tensor:
        return (self.eeg_frequency_position + self.eeg_time_position).reshape(
            1,
            self.num_eeg_tokens,
            self.embedding_dim,
        )

    def _restore_joint_sequence(
        self,
        encoded: Tensor,
        drop_mask: Tensor,
        all_mask: Tensor,
    ) -> Tensor:
        encoded = self.encoder_to_decoder(encoded)
        batch_size = encoded.shape[0]

        respiration_features = encoded[:, : self.num_respiration_tokens]
        kept_eeg_features = encoded[:, self.num_respiration_tokens :]

        expected_kept = self.num_eeg_tokens - int(drop_mask[0].sum().item())
        if kept_eeg_features.shape[1] != expected_kept:
            raise ValueError(
                f"Expected {expected_kept} kept EEG encoder features, "
                f"got {kept_eeg_features.shape[1]}"
            )

        # Under CUDA autocast, encoder features are float16 while learned
        # parameters remain stored in float32.  Cast decoder-only tokens and
        # positions to the active feature dtype before indexed restoration.
        mask_tokens = self.mask_token.to(dtype=encoded.dtype).expand(
            batch_size,
            self.num_eeg_tokens,
            self.embedding_dim,
        )
        eeg_features = mask_tokens.clone()

        # First restore every position that was not physically dropped.
        keep_mask = ~drop_mask
        eeg_features[keep_mask] = kept_eeg_features.reshape(
            -1,
            self.embedding_dim,
        )

        # MAGE then replaces every masked position, including masked tokens
        # retained in the encoder, with the learned decoder mask token.
        eeg_features = torch.where(
            all_mask.unsqueeze(-1),
            mask_tokens,
            eeg_features,
        )

        respiration_features = respiration_features + self.respiration_position.to(
            dtype=encoded.dtype
        )
        eeg_features = eeg_features + self._eeg_positions().to(
            dtype=encoded.dtype
        )
        return torch.cat((respiration_features, eeg_features), dim=1)

    def forward(
        self,
        encoded: Tensor,
        drop_mask: Tensor,
        all_mask: Tensor,
        token_embedding_weight: Tensor,
    ) -> Tensor:
        decoder_input = self._restore_joint_sequence(
            encoded=encoded,
            drop_mask=drop_mask,
            all_mask=all_mask,
        )
        decoded = self.blocks(decoder_input)
        eeg_features = decoded[:, self.num_respiration_tokens :]

        # Only the 8192 real VQ codes are valid prediction targets.  The mask
        # embedding is excluded from the output vocabulary.
        return self.mlm_head(
            eeg_features,
            token_embedding_weight[: self.codebook_size],
        )
