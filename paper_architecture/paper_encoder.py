"""Encoder for the Physiology-as-Language reproduction and controlled ablations.

Paper-specified choices:
  * raw 4-minute respiration segments are mapped with one linear layer;
  * respiration and partially masked EEG tokens form one joint sequence;
  * learnable 1-D respiration and 2-D EEG positional embeddings;
  * 8 Transformer blocks, width 768, 8 attention heads.

The paper does not publish the exact encoder/decoder transition. The two-mask
mechanism follows the official MAGE implementation cited by the paper:
``all_mask`` selects every masked EEG token, while ``drop_mask`` selects the
fixed 50% subset physically removed before the encoder.

Experimental ablations are disabled by default. ``temporal_attention_mask``
restricts only cross-modal self-attention: respiration token t and EEG tokens
whose temporal column is t may exchange information, while cross-modal links
between different temporal columns are blocked. Respiration-respiration and
EEG-EEG attention remain global.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class PaperRespirationEEGEncoder(nn.Module):
    def __init__(
        self,
        respiration_samples: int = 2400,
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
        mask_ratio_min: float = 0.5,
        mask_ratio_max: float = 1.0,
        mask_ratio_mu: float = 0.55,
        mask_ratio_std: float = 0.25,
        shared_temporal_position: bool = False,
        temporal_alignment_tag: bool = False,
        temporal_attention_mask: bool = False,
    ) -> None:
        super().__init__()

        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")
        if not 0.0 <= mask_ratio_min <= mask_ratio_mu <= mask_ratio_max <= 1.0:
            raise ValueError("Invalid masking-ratio parameters")
        if num_respiration_tokens != eeg_grid_width:
            raise ValueError(
                "Temporal alignment requires num_respiration_tokens == "
                "eeg_grid_width"
            )
        if shared_temporal_position and temporal_alignment_tag:
            raise ValueError(
                "shared_temporal_position and temporal_alignment_tag are "
                "separate ablations and cannot be enabled together"
            )

        self.respiration_samples = respiration_samples
        self.num_respiration_tokens = num_respiration_tokens
        self.eeg_grid_height = eeg_grid_height
        self.eeg_grid_width = eeg_grid_width
        self.num_eeg_tokens = eeg_grid_height * eeg_grid_width
        self.codebook_size = codebook_size
        self.mask_token_id = codebook_size
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.shared_temporal_position = shared_temporal_position
        self.temporal_alignment_tag_enabled = temporal_alignment_tag
        self.temporal_attention_mask = temporal_attention_mask

        self.mask_ratio_min = mask_ratio_min
        self.mask_ratio_max = mask_ratio_max
        self.mask_ratio_mu = mask_ratio_mu
        self.mask_ratio_std = mask_ratio_std

        # Paper: one linear projection per raw 4-minute breathing segment.
        self.respiration_projection = nn.Linear(
            respiration_samples,
            embedding_dim,
        )

        if self.shared_temporal_position:
            self.register_parameter(
                "respiration_position",
                None,
            )
        else:
            self.respiration_position = nn.Parameter(
                torch.zeros(
                    1,
                    num_respiration_tokens,
                    embedding_dim,
                )
            )

        # One extra ID is reserved for the EEG mask token.
        self.eeg_token_embedding = nn.Embedding(
            codebook_size + 1,
            embedding_dim,
        )
        self.eeg_frequency_position = nn.Parameter(
            torch.zeros(1, eeg_grid_height, 1, embedding_dim)
        )

        if self.shared_temporal_position:
            self.register_parameter(
                "eeg_time_position",
                None,
            )
            self.shared_time_position = nn.Parameter(
                torch.zeros(
                    1,
                    num_respiration_tokens,
                    embedding_dim,
                )
            )
        else:
            self.eeg_time_position = nn.Parameter(
                torch.zeros(
                    1,
                    1,
                    eeg_grid_width,
                    embedding_dim,
                )
            )
            self.register_parameter(
                "shared_time_position",
                None,
            )

        # AlignmentTag ablation: a zero-initialized shared 4-minute identity.
        # It is deliberately separate from positional embeddings.
        if self.temporal_alignment_tag_enabled:
            self.temporal_alignment_tag = nn.Parameter(
                torch.zeros(
                    1,
                    num_respiration_tokens,
                    embedding_dim,
                )
            )
        else:
            self.register_parameter(
                "temporal_alignment_tag",
                None,
            )

        # LayerNorm/dropout and block internals follow MAGE/ViT because the
        # paper does not specify these details in the supplement.
        self.embedding_norm = nn.LayerNorm(
            embedding_dim,
            eps=layer_norm_eps,
        )
        self.embedding_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=int(embedding_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
            layer_norm_eps=layer_norm_eps,
        )
        self.blocks = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(
                embedding_dim,
                eps=layer_norm_eps,
            ),
            enable_nested_tensor=False,
        )

        self.apply(self._initialize_module)

        if self.respiration_position is not None:
            nn.init.normal_(
                self.respiration_position,
                std=0.02,
            )

        nn.init.normal_(
            self.eeg_frequency_position,
            std=0.02,
        )

        if self.eeg_time_position is not None:
            nn.init.normal_(
                self.eeg_time_position,
                std=0.02,
            )

        if self.shared_time_position is not None:
            nn.init.normal_(
                self.shared_time_position,
                std=0.02,
            )

        # IMPORTANT: temporal_alignment_tag stays exactly zero at
        # initialization so enabling the tag can preserve the baseline
        # function before fine-tuning.

        nn.init.normal_(
            self.eeg_token_embedding.weight,
            std=0.02,
        )

    @staticmethod
    def _initialize_module(module: nn.Module) -> None:
        # MAGE uses the official JAX ViT Xavier initialization for Linear.
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _eeg_positions(self) -> Tensor:
        if self.shared_temporal_position:
            time_position = (
                self.shared_time_position[:, None, :, :]
            )
        else:
            time_position = self.eeg_time_position

        return (
            self.eeg_frequency_position
            + time_position
        ).reshape(
            1,
            self.num_eeg_tokens,
            self.embedding_dim,
        )

    def _eeg_alignment_tags(self) -> Tensor:
        if self.temporal_alignment_tag is None:
            raise RuntimeError(
                "Temporal alignment tags are not enabled"
            )

        # Flattening is frequency-major:
        # EEG[f, t] -> f * eeg_grid_width + t.
        return (
            self.temporal_alignment_tag[:, None, :, :]
            .expand(
                -1,
                self.eeg_grid_height,
                -1,
                -1,
            )
            .reshape(
                1,
                self.num_eeg_tokens,
                self.embedding_dim,
            )
        )

    def _flatten_eeg_tokens(
        self,
        eeg_tokens: Tensor,
    ) -> Tensor:
        if eeg_tokens.ndim == 3:
            expected = (
                self.eeg_grid_height,
                self.eeg_grid_width,
            )
            if tuple(eeg_tokens.shape[1:]) != expected:
                raise ValueError(
                    f"Expected EEG grid (B, {expected[0]}, {expected[1]}), "
                    f"got {tuple(eeg_tokens.shape)}"
                )
            eeg_tokens = eeg_tokens.flatten(
                start_dim=1
            )

        if (
            eeg_tokens.ndim != 2
            or eeg_tokens.shape[1]
            != self.num_eeg_tokens
        ):
            raise ValueError(
                f"Expected EEG tokens (B, {self.num_eeg_tokens}), "
                f"got {tuple(eeg_tokens.shape)}"
            )

        eeg_tokens = eeg_tokens.long()
        if (
            torch.any(eeg_tokens < 0)
            or torch.any(
                eeg_tokens >= self.codebook_size
            )
        ):
            raise ValueError(
                "Ground-truth EEG token IDs must be in "
                f"[0, {self.codebook_size - 1}]"
            )
        return eeg_tokens

    def _sample_truncated_mask_ratio(self) -> float:
        """Sample the MAGE truncated Gaussian without SciPy."""
        while True:
            value = torch.empty(()).normal_(
                mean=self.mask_ratio_mu,
                std=self.mask_ratio_std,
            )
            ratio = float(value.item())
            if (
                self.mask_ratio_min
                <= ratio
                <= self.mask_ratio_max
            ):
                return ratio

    def _make_masks(
        self,
        batch_size: int,
        device: torch.device,
        mask_ratio: float | None,
    ) -> tuple[Tensor, Tensor, float]:
        if mask_ratio is None:
            mask_ratio = (
                self._sample_truncated_mask_ratio()
            )
        mask_ratio = float(mask_ratio)

        if not (
            self.mask_ratio_min
            <= mask_ratio
            <= self.mask_ratio_max
        ):
            raise ValueError(
                f"mask_ratio must be in [{self.mask_ratio_min}, "
                f"{self.mask_ratio_max}], got {mask_ratio}"
            )

        num_dropped = math.ceil(
            self.num_eeg_tokens
            * self.mask_ratio_min
        )
        num_masked = math.ceil(
            self.num_eeg_tokens
            * mask_ratio
        )

        # Same ratio/count across the batch; independent random order per
        # example, as in MAGE.
        noise = torch.rand(
            batch_size,
            self.num_eeg_tokens,
            device=device,
        )
        order = torch.argsort(
            noise,
            dim=1,
        )
        ranks = torch.argsort(
            order,
            dim=1,
        )

        drop_mask = ranks < num_dropped
        all_mask = ranks < num_masked
        return drop_mask, all_mask, mask_ratio

    def _build_temporal_cross_modal_attention_mask(
        self,
        drop_mask: Tensor,
    ) -> Tensor:
        """Build the encoder hard temporal cross-modal attention mask.

        PyTorch boolean attention-mask semantics:
            False -> attention is allowed
            True  -> attention is blocked

        The encoder contains all 64 respiration tokens but only the EEG
        positions that survived MAGE's physical ``drop_mask``. Because those
        surviving EEG positions differ for every sample, this mask is
        batch-specific and is expanded to ``B * num_heads`` as required by
        MultiheadAttention for a 3-D attention mask.

        Allowed:
            Resp[t] <-> Resp[*]      (global respiration attention)
            EEG[*]  <-> EEG[*]       (global EEG attention)
            Resp[t] <-> EEG[f, t]    (same 4-minute interval)

        Blocked:
            Resp[t] <-> EEG[f, s] for s != t
        """
        if drop_mask.ndim != 2:
            raise ValueError(
                "drop_mask must have shape "
                f"(B, {self.num_eeg_tokens})"
            )
        if drop_mask.shape[1] != self.num_eeg_tokens:
            raise ValueError(
                "Unexpected drop_mask width: "
                f"{tuple(drop_mask.shape)}"
            )

        batch_size = drop_mask.shape[0]
        keep_mask = ~drop_mask
        kept_count = int(
            keep_mask[0].sum().item()
        )

        if not torch.all(
            keep_mask.sum(dim=1) == kept_count
        ):
            raise ValueError(
                "Every sample must keep the same number "
                "of EEG tokens"
            )

        original_indices = (
            torch.arange(
                self.num_eeg_tokens,
                device=drop_mask.device,
            )
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        kept_indices = original_indices[
            keep_mask
        ].reshape(
            batch_size,
            kept_count,
        )

        # Frequency-major flattening means index % width is temporal column.
        kept_times = (
            kept_indices
            % self.eeg_grid_width
        )
        respiration_times = torch.arange(
            self.num_respiration_tokens,
            device=drop_mask.device,
        )

        same_time = (
            respiration_times[
                None,
                :,
                None,
            ]
            == kept_times[
                :,
                None,
                :,
            ]
        )

        sequence_length = (
            self.num_respiration_tokens
            + kept_count
        )
        attention_mask = torch.zeros(
            batch_size,
            sequence_length,
            sequence_length,
            dtype=torch.bool,
            device=drop_mask.device,
        )

        # Respiration queries -> EEG keys.
        attention_mask[
            :,
            : self.num_respiration_tokens,
            self.num_respiration_tokens :,
        ] = ~same_time

        # EEG queries -> respiration keys.
        attention_mask[
            :,
            self.num_respiration_tokens :,
            : self.num_respiration_tokens,
        ] = ~same_time.transpose(1, 2)

        # MultiheadAttention accepts a sample-specific 3-D mask with shape
        # (B * num_heads, target_len, source_len).
        attention_mask = (
            attention_mask[:, None, :, :]
            .expand(
                batch_size,
                self.num_heads,
                sequence_length,
                sequence_length,
            )
            .reshape(
                batch_size * self.num_heads,
                sequence_length,
                sequence_length,
            )
        )
        return attention_mask

    def forward(
        self,
        respiration: Tensor,
        eeg_tokens: Tensor,
        mask_ratio: float | None = None,
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        float,
    ]:
        if respiration.ndim != 3:
            raise ValueError(
                "Respiration must have shape "
                f"(B, {self.num_respiration_tokens}, "
                f"{self.respiration_samples})"
            )
        if respiration.shape[1:] != (
            self.num_respiration_tokens,
            self.respiration_samples,
        ):
            raise ValueError(
                "Expected respiration shape "
                f"(B, {self.num_respiration_tokens}, "
                f"{self.respiration_samples}), "
                f"got {tuple(respiration.shape)}"
            )

        targets = self._flatten_eeg_tokens(
            eeg_tokens
        )
        if targets.shape[0] != respiration.shape[0]:
            raise ValueError(
                "Respiration and EEG batch sizes do not match"
            )

        batch_size = targets.shape[0]
        (
            drop_mask,
            all_mask,
            used_mask_ratio,
        ) = self._make_masks(
            batch_size=batch_size,
            device=targets.device,
            mask_ratio=mask_ratio,
        )

        masked_ids = targets.masked_fill(
            all_mask,
            self.mask_token_id,
        )

        if self.shared_temporal_position:
            respiration_position = (
                self.shared_time_position
            )
        else:
            respiration_position = (
                self.respiration_position
            )

        respiration_embeddings = (
            self.respiration_projection(
                respiration
            )
            + respiration_position
        )
        eeg_embeddings = (
            self.eeg_token_embedding(
                masked_ids
            )
            + self._eeg_positions()
        )

        if self.temporal_alignment_tag_enabled:
            respiration_embeddings = (
                respiration_embeddings
                + self.temporal_alignment_tag
            )
            eeg_embeddings = (
                eeg_embeddings
                + self._eeg_alignment_tags()
            )

        # Normalize the complete joint embedding before dropping the fixed
        # MAGE subset of EEG positions.
        joint_embeddings = torch.cat(
            (
                respiration_embeddings,
                eeg_embeddings,
            ),
            dim=1,
        )
        joint_embeddings = (
            self.embedding_dropout(
                self.embedding_norm(
                    joint_embeddings
                )
            )
        )

        respiration_embeddings = joint_embeddings[
            :,
            : self.num_respiration_tokens,
        ]
        eeg_embeddings = joint_embeddings[
            :,
            self.num_respiration_tokens :,
        ]

        keep_mask = ~drop_mask
        kept_eeg_embeddings = eeg_embeddings[
            keep_mask
        ].reshape(
            batch_size,
            self.num_eeg_tokens
            - int(
                drop_mask[0].sum().item()
            ),
            self.embedding_dim,
        )
        encoder_input = torch.cat(
            (
                respiration_embeddings,
                kept_eeg_embeddings,
            ),
            dim=1,
        )

        if self.temporal_attention_mask:
            attention_mask = (
                self._build_temporal_cross_modal_attention_mask(
                    drop_mask
                )
            )
            encoded = self.blocks(
                encoder_input,
                mask=attention_mask,
            )
        else:
            encoded = self.blocks(
                encoder_input
            )

        return (
            encoded,
            targets,
            drop_mask,
            all_mask,
            used_mask_ratio,
        )
