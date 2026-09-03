"""Complete paper-oriented respiration-to-EEG masked Transformer."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .paper_decoder import PaperEEGTokenDecoder
from .paper_encoder import PaperRespirationEEGEncoder


class PaperMaskedRespirationToEEGTransformer(nn.Module):
    """8-block encoder + 8-block decoder from Physiology as Language.

    Inputs
    ------
    respiration:
        Float tensor shaped ``(B, 64, 2400)`` for the current SHHS 10-Hz
        preprocessing.
    eeg_tokens:
        Ground-truth VQ token IDs shaped ``(B, 8, 64)`` or ``(B, 512)``.

    The forward pass performs MAGE-style masking internally and returns the
    masked-token cross-entropy required by the paper.
    """

    def __init__(
        self,
        respiration_samples: int = 2400,
        num_respiration_tokens: int = 64,
        eeg_grid_height: int = 8,
        eeg_grid_width: int = 64,
        codebook_size: int = 8192,
        embedding_dim: int = 768,
        num_heads: int = 8,
        num_encoder_layers: int = 8,
        num_decoder_layers: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-6,
        mask_ratio_min: float = 0.5,
        mask_ratio_max: float = 1.0,
        mask_ratio_mu: float = 0.55,
        mask_ratio_std: float = 0.25,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()

        self.num_eeg_tokens = eeg_grid_height * eeg_grid_width
        self.codebook_size = codebook_size
        self.label_smoothing = label_smoothing
        self.model_config = {
            "respiration_samples": respiration_samples,
            "num_respiration_tokens": num_respiration_tokens,
            "eeg_grid_height": eeg_grid_height,
            "eeg_grid_width": eeg_grid_width,
            "codebook_size": codebook_size,
            "embedding_dim": embedding_dim,
            "num_heads": num_heads,
            "num_encoder_layers": num_encoder_layers,
            "num_decoder_layers": num_decoder_layers,
            "mlp_ratio": mlp_ratio,
            "dropout": dropout,
            "layer_norm_eps": layer_norm_eps,
            "mask_ratio_min": mask_ratio_min,
            "mask_ratio_max": mask_ratio_max,
            "mask_ratio_mu": mask_ratio_mu,
            "mask_ratio_std": mask_ratio_std,
            "label_smoothing": label_smoothing,
        }

        self.encoder = PaperRespirationEEGEncoder(
            respiration_samples=respiration_samples,
            num_respiration_tokens=num_respiration_tokens,
            eeg_grid_height=eeg_grid_height,
            eeg_grid_width=eeg_grid_width,
            codebook_size=codebook_size,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_encoder_layers,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            layer_norm_eps=layer_norm_eps,
            mask_ratio_min=mask_ratio_min,
            mask_ratio_max=mask_ratio_max,
            mask_ratio_mu=mask_ratio_mu,
            mask_ratio_std=mask_ratio_std,
        )
        self.decoder = PaperEEGTokenDecoder(
            num_respiration_tokens=num_respiration_tokens,
            eeg_grid_height=eeg_grid_height,
            eeg_grid_width=eeg_grid_width,
            codebook_size=codebook_size,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_decoder_layers,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            layer_norm_eps=layer_norm_eps,
        )

    @property
    def mask_token_id(self) -> int:
        return self.encoder.mask_token_id

    def get_config(self) -> dict[str, int | float]:
        return dict(self.model_config)

    def masked_token_loss(
        self,
        logits: Tensor,
        targets: Tensor,
        all_mask: Tensor,
    ) -> Tensor:
        per_token_loss = F.cross_entropy(
            logits.reshape(-1, self.codebook_size),
            targets.reshape(-1),
            reduction="none",
            label_smoothing=self.label_smoothing,
        ).reshape_as(targets)

        denominator = all_mask.sum().clamp_min(1)
        return (per_token_loss * all_mask).sum() / denominator

    def forward(
        self,
        respiration: Tensor,
        eeg_tokens: Tensor,
        mask_ratio: float | None = None,
    ) -> dict[str, Tensor | float]:
        (
            encoded,
            targets,
            drop_mask,
            all_mask,
            used_mask_ratio,
        ) = self.encoder(
            respiration=respiration,
            eeg_tokens=eeg_tokens,
            mask_ratio=mask_ratio,
        )

        logits = self.decoder(
            encoded=encoded,
            drop_mask=drop_mask,
            all_mask=all_mask,
            token_embedding_weight=self.encoder.eeg_token_embedding.weight,
        )
        loss = self.masked_token_loss(
            logits=logits,
            targets=targets,
            all_mask=all_mask,
        )

        return {
            "loss": loss,
            "logits": logits,
            "targets": targets,
            "mask": all_mask,
            "drop_mask": drop_mask,
            "mask_ratio": used_mask_ratio,
        }

    @torch.no_grad()
    def predict_from_respiration(self, respiration: Tensor) -> Tensor:
        """Predict all 512 EEG-token logits from respiration alone."""
        dummy_eeg_tokens = torch.zeros(
            respiration.shape[0],
            self.num_eeg_tokens,
            dtype=torch.long,
            device=respiration.device,
        )
        outputs = self.forward(
            respiration=respiration,
            eeg_tokens=dummy_eeg_tokens,
            mask_ratio=1.0,
        )
        return outputs["logits"]
