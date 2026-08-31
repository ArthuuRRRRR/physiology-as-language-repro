import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RespirationToEEGTransformer(nn.Module):
    """
    Cross-modal masked Transformer for respiration-to-EEG
    translation.

    The model concatenates:

        64 respiration tokens
        + 512 masked EEG tokens
        = 576 joint tokens

    Shapes
    ------
    respiration:
        (B, 64, 2400)

    masked_eeg_tokens:
        (B, 8, 64) or (B, 512)

    output logits:
        (B, 512, 8192)

    When use_codebook_tied_output=True, each EEG feature is
    projected directly into the frozen VQGAN latent space. The
    8192 logits are cosine similarities with the real codebook,
    rather than outputs of an unrelated classification layer.
    """

    def __init__(
        self,
        respiration_samples=2400,
        num_respiration_tokens=64,
        eeg_grid_height=8,
        eeg_grid_width=64,
        codebook_size=8192,
        embedding_dim=384,
        num_heads=6,
        num_encoder_layers=4,
        num_decoder_layers=4,
        mlp_ratio=4,
        dropout=0.1,
        num_window_types=2,
        use_window_embedding=True,
        codebook_dim=32,
        use_codebook_tied_output=False,
        codebook_temperature=0.07,
    ):
        super().__init__()

        if embedding_dim < 1:
            raise ValueError(
                "embedding_dim must be positive"
            )

        if num_heads < 1:
            raise ValueError(
                "num_heads must be positive"
            )

        if num_encoder_layers < 1:
            raise ValueError(
                "num_encoder_layers must be positive"
            )

        if num_decoder_layers < 1:
            raise ValueError(
                "num_decoder_layers must be positive"
            )

        if mlp_ratio < 1:
            raise ValueError(
                "mlp_ratio must be positive"
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "dropout must be in [0, 1)"
            )

        if embedding_dim % num_heads != 0:
            raise ValueError(
                "embedding_dim must be divisible by num_heads"
            )

        if num_window_types < 1:
            raise ValueError(
                "num_window_types must be at least 1"
            )

        if codebook_dim < 1:
            raise ValueError(
                "codebook_dim must be positive"
            )

        if codebook_temperature <= 0:
            raise ValueError(
                "codebook_temperature must be positive"
            )

        self.num_respiration_tokens = (
            num_respiration_tokens
        )

        self.eeg_grid_height = (
            eeg_grid_height
        )

        self.eeg_grid_width = (
            eeg_grid_width
        )

        self.num_eeg_tokens = (
            eeg_grid_height
            * eeg_grid_width
        )

        self.codebook_size = (
            codebook_size
        )

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_encoder_layers = (
            num_encoder_layers
        )
        self.num_decoder_layers = (
            num_decoder_layers
        )
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        self.num_window_types = num_window_types
        self.use_window_embedding = (
            use_window_embedding
        )
        self.codebook_dim = codebook_dim
        self.use_codebook_tied_output = (
            use_codebook_tied_output
        )
        self.codebook_temperature = (
            codebook_temperature
        )
        self.respiration_samples = (
            respiration_samples
        )

        # EEG codebook IDs are 0 to 8191.
        # ID 8192 is reserved for [MASK].
        self.mask_token_id = (
            codebook_size
        )

        # Minimal linear projection of each raw
        # four-minute respiration segment.
        self.respiration_projection = nn.Linear(
            respiration_samples,
            embedding_dim,
        )

        # Learnable 1D temporal position for respiration.
        self.respiration_position = nn.Parameter(
            torch.zeros(
                1,
                num_respiration_tokens,
                embedding_dim,
            )
        )

        # 8192 EEG codes plus one mask token.
        self.eeg_token_embedding = nn.Embedding(
            codebook_size + 1,
            embedding_dim,
        )

        # Learnable 2D EEG positions:
        # frequency position + temporal position.
        self.eeg_frequency_position = nn.Parameter(
            torch.zeros(
                1,
                eeg_grid_height,
                1,
                embedding_dim,
            )
        )

        self.eeg_time_position = nn.Parameter(
            torch.zeros(
                1,
                1,
                eeg_grid_width,
                embedding_dim,
            )
        )

        if self.use_window_embedding:
            self.window_embedding = nn.Embedding(
                num_window_types,
                embedding_dim,
            )

        else:
            self.window_embedding = None

        feedforward_dim = (
            embedding_dim * mlp_ratio
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        # First bidirectional Transformer stack.
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
            norm=nn.LayerNorm(
                embedding_dim
            ),
            enable_nested_tensor=False,
        )

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        # The paper does not publish the detailed decoder
        # wiring used for this concatenated representation.
        #
        # We use a second bidirectional self-attention stack
        # over the same joint sequence, consistent with the
        # explicit concatenation described in the paper.
        self.decoder = nn.TransformerEncoder(
            decoder_layer,
            num_layers=num_decoder_layers,
            norm=nn.LayerNorm(
                embedding_dim
            ),
            enable_nested_tensor=False,
        )

        if self.use_codebook_tied_output:
            # Model V3: predict a continuous VQGAN latent.
            self.latent_projection = nn.Linear(
                embedding_dim,
                codebook_dim,
            )

            self.logit_scale = nn.Parameter(
                torch.tensor(
                    math.log(
                        1.0 / codebook_temperature
                    ),
                    dtype=torch.float32,
                )
            )

            self.output_projection = None

        else:
            # Legacy and V2 output head.
            self.output_projection = nn.Linear(
                embedding_dim,
                codebook_size,
            )

            self.latent_projection = None
            self.register_parameter(
                "logit_scale",
                None,
            )

        self._initialize_parameters()

    def _initialize_parameters(self):
        nn.init.trunc_normal_(
            self.respiration_position,
            std=0.02,
        )

        nn.init.trunc_normal_(
            self.eeg_frequency_position,
            std=0.02,
        )

        nn.init.trunc_normal_(
            self.eeg_time_position,
            std=0.02,
        )

        nn.init.normal_(
            self.eeg_token_embedding.weight,
            std=0.02,
        )

        if self.window_embedding is not None:
            nn.init.trunc_normal_(
                self.window_embedding.weight,
                std=0.02,
            )

    def get_config(self):
        """Return the constructor configuration saved in checkpoints."""

        return {
            "respiration_samples": (
                self.respiration_samples
            ),
            "num_respiration_tokens": (
                self.num_respiration_tokens
            ),
            "eeg_grid_height": self.eeg_grid_height,
            "eeg_grid_width": self.eeg_grid_width,
            "codebook_size": self.codebook_size,
            "embedding_dim": self.embedding_dim,
            "num_heads": self.num_heads,
            "num_encoder_layers": (
                self.num_encoder_layers
            ),
            "num_decoder_layers": (
                self.num_decoder_layers
            ),
            "mlp_ratio": self.mlp_ratio,
            "dropout": self.dropout,
            "num_window_types": (
                self.num_window_types
            ),
            "use_window_embedding": (
                self.use_window_embedding
            ),
            "codebook_dim": self.codebook_dim,
            "use_codebook_tied_output": (
                self.use_codebook_tied_output
            ),
            "codebook_temperature": (
                self.codebook_temperature
            ),
        }

    def embed_window_index(
        self,
        window_index,
        batch_size,
        device,
    ):
        """Return one global night-window embedding per sample."""

        if not self.use_window_embedding:
            return None

        if window_index is None:
            raise ValueError(
                "window_index is required when "
                "use_window_embedding=True"
            )

        window_index = torch.as_tensor(
            window_index,
            device=device,
            dtype=torch.long,
        ).reshape(-1)

        if window_index.numel() != batch_size:
            raise ValueError(
                "Expected one window_index per sample, got "
                f"{window_index.numel()} for batch size "
                f"{batch_size}"
            )

        if (
            (window_index < 0).any()
            or (
                window_index
                >= self.num_window_types
            ).any()
        ):
            raise ValueError(
                "window_index must be in [0, "
                f"{self.num_window_types - 1}], got "
                f"{window_index.detach().cpu().tolist()}"
            )

        return self.window_embedding(
            window_index
        ).unsqueeze(1)

    def embed_respiration(
        self,
        respiration,
    ):
        """
        Convert 64 raw respiration segments into
        64 continuous embeddings.
        """

        if respiration.ndim != 3:
            raise ValueError(
                "Respiration must have shape "
                "(B, 64, 2400), got "
                f"{tuple(respiration.shape)}"
            )

        if respiration.shape[1] != (
            self.num_respiration_tokens
        ):
            raise ValueError(
                "Expected "
                f"{self.num_respiration_tokens} "
                "respiration segments, got "
                f"{respiration.shape[1]}"
            )

        respiration_embeddings = (
            self.respiration_projection(
                respiration
            )
        )

        return (
            respiration_embeddings
            + self.respiration_position
        )

    def embed_eeg(
        self,
        eeg_tokens,
    ):
        """
        Convert the 512 discrete or masked EEG tokens
        into continuous embeddings with 2D positions.
        """

        if eeg_tokens.ndim == 3:
            eeg_tokens = eeg_tokens.flatten(
                start_dim=1
            )

        if (
            eeg_tokens.ndim != 2
            or eeg_tokens.shape[1]
            != self.num_eeg_tokens
        ):
            raise ValueError(
                "EEG tokens must have shape "
                "(B, 8, 64) or (B, 512), got "
                f"{tuple(eeg_tokens.shape)}"
            )

        eeg_embeddings = (
            self.eeg_token_embedding(
                eeg_tokens
            )
        )

        eeg_positions = (
            self.eeg_frequency_position
            + self.eeg_time_position
        ).reshape(
            1,
            self.num_eeg_tokens,
            -1,
        )

        return (
            eeg_embeddings
            + eeg_positions
        )

    def forward(
        self,
        respiration,
        masked_eeg_tokens,
        window_index=None,
        codebook=None,
        return_latents=False,
    ):
        respiration_embeddings = (
            self.embed_respiration(
                respiration
            )
        )

        eeg_embeddings = self.embed_eeg(
            masked_eeg_tokens
        )

        window_embeddings = self.embed_window_index(
            window_index=window_index,
            batch_size=respiration.shape[0],
            device=respiration.device,
        )

        if window_embeddings is not None:
            respiration_embeddings = (
                respiration_embeddings
                + window_embeddings
            )
            eeg_embeddings = (
                eeg_embeddings
                + window_embeddings
            )

        # Explicit cross-modal concatenation:
        #
        # (B, 64, D) + (B, 512, D)
        # -> (B, 576, D)
        joint_sequence = torch.cat(
            (
                respiration_embeddings,
                eeg_embeddings,
            ),
            dim=1,
        )

        # Every respiration and EEG position can attend
        # bidirectionally to every other position.
        joint_features = self.encoder(
            joint_sequence
        )

        joint_features = self.decoder(
            joint_features
        )

        # The first 64 outputs correspond to respiration.
        # Only the final 512 EEG outputs are classified.
        eeg_features = joint_features[
            :,
            self.num_respiration_tokens:,
        ]

        if self.use_codebook_tied_output:
            if codebook is None:
                raise ValueError(
                    "codebook is required when "
                    "use_codebook_tied_output=True"
                )

            if (
                codebook.ndim != 2
                or codebook.shape[0]
                != self.codebook_size
                or codebook.shape[1]
                != self.codebook_dim
            ):
                raise ValueError(
                    "Expected codebook shape "
                    f"({self.codebook_size}, "
                    f"{self.codebook_dim}), got "
                    f"{tuple(codebook.shape)}"
                )

            # Keep the tied-codebook head in float32. With 8192
            # similarities and a low temperature, float16 can
            # overflow during backward and make GradScaler skip
            # every optimizer update.
            with torch.autocast(
                device_type=eeg_features.device.type,
                enabled=False,
            ):
                predicted_latents = (
                    self.latent_projection(
                        eeg_features.float()
                    )
                )

                predicted_unit = F.normalize(
                    predicted_latents,
                    p=2,
                    dim=-1,
                    eps=1e-6,
                )

                codebook_unit = F.normalize(
                    codebook.float(),
                    p=2,
                    dim=-1,
                    eps=1e-6,
                )

                # Clamping follows the standard learned-temperature
                # practice and prevents numerical overflow.
                scale = self.logit_scale.exp().clamp(
                    max=100.0
                )

                logits = scale * torch.matmul(
                    predicted_unit,
                    codebook_unit.transpose(0, 1),
                )

            if return_latents:
                return logits, predicted_latents

            return logits

        logits = self.output_projection(
            eeg_features
        )

        if return_latents:
            raise ValueError(
                "return_latents requires the "
                "codebook-tied V3 output head"
            )

        return logits

def mask_eeg_tokens(
    eeg_tokens,
    mask_token_id=8192,
    mask_ratio_mean=0.55,
    mask_ratio_std=0.25,
    min_mask_ratio=0.50,
    max_mask_ratio=1.00,
    full_mask_probability=0.00,
):
    """
    Mask EEG tokens during training.

    For each training example:
    - with probability full_mask_probability, all EEG tokens
      are masked;
    - otherwise, the masking ratio follows the paper-inspired
      truncated Gaussian distribution.

    Input
    -----
    eeg_tokens:
        (B, 8, 64)

    Returns
    -------
    masked_tokens:
        (B, 8, 64)

    mask:
        Boolean tensor (B, 8, 64).
        True indicates a token used in the loss.
    """

    if not 0.0 <= full_mask_probability <= 1.0:
        raise ValueError(
            "full_mask_probability must be "
            "between 0 and 1"
        )

    batch_size = eeg_tokens.shape[0]

    flat_tokens = eeg_tokens.reshape(
        batch_size,
        -1,
    )

    num_tokens = flat_tokens.shape[1]

    # Paper-inspired random masking ratio.
    mask_ratios = (
        torch.randn(
            batch_size,
            device=eeg_tokens.device,
        )
        * mask_ratio_std
        + mask_ratio_mean
    )

    mask_ratios = mask_ratios.clamp(
        min=min_mask_ratio,
        max=max_mask_ratio,
    )

    # Reproduction adaptation: optionally force complete
    # masking so training matches respiration-only inference.
    force_full_mask = (
        torch.rand(
            batch_size,
            device=eeg_tokens.device,
        )
        < full_mask_probability
    )

    mask_ratios = torch.where(
        force_full_mask,
        torch.ones_like(mask_ratios),
        mask_ratios,
    )

    num_masked_tokens = torch.round(
        mask_ratios * num_tokens
    ).long()

    num_masked_tokens = (
        num_masked_tokens.clamp(
            min=1,
            max=num_tokens,
        )
    )

    # Generate a random ranking of token positions
    # independently for every example.
    random_values = torch.rand(
        batch_size,
        num_tokens,
        device=eeg_tokens.device,
    )

    random_ranks = (
        random_values
        .argsort(dim=1)
        .argsort(dim=1)
    )

    mask = (
        random_ranks
        < num_masked_tokens.unsqueeze(1)
    )

    masked_tokens = flat_tokens.clone()

    masked_tokens[mask] = mask_token_id

    masked_tokens = masked_tokens.reshape_as(
        eeg_tokens
    )

    mask = mask.reshape_as(
        eeg_tokens
    )

    return masked_tokens, mask, mask_ratios
