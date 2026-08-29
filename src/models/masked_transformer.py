import torch
import torch.nn as nn


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
    """

    def __init__(
        self,
        respiration_samples=2400,
        num_respiration_tokens=64,
        eeg_grid_height=8,
        eeg_grid_width=64,
        codebook_size=8192,
        embedding_dim=768,
        num_heads=8,
        num_encoder_layers=8,
        num_decoder_layers=8,
        mlp_ratio=4,
        dropout=0.1,
    ):
        super().__init__()

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

        # First eight bidirectional transformer blocks.
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

        # The paper specifies eight decoder blocks but does
        # not publish their detailed wiring.
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

        # Predict one of the 8192 EEG codebook IDs
        # at each EEG position.
        self.output_projection = nn.Linear(
            embedding_dim,
            codebook_size,
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
    ):
        respiration_embeddings = (
            self.embed_respiration(
                respiration
            )
        )

        eeg_embeddings = self.embed_eeg(
            masked_eeg_tokens
        )

        # Explicit cross-modal concatenation:
        #
        # (B, 64, 768) + (B, 512, 768)
        # -> (B, 576, 768)
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

        logits = self.output_projection(
            eeg_features
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
    - with probability 0.00, all EEG tokens are masked;
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

    # Reproduction adaptation:
    # force complete masking for 50% of examples.
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