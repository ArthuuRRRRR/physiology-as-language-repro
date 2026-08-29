import torch
import torch.nn as nn


class MAGERespirationToEEGTransformer(nn.Module):
    """
    MAGE-style respiration-to-EEG Transformer.

    Encoder:
        respiration tokens + visible EEG tokens only.

    Decoder:
        encoded respiration + encoded visible EEG
        + learned mask tokens at missing EEG positions.

    Input
    -----
    respiration:
        (B, 64, 2400)

    masked_eeg_tokens:
        (B, 8, 64)

    Output
    ------
    logits:
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

        self.codebook_size = codebook_size
        self.mask_token_id = codebook_size

        # Raw four-minute respiration segment -> token.
        self.respiration_projection = nn.Linear(
            respiration_samples,
            embedding_dim,
        )

        # Encoder positions.
        self.respiration_position = nn.Parameter(
            torch.zeros(
                1,
                num_respiration_tokens,
                embedding_dim,
            )
        )

        self.eeg_token_embedding = nn.Embedding(
            codebook_size + 1,
            embedding_dim,
        )

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

        encoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=num_heads,
                dim_feedforward=feedforward_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
            norm=nn.LayerNorm(
                embedding_dim
            ),
            enable_nested_tensor=False,
        )

        # Projection from encoder space to decoder space.
        self.encoder_to_decoder = nn.Linear(
            embedding_dim,
            embedding_dim,
        )

        # Learned token inserted at missing EEG positions.
        self.decoder_mask_token = nn.Parameter(
            torch.zeros(
                1,
                1,
                embedding_dim,
            )
        )

        # Separate decoder positions.
        self.decoder_respiration_position = (
            nn.Parameter(
                torch.zeros(
                    1,
                    num_respiration_tokens,
                    embedding_dim,
                )
            )
        )

        self.decoder_eeg_frequency_position = (
            nn.Parameter(
                torch.zeros(
                    1,
                    eeg_grid_height,
                    1,
                    embedding_dim,
                )
            )
        )

        self.decoder_eeg_time_position = (
            nn.Parameter(
                torch.zeros(
                    1,
                    1,
                    eeg_grid_width,
                    embedding_dim,
                )
            )
        )

        decoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=num_heads,
                dim_feedforward=feedforward_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
        )

        # MAE/MAGE-style decoder:
        # self-attention over the restored complete sequence.
        self.decoder = nn.TransformerEncoder(
            decoder_layer,
            num_layers=num_decoder_layers,
            norm=nn.LayerNorm(
                embedding_dim
            ),
            enable_nested_tensor=False,
        )

        self.output_projection = nn.Linear(
            embedding_dim,
            codebook_size,
        )

        self._initialize_parameters()

    def _initialize_parameters(self):
        positional_parameters = (
            self.respiration_position,
            self.eeg_frequency_position,
            self.eeg_time_position,
            self.decoder_respiration_position,
            self.decoder_eeg_frequency_position,
            self.decoder_eeg_time_position,
            self.decoder_mask_token,
        )

        for parameter in positional_parameters:
            nn.init.trunc_normal_(
                parameter,
                std=0.02,
            )

        nn.init.normal_(
            self.eeg_token_embedding.weight,
            std=0.02,
        )

    def _encoder_eeg_positions(self):
        return (
            self.eeg_frequency_position
            + self.eeg_time_position
        ).reshape(
            1,
            self.num_eeg_tokens,
            -1,
        )

    def _decoder_eeg_positions(self):
        return (
            self.decoder_eeg_frequency_position
            + self.decoder_eeg_time_position
        ).reshape(
            1,
            self.num_eeg_tokens,
            -1,
        )

    def embed_respiration(
        self,
        respiration,
    ):
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

        return (
            self.respiration_projection(
                respiration
            )
            + self.respiration_position
        )

    def flatten_eeg_tokens(
        self,
        eeg_tokens,
    ):
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

        return eeg_tokens

    def pack_encoder_sequence(
        self,
        respiration_embeddings,
        flat_eeg_tokens,
    ):
        """
        Remove masked EEG tokens before the encoder.

        Because examples can have different mask ratios,
        sequences are padded inside the batch.
        """

        batch_size = (
            flat_eeg_tokens.shape[0]
        )

        visible_masks = (
            flat_eeg_tokens
            != self.mask_token_id
        )

        eeg_embeddings = (
            self.eeg_token_embedding(
                flat_eeg_tokens
            )
            + self._encoder_eeg_positions()
        )

        visible_indices = [
            torch.nonzero(
                visible_masks[batch_index],
                as_tuple=False,
            ).squeeze(1)
            for batch_index in range(
                batch_size
            )
        ]

        maximum_visible = max(
            (
                indices.numel()
                for indices in visible_indices
            ),
            default=0,
        )

        maximum_length = (
            self.num_respiration_tokens
            + maximum_visible
        )

        encoder_sequence = (
            respiration_embeddings.new_zeros(
                batch_size,
                maximum_length,
                respiration_embeddings.shape[-1],
            )
        )

        encoder_padding_mask = torch.ones(
            batch_size,
            maximum_length,
            dtype=torch.bool,
            device=(
                respiration_embeddings.device
            ),
        )

        # Respiration is always visible.
        encoder_sequence[
            :,
            :self.num_respiration_tokens,
        ] = respiration_embeddings

        encoder_padding_mask[
            :,
            :self.num_respiration_tokens,
        ] = False

        # Append only visible EEG tokens.
        for batch_index, indices in enumerate(
            visible_indices
        ):
            number_visible = indices.numel()

            if number_visible == 0:
                continue

            start = (
                self.num_respiration_tokens
            )

            stop = (
                start + number_visible
            )

            encoder_sequence[
                batch_index,
                start:stop,
            ] = eeg_embeddings[
                batch_index,
                indices,
            ]

            encoder_padding_mask[
                batch_index,
                start:stop,
            ] = False

        return (
            encoder_sequence,
            encoder_padding_mask,
            visible_indices,
        )

    def build_decoder_sequence(
        self,
        encoded_sequence,
        visible_indices,
    ):
        """
        Restore the complete 64 + 512 sequence.

        Visible EEG positions receive encoder features.
        Missing positions receive the learned mask token.
        """

        encoded_sequence = (
            self.encoder_to_decoder(
                encoded_sequence
            )
        )

        batch_size = (
            encoded_sequence.shape[0]
        )

        respiration_features = (
            encoded_sequence[
                :,
                :self.num_respiration_tokens,
            ]
            + self.decoder_respiration_position
        )

        decoder_eeg_positions = (
            self._decoder_eeg_positions()
        )

        eeg_features = (
            self.decoder_mask_token.expand(
                batch_size,
                self.num_eeg_tokens,
                -1,
            )
            + decoder_eeg_positions
        ).clone()

        for batch_index, indices in enumerate(
            visible_indices
        ):
            number_visible = indices.numel()

            if number_visible == 0:
                continue

            start = (
                self.num_respiration_tokens
            )

            stop = (
                start + number_visible
            )

            eeg_features[
                batch_index,
                indices,
            ] = (
                encoded_sequence[
                    batch_index,
                    start:stop,
                ]
                + decoder_eeg_positions[
                    0,
                    indices,
                ]
            )

        return torch.cat(
            (
                respiration_features,
                eeg_features,
            ),
            dim=1,
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

        flat_eeg_tokens = (
            self.flatten_eeg_tokens(
                masked_eeg_tokens
            )
        )

        (
            encoder_sequence,
            encoder_padding_mask,
            visible_indices,
        ) = self.pack_encoder_sequence(
            respiration_embeddings,
            flat_eeg_tokens,
        )

        # Encoder never sees mask tokens.
        encoded_sequence = self.encoder(
            encoder_sequence,
            src_key_padding_mask=(
                encoder_padding_mask
            ),
        )

        decoder_sequence = (
            self.build_decoder_sequence(
                encoded_sequence,
                visible_indices,
            )
        )

        # Decoder receives the complete restored sequence.
        decoded_sequence = self.decoder(
            decoder_sequence
        )

        eeg_features = decoded_sequence[
            :,
            self.num_respiration_tokens:,
        ]

        return self.output_projection(
            eeg_features
        )