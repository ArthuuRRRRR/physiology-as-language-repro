import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
            ),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
            ),
        )

    def forward(self, x):
        return x + self.block(x)


class EncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        pool_scale=None,
    ):
        super().__init__()

        self.projection = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )

        self.residual_layers = nn.Sequential(
            ResidualBlock(out_channels),
            ResidualBlock(out_channels),
        )

        self.pool_scale = pool_scale

    def forward(self, x):
        x = self.projection(x)
        x = self.residual_layers(x)

        if self.pool_scale is not None:
            x = F.avg_pool2d(
                x,
                kernel_size=self.pool_scale,
                stride=self.pool_scale,
            )

        return x


class EEGEncoder(nn.Module):
    """
    EEG spectrogram encoder.

    Input:
        (B, 1, 256, 512)

    Output:
        (B, 32, 8, 64)
    """

    def __init__(self):
        super().__init__()

        self.blocks = nn.Sequential(
            EncoderBlock(
                1,
                64,
                pool_scale=(4, 2),
            ),
            EncoderBlock(
                64,
                128,
                pool_scale=(2, 2),
            ),
            EncoderBlock(
                128,
                256,
                pool_scale=(2, 2),
            ),
            EncoderBlock(
                256,
                256,
                pool_scale=(2, 1),
            ),
            EncoderBlock(
                256,
                32,
                pool_scale=None,
            ),
        )

    def forward(self, x):
        return self.blocks(x)


class VectorQuantizer(nn.Module):
    """
    Vector quantization with an EEG codebook.

    Input:
        (B, 32, 8, 64)

    Output:
        quantized : (B, 32, 8, 64)
        indices   : (B, 8, 64)
        loss      : scalar
    """

    def __init__(
        self,
        num_embeddings=8192,
        embedding_dim=32,
        commitment_beta=0.25,
        chunk_size=4096,
    ):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_beta = commitment_beta
        self.chunk_size = chunk_size

        self.codebook = nn.Embedding(
            num_embeddings,
            embedding_dim,
        )

        nn.init.uniform_(
            self.codebook.weight,
            -1.0 / num_embeddings,
            1.0 / num_embeddings,
        )

    def forward(self, z):
        # B, C, H, W
        b, c, h, w = z.shape

        if c != self.embedding_dim:
            raise ValueError(
                f"Expected latent dimension {self.embedding_dim}, got {c}"
            )

        # (B, C, H, W)
        # ->
        # (B, H, W, C)
        # ->
        # (B*H*W, C)
        z_flat = (
            z.permute(0, 2, 3, 1)
            .contiguous()
            .view(-1, c)
        )

        # L2 normalization during quantization
        z_norm = F.normalize(
            z_flat,
            p=2,
            dim=1,
        )

        codebook_norm = F.normalize(
            self.codebook.weight,
            p=2,
            dim=1,
        )

        indices_chunks = []

        # Chunked computation to avoid using too much GPU memory
        for start in range(
            0,
            z_norm.shape[0],
            self.chunk_size,
        ):
            z_chunk = z_norm[
                start:start + self.chunk_size
            ]

            # Squared Euclidean distance:
            # ||z - e||²
            distances = (
                torch.sum(
                    z_chunk ** 2,
                    dim=1,
                    keepdim=True,
                )
                + torch.sum(
                    codebook_norm ** 2,
                    dim=1,
                ).unsqueeze(0)
                - 2 * z_chunk @ codebook_norm.t()
            )

            indices = torch.argmin(
                distances,
                dim=1,
            )

            indices_chunks.append(indices)

        indices = torch.cat(
            indices_chunks,
            dim=0,
        )

        # Retrieve selected codebook vectors
        quantized = codebook_norm[indices]

        # Standard VQ losses
        codebook_loss = F.mse_loss(
            quantized,
            z_norm.detach(),
        )

        commitment_loss = F.mse_loss(
            z_norm,
            quantized.detach(),
        )

        vq_loss = (
            codebook_loss
            + self.commitment_beta * commitment_loss
        )

        # Straight-through estimator
        quantized = (
            z_norm
            + (quantized - z_norm).detach()
        )

        # Restore spatial dimensions
        quantized = (
            quantized
            .view(b, h, w, c)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        indices = indices.view(
            b,
            h,
            w,
        )

        return quantized, indices, vq_loss



class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        upsample_scale=None,
    ):
        super().__init__()

        self.residual_layers = nn.Sequential(
            ResidualBlock(in_channels),
            ResidualBlock(in_channels),
        )

        self.projection = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )

        self.upsample_scale = upsample_scale

    def forward(self, x):
        x = self.residual_layers(x)

        if self.upsample_scale is not None:
            x = F.interpolate(
                x,
                scale_factor=self.upsample_scale,
                mode="nearest",
            )

        x = self.projection(x)

        return x


class EEGDecoder(nn.Module):
    """
    EEG spectrogram decoder.

    Input:
        (B, 32, 8, 64)

    Output:
        (B, 1, 256, 512)
    """

    def __init__(self):
        super().__init__()

        self.blocks = nn.Sequential(
            DecoderBlock(
                32,
                256,
                upsample_scale=None,
            ),
            DecoderBlock(
                256,
                256,
                upsample_scale=(2, 1),
            ),
            DecoderBlock(
                256,
                128,
                upsample_scale=(2, 2),
            ),
            DecoderBlock(
                128,
                64,
                upsample_scale=(2, 2),
            ),
            DecoderBlock(
                64,
                1,
                upsample_scale=(4, 2),
            ),
        )

    def forward(self, x):
        return self.blocks(x)



class VQGAN(nn.Module):
    """
    EEG VQGAN tokenizer.

    Input:
        eeg_spectrogram: (B, 1, 256, 512)

    Returns:
        reconstruction: (B, 1, 256, 512)
        indices:        (B, 8, 64)
        vq_loss:        scalar
    """

    def __init__(
        self,
        num_embeddings=8192,
        embedding_dim=32,
        commitment_beta=0.25,
    ):
        super().__init__()

        self.encoder = EEGEncoder()

        self.quantizer = VectorQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            commitment_beta=commitment_beta,
        )

        self.decoder = EEGDecoder()

    def forward(self, x):
        z = self.encoder(x)

        z_q, indices, vq_loss = self.quantizer(z)

        reconstruction = self.decoder(z_q)

        return reconstruction, indices, vq_loss

    def encode(self, x):
        """
        Convert EEG spectrogram into discrete token IDs.
        """
        z = self.encoder(x)

        z_q, indices, vq_loss = self.quantizer(z)

        return z_q, indices, vq_loss

    def decode(self, z_q):
        """
        Decode quantized latent representation.
        """
        return self.decoder(z_q)