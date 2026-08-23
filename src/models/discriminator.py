import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchDiscriminator(nn.Module):
    """
    PatchGAN discriminator for EEG spectrograms.

    Input
    -----
    (B, 1, 256, 512)

    Output
    ------
    Patch-level real/fake logits.

    Instead of producing one score for the entire spectrogram,
    the discriminator evaluates local spectro-temporal patches.
    """

    def __init__(
        self,
        in_channels=1,
        base_channels=64,
    ):
        super().__init__()

        self.model = nn.Sequential(

            # 256 x 512
            nn.Conv2d(
                in_channels,
                base_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.LeakyReLU(
                0.2,
                inplace=True,
            ),

            # 128 x 256
            nn.Conv2d(
                base_channels,
                base_channels * 2,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(
                base_channels * 2
            ),
            nn.LeakyReLU(
                0.2,
                inplace=True,
            ),

            # 64 x 128
            nn.Conv2d(
                base_channels * 2,
                base_channels * 4,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(
                base_channels * 4
            ),
            nn.LeakyReLU(
                0.2,
                inplace=True,
            ),

            # 32 x 64
            nn.Conv2d(
                base_channels * 4,
                base_channels * 8,
                kernel_size=4,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(
                base_channels * 8
            ),
            nn.LeakyReLU(
                0.2,
                inplace=True,
            ),

            # Patch-level logits
            nn.Conv2d(
                base_channels * 8,
                1,
                kernel_size=4,
                stride=1,
                padding=1,
            ),
        )

    def forward(self, x):
        return self.model(x)


def discriminator_hinge_loss(
    logits_real,
    logits_fake,
):
    """
    Standard hinge discriminator loss used in VQGAN.
    """

    loss_real = torch.mean(
        F.relu(1.0 - logits_real)
    )

    loss_fake = torch.mean(
        F.relu(1.0 + logits_fake)
    )

    return 0.5 * (
        loss_real + loss_fake
    )


def generator_adversarial_loss(
    logits_fake,
):
    """
    Generator tries to make fake samples
    look real to the discriminator.
    """

    return -torch.mean(
        logits_fake
    )
