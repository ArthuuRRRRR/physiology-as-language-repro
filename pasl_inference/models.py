"""Chargement des poids et petites fonctions utilisées pendant l'inférence."""

import torch
import torch.nn.functional as F

# Ce sont aussi les classes importées par le trainer de retraining_wav2sleep.
from paper_architecture import PaperMaskedRespirationToEEGTransformer
from src.models.vqgan import VQGAN

from .experiment import check_checkpoint_metadata


def load_state_dict(checkpoint, possible_keys):
    for key in possible_keys:
        if key in checkpoint:
            return checkpoint[key]

    raise KeyError("Aucun dictionnaire de poids reconnu dans le checkpoint.")


def standardize_respiration(respiration, eps=1e-6):
    """Même standardisation par fenêtre que pendant l'entraînement."""
    mean = respiration.mean(dim=(1, 2), keepdim=True)
    std = respiration.std(dim=(1, 2), keepdim=True, unbiased=False)
    return (respiration - mean) / (std + eps)


def decode_token_ids(vqgan, token_ids):
    """Reconstituer le spectrogramme avec le codebook normalisé du VQGAN."""
    codebook = F.normalize(vqgan.quantizer.codebook.weight, p=2, dim=1)
    features = F.embedding(token_ids, codebook)
    features = features.permute(0, 3, 1, 2).contiguous()
    return vqgan.decode(features)


def load_models(experiment, device):
    # Charger uniquement des checkpoints de confiance.
    checkpoint = torch.load(
        experiment["transformer"],
        map_location="cpu",
        weights_only=False,
    )

    config, training = check_checkpoint_metadata(checkpoint, experiment)

    # Toute la configuration vient du checkpoint, y compris le masque temporel.
    transformer = PaperMaskedRespirationToEEGTransformer(**config)
    transformer.load_state_dict(
        load_state_dict(
            checkpoint, ["model_state_dict", "transformer_state_dict"]
        ),
        strict=True,
    )
    del checkpoint
    transformer = transformer.to(device).eval()

    checkpoint = torch.load(
        experiment["vqgan"],
        map_location="cpu",
        weights_only=False,
    )

    vqgan = VQGAN()
    vqgan.load_state_dict(
        load_state_dict(
            checkpoint,
            [
                "model_state_dict",
                "vqgan_state_dict",
                "generator_state_dict",
                "state_dict",
            ],
        ),
        strict=True,
    )

    bounds = {
        name: float(checkpoint[name])
        for name in ("min_db", "max_db")
        if name in checkpoint
    }

    del checkpoint
    vqgan = vqgan.to(device).eval()

    return transformer, vqgan, config, bounds, training
