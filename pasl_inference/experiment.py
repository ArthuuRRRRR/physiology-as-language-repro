"""Retrouver les fichiers et vérifier l'expérience correspondant à une seed."""

import json
from pathlib import Path

from .preprocessing import validate_stats


SEEDS = (0, 1, 42)


def resolve_experiment(runs_root, seed):
    if seed not in SEEDS:
        raise ValueError(f"Seed inconnue : {seed}")

    folder = Path(runs_root) / f"seed_{seed}"

    return {
        "seed": seed,
        "transformer": folder / "transformer/checkpoint_best_mae.pt",
        "vqgan": folder / "vqgan/checkpoint_best_final.pt",
        "respiration_stats": folder / "transformer_data/respiration_stats.json",
    }


def check_checkpoint_metadata(checkpoint, experiment):
    training = checkpoint["training_config"]
    config = checkpoint["model_config"]
    seed = experiment["seed"]

    if training.get("seed") != seed:
        raise ValueError(
            f"Seed demandée : {seed}, "
            f"seed du checkpoint : {training.get('seed')}"
        )

    # Propriétés communes aux trois expériences inspectées sur ton serveur.
    expected = {
        "respiration_samples": 2400,
        "num_respiration_tokens": 64,
        "eeg_grid_height": 8,
        "eeg_grid_width": 64,
        "codebook_size": 8192,
        "embedding_dim": 768,
        "num_heads": 8,
        "num_encoder_layers": 8,
        "num_decoder_layers": 8,
        "mask_ratio_mu": 0.9,
        "temporal_attention_mask": True,
    }

    for name, value in expected.items():
        if config.get(name) != value:
            raise ValueError(
                f"Configuration inattendue : {name}={config.get(name)} "
                f"(attendu : {value})"
            )

    reference = Path(training.get("mae_vqgan_checkpoint", ""))
    expected_suffix = (
        f"seed_{seed}", "vqgan", "checkpoint_best_final.pt"
    )

    if reference.parts[-3:] != expected_suffix:
        raise ValueError(f"Référence VQGAN inattendue : {reference}")

    return config, training


def load_respiration_stats(path, key):
    with Path(path).open() as file:
        statistics = json.load(file)

    if key not in statistics:
        raise ValueError(
            f"Statistiques '{key}' absentes de {path}. "
            "Ne pas utiliser automatiquement celles d'un autre dataset."
        )

    return validate_stats(statistics[key])
