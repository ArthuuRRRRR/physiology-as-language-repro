"""Préparation de la respiration comme dans le builder du réentraînement."""

import numpy as np
from scipy.signal import resample_poly


SOURCE_FS = 100
EPOCH_SECONDS = 30
WINDOW_EPOCHS = 512

# Une fenêtre = 256 minutes à 100 Hz.
RAW_WINDOW_SAMPLES = WINDOW_EPOCHS * EPOCH_SECONDS * SOURCE_FS


def validate_recording(signal, sampling_rate):
    """Vérifier le canal déjà préparé par le lecteur du dataset."""
    signal = np.asarray(signal)

    if sampling_rate != SOURCE_FS:
        raise ValueError(
            "Le builder du réentraînement reçoit un signal à 100 Hz. "
            "Il faut vérifier la préparation en amont du dataset."
        )

    if signal.ndim != 1 or signal.size == 0:
        raise ValueError("Le signal doit être un tableau 1D non vide.")

    if not np.issubdtype(signal.dtype, np.number):
        raise ValueError("Le signal doit contenir des nombres.")

    if np.iscomplexobj(signal) or not np.isfinite(signal).all():
        raise ValueError("Le signal contient des valeurs complexes, NaN ou inf.")

    signal = signal.astype(np.float32, copy=False)

    if not np.isfinite(signal).all():
        raise ValueError("Les valeurs dépassent la plage du float32.")

    return signal


def validate_stats(stats):
    """Lire les statistiques globales du dataset/canal."""
    mean = float(stats["mean"])
    std = float(stats["std"])

    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
        raise ValueError("Les statistiques doivent avoir une std positive.")

    return {"mean": mean, "std": std}


def window_layout(sample_count, tail="drop"):
    """Découper en fenêtres complètes, sans ajouter de padding."""
    if tail not in ("drop", "error"):
        raise ValueError("La gestion de la fin doit être 'drop' ou 'error'.")

    count, remainder = divmod(sample_count, RAW_WINDOW_SAMPLES)

    if tail == "error" and remainder:
        raise ValueError("Il reste des échantillons après la dernière fenêtre.")

    if count == 0:
        raise ValueError("Le signal ne contient aucune fenêtre de 256 minutes.")

    windows = [(i * WINDOW_EPOCHS, WINDOW_EPOCHS) for i in range(count)]
    return windows, remainder


def make_batch(signal, windows, stats):
    """Reprendre les opérations de extract_respiration(), dans le même ordre."""
    stats = validate_stats(stats)
    batch = np.empty((len(windows), 64, 2400), dtype=np.float32)

    for i, (start_epoch, valid_epochs) in enumerate(windows):
        if valid_epochs != WINDOW_EPOCHS:
            raise ValueError("Cette pipeline utilise des fenêtres complètes.")

        start = start_epoch * EPOCH_SECONDS * SOURCE_FS
        window = np.asarray(
            signal[start:start + RAW_WINDOW_SAMPLES],
            dtype=np.float32,
        )

        if window.size != RAW_WINDOW_SAMPLES:
            raise ValueError("La fenêtre dépasse la longueur du signal.")

        # Même forme et même normalisation globale que le builder.
        epochs = window.reshape(WINDOW_EPOCHS, EPOCH_SECONDS * SOURCE_FS)
        epochs = (epochs - stats["mean"]) / stats["std"]

        # Le rééchantillonnage se fait fenêtre par fenêtre.
        respiration = resample_poly(
            epochs.reshape(-1), up=1, down=10
        ).astype(np.float32)

        if not np.isfinite(respiration).all():
            raise ValueError("Le rééchantillonnage a produit des valeurs invalides.")

        batch[i] = respiration.reshape(64, 2400)

    return batch
