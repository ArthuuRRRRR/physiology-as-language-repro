"""Comparer une vraie fenêtre MESA avec le cache du réentraînement."""

import gc
import json
from pathlib import Path

import numpy as np
import torch

from retraining_wav2sleep.code import build_transformer_dataset as builder

from paper_architecture.evaluate_paper_masked_transformer import (
    standardize_respiration as original_standardize,
    decode_token_ids as original_decode,
)

from .experiment import resolve_experiment, load_respiration_stats
from .models import load_models, standardize_respiration, decode_token_ids
from .preprocessing import make_batch


ROOT = Path("retraining_wav2sleep/outputs")


def find_mesa_window(folder):
    with (folder / "index.jsonl").open() as file:
        for line in file:
            if line.strip():
                row = json.loads(line)
                if row["dataset"].startswith("mesa"):
                    return row

    raise ValueError(f"Aucune fenêtre MESA dans {folder}")


@torch.inference_mode()
def predict(transformer, vqgan, respiration, device, standardize, decode):
    respiration = torch.from_numpy(
        np.array(respiration, dtype=np.float32, copy=True)
    ).to(device)
    respiration = standardize(respiration)

    # Même masque aléatoire pour comparer les deux chemins.
    torch.manual_seed(123)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(123)

    logits = transformer.predict_from_respiration(respiration)
    tokens = logits.argmax(dim=-1).reshape(-1, 8, 64)

    spectrogram = decode(vqgan, tokens)[:, 0].float().cpu().numpy()
    return tokens.cpu().numpy(), np.clip(spectrogram, 0, 1)


def check_seed(seed, device):
    print(f"\n=== SEED {seed} ===", flush=True)

    experiment = resolve_experiment(ROOT, seed)
    folder = ROOT / f"seed_{seed}/transformer_data/test"
    row = find_mesa_window(folder)

    print(
        f"Dataset : {row['dataset']}, fenêtre : {row['window_index']}",
        flush=True,
    )

    # Retrouver le signal source et les statistiques de cette expérience.
    stats_key = row.get("resp_stats_from", row["dataset"])
    stats = load_respiration_stats(
        experiment["respiration_stats"], stats_key
    )

    source = builder.ABDSource(row["dataset"])
    subject_start = int(source.subjects[row["subject_id"]]["pos"][0])
    start = subject_start + int(row["window_index"]) * 512

    raw_window = np.asarray(
        source.data[start:start + 512], dtype=np.float32
    ).reshape(-1)

    # Comparer la nouvelle préparation à la fonction originale.
    prepared = make_batch(raw_window, [(0, 512)], stats)
    original = builder.extract_respiration(row, source, stats)
    np.testing.assert_array_equal(prepared[0], original)

    # Comparer au cache réellement enregistré.
    metadata = json.loads((folder / "metadata.json").read_text())
    cache = np.memmap(
        folder / "respiration.mmap",
        dtype=np.float32,
        mode="r",
        shape=tuple(metadata["respiration"]["shape"]),
    )
    cached_window = np.array(
        cache[int(row["index"])], dtype=np.float32, copy=True
    )[None]

    np.testing.assert_array_equal(prepared, cached_window)
    print("Préparation = builder original = cache : OK", flush=True)

    # Comparer les prédictions avec les mêmes poids et la même graine.
    transformer, vqgan, _, _, _ = load_models(experiment, device)

    new_tokens, new_spec = predict(
        transformer, vqgan, prepared, device,
        standardize_respiration, decode_token_ids,
    )
    old_tokens, old_spec = predict(
        transformer, vqgan, cached_window, device,
        original_standardize, original_decode,
    )

    np.testing.assert_array_equal(new_tokens, old_tokens)
    np.testing.assert_allclose(new_spec, old_spec, rtol=1e-6, atol=1e-6)

    print("Tokens identiques : OK", flush=True)
    print(
        "Spectrogrammes concordants : OK "
        f"(écart maximal = {np.max(np.abs(new_spec - old_spec)):.3g})",
        flush=True,
    )

    del transformer, vqgan, source, cache
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Appareil utilisé : {device}", flush=True)

    for seed in (0, 1, 42):
        check_seed(seed, device)


if __name__ == "__main__":
    main()
