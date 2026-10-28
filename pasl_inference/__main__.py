"""Générer des spectrogrammes avec une ou plusieurs seeds du réentraînement."""

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from .experiment import SEEDS, load_respiration_stats, resolve_experiment
from .preprocessing import make_batch, validate_recording, window_layout


def read_manifest(path):
    """Lire les chemins des signaux et les informations de leur canal."""
    path = Path(path).resolve()

    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required = {"recording_id", "path", "sampling_rate", "channel"}

        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Colonnes nécessaires : {sorted(required)}")

        rows = list(reader)

    if not rows:
        raise ValueError("Le manifest est vide.")

    seen = set()

    for row in rows:
        recording_id = row["recording_id"]

        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", recording_id or ""):
            raise ValueError(f"Identifiant invalide : {recording_id}")

        if recording_id in seen:
            raise ValueError(f"Identifiant répété : {recording_id}")

        seen.add(recording_id)
        row["source"] = (path.parent / row["path"]).resolve()
        row["sampling_rate"] = float(row["sampling_rate"])

        if row["source"].suffix.lower() != ".npy" or not row["source"].is_file():
            raise ValueError(f"Fichier NumPy introuvable : {row['source']}")

        if row["sampling_rate"] != 100:
            raise ValueError("Le signal préparé en amont doit être à 100 Hz.")

        if not row["channel"].strip():
            raise ValueError("Le nom du canal doit être renseigné.")

    if len({row["channel"] for row in rows}) != 1:
        raise ValueError("Utiliser un manifest et des statistiques par canal.")

    return rows


def read_recording(row, tail):
    signal = np.load(row["source"], mmap_mode="r", allow_pickle=False)
    signal = validate_recording(signal, row["sampling_rate"])
    windows, remainder = window_layout(signal.size, tail)
    return signal, windows, remainder


def sha256(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def generate(args, rows, experiment, statistics, statistics_file):
    # Import différé : le dry-run fonctionne sans charger PyTorch.
    import torch
    from .models import decode_token_ids, load_models, standardize_respiration

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA demandé mais indisponible.")

    transformer, vqgan, config, bounds, training = load_models(experiment, device)

    # Réinitialiser après le chargement, pour chaque expérience.
    torch.manual_seed(args.rng_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.rng_seed)

    seed = experiment["seed"]
    checkpoint_hashes = {
        name: sha256(experiment[name])
        for name in ("transformer", "vqgan")
    }

    for row in rows:
        signal, windows, remainder = read_recording(row, args.tail)
        folder = args.output / f"seed_{seed}" / row["recording_id"]
        folder.mkdir(parents=True, exist_ok=False)

        spectrograms = np.lib.format.open_memmap(
            folder / "spectrograms.npy",
            mode="w+",
            dtype=np.float32,
            shape=(len(windows), 256, 512),
        )
        index = []

        with torch.inference_mode():
            for start in range(0, len(windows), args.batch_size):
                current = windows[start:start + args.batch_size]
                respiration = make_batch(signal, current, statistics)
                respiration = torch.from_numpy(respiration).to(device)
                respiration = standardize_respiration(respiration)

                use_fp16 = device.type == "cuda" and args.precision == "fp16"

                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=use_fp16,
                ):
                    logits = transformer.predict_from_respiration(respiration)

                tokens = logits.argmax(dim=-1).reshape(-1, 8, 64)
                decoded = decode_token_ids(vqgan, tokens)
                decoded = decoded[:, 0].float().cpu().numpy()

                if decoded.shape != (len(current), 256, 512):
                    raise ValueError(f"Forme de sortie inattendue : {decoded.shape}")

                if not np.isfinite(decoded).all():
                    raise ValueError("Le modèle a produit des valeurs invalides.")

                spectrograms[start:start + len(current)] = np.clip(decoded, 0, 1)

                for number, (epoch, valid) in enumerate(current, start):
                    index.append({
                        "window_index": number,
                        "start_epoch": epoch,
                        "start_seconds": epoch * 30,
                        "valid_epochs": valid,
                    })

                spectrograms.flush()
                print(
                    f"seed={seed}, {row['recording_id']} : "
                    f"{start + len(current)}/{len(windows)} fenêtres",
                    flush=True,
                )

        del spectrograms

        with (folder / "index.jsonl").open("w") as file:
            for item in index:
                file.write(json.dumps(item) + "\n")

        metadata = {
            "training_seed": seed,
            "inference_rng_seed": args.rng_seed,
            "model_config": config,
            "checkpoint_training_config": training,
            "checkpoint_sha256": checkpoint_hashes,
            "respiration_stats": statistics,
            "respiration_stats_file": str(statistics_file),
            "respiration_stats_key": args.stats_key,
            "vqgan_db_bounds": bounds,
            "recording_id": row["recording_id"],
            "channel": row["channel"],
            "upstream_preprocessing": row.get("preprocessing", "unspecified"),
            "shape": [len(windows), 256, 512],
            "epoch_seconds": 30,
            "discarded_samples_100hz": remainder,
            "padding": False,
            "representation": "VQGAN-native, clipped to [0,1]",
            "batch_size": args.batch_size,
            "device": str(device),
            "precision": args.precision if device.type == "cuda" else "fp32",
            "torch_version": str(torch.__version__),
        }

        # La présence de metadata.json marque une génération terminée.
        (folder / "metadata.json").write_text(
            json.dumps(metadata, indent=2, allow_nan=False) + "\n"
        )

    del transformer, vqgan
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Inference PaSL par seed.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--seed", required=True, nargs="+", type=int, choices=SEEDS)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("retraining_wav2sleep/outputs"),
    )
    parser.add_argument("--stats-file", type=Path)
    parser.add_argument("--stats-key", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--rng-seed", type=int, default=0)
    parser.add_argument("--tail", choices=("drop", "error"), default="drop")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.batch_size < 1 or len(set(args.seed)) != len(args.seed):
        parser.error("Utiliser un batch positif et des seeds distinctes.")

    if not args.dry_run and args.output is None:
        parser.error("--output est nécessaire pour générer les spectrogrammes.")

    rows = read_manifest(args.manifest)
    experiments = []

    for seed in args.seed:
        experiment = resolve_experiment(args.runs_root, seed)
        stats_file = args.stats_file or experiment["respiration_stats"]
        statistics = load_respiration_stats(stats_file, args.stats_key)

        if not args.dry_run:
            for name in ("transformer", "vqgan"):
                if not experiment[name].is_file():
                    raise FileNotFoundError(experiment[name])

            for row in rows:
                folder = args.output / f"seed_{seed}" / row["recording_id"]
                if folder.exists():
                    raise FileExistsError(f"Sortie déjà présente : {folder}")

        experiments.append((experiment, statistics, stats_file))

    for experiment, statistics, stats_file in experiments:
        if args.dry_run:
            for row in rows:
                signal, windows, remainder = read_recording(row, args.tail)

                for window in windows:
                    make_batch(signal, [window], statistics)

                print(
                    f"seed={experiment['seed']}, {row['recording_id']} : "
                    f"{len(windows)} fenêtres, {remainder} échantillons exclus"
                )
        else:
            generate(args, rows, experiment, statistics, stats_file)


if __name__ == "__main__":
    main()
