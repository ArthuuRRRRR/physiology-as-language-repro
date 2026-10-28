# PaSL inference — wav2sleep retraining

Run from the repository root. Model definitions are imported from
paper_architecture and src/models/vqgan.py, as in the retraining trainer.

## Experiments
--seed selects training checkpoints: 0, 1, or 42.
--rng-seed controls inference randomness and does not select model weights.

Each selected seed loads:
- seed_N/transformer/checkpoint_best_mae.pt
- seed_N/vqgan/checkpoint_best_final.pt
- seed_N/transformer_data/respiration_stats.json

The default root is retraining_wav2sleep/outputs.
Use --runs-root to relocate these files.

## Input
CSV columns: recording_id,path,sampling_rate,channel
Optional column: preprocessing

Each .npy contains one continuous, upstream-preprocessed channel at 100 Hz.
Relative paths resolve against the CSV directory.
The retraining input channel is abdominal respiration.
Raw HSP channel mapping and upstream preparation still need verification.

Preparation follows the retraining builder:
complete 256-minute window -> dataset/channel global z-score ->
per-window resampling to 10 Hz -> 64 x 2400 blocks ->
per-window standardization before the Transformer.

No padding is applied. Trailing samples are dropped or rejected with --tail error.

## Example using an existing cohort's saved statistics
python -m pasl_inference --manifest records.csv --seed 0 1 42 --stats-key mesa_train --dry-run

python -m pasl_inference --manifest records.csv --seed 0 1 42 --stats-key mesa_train --output generated

## New cohort calibration
HSP statistics are not inferred from the existing training datasets.
If training-cohort calibration is the agreed experimental protocol:

python -m pasl_inference.estimate_stats --train-manifest hsp_train.csv --key hsp_train --output hsp_stats.json

python -m pasl_inference --manifest hsp_records.csv --seed 0 1 42 --stats-file hsp_stats.json --stats-key hsp_train --output generated_hsp

Fit statistics on training patients only, using the same upstream preparation.
Use training-fold-specific statistics for cross-validation.

## Outputs
output/seed_N/recording_id/ contains:
- spectrograms.npy: float32 [windows, 256, 512], VQGAN-native values in [0,1]
- index.jsonl: window times relative to the supplied array
- metadata.json: completion marker and experiment provenance

Time columns represent 30-second epochs.
These are synthesized representations, not measured EEG or absolute physical PSD.
Existing recording directories are never overwritten.

## Validation
python -m unittest discover -s pasl_inference/tests -v

The preparation test compares directly with the original retraining function.
Actual checkpoint loading, complete inference parity and the HSP adapter still
require validation on Arthur's server before collaborator handoff.
