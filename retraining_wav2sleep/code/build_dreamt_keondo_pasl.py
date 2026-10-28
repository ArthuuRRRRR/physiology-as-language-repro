#!/usr/bin/env python3
"""Full-night DREAMT staging: Keondo-style GT (Welch) and PaSL synthetic EEG.

Run from ~/physiology_as_language_repro:
  python retraining_wav2sleep/code/build_dreamt_keondo_pasl.py --stage check
  python retraining_wav2sleep/code/build_dreamt_keondo_pasl.py --stage gt
  python retraining_wav2sleep/code/build_dreamt_keondo_pasl.py --stage synth --seed 42 --channel ABDOMEN

GT has the same exploratory Welch convention as build_cfs_keondo_gt.py,
not necessarily the exact VQGAN reference EEG spectrogram convention.
"""
from __future__ import annotations
import argparse
import json
import pickle
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.signal import welch, resample_poly

ROOT = Path('retraining_wav2sleep/outputs')
DATA = Path('/hdd2/kdpark/sleep_datasets/dreamt/preprocessing_output')
PROCESSED = DATA / 'processed'
PACKED = PROCESSED / 'mmap_signals_MAE_processed'
GT_ROOT = ROOT / 'downstream/dreamt_keondo_gt/dreamt'
SPLITS = ('train', 'val', 'test')
STAGE_MAP = np.array([0, 1, 1, 2, 3], dtype=np.int16)
N_EPOCHS = 512
EPOCH_SAMPLES = 3000
CHANNELS = {'ABDOMEN': ('ABD', 'ABDOMEN'), 'FLOW': ('AIRFLOW', 'FLOW'),
            'PTAF': ('AIRFLOW', 'PTAF')}


def channel_data(split, group, name):
    base = PACKED / split / group
    info_path = base / f'{name}.pickle'
    data_path = base / f'{name}.mmap'
    with info_path.open('rb') as f:
        meta = pickle.load(f)
    shape = tuple(int(v) for v in meta['data_shape'])
    if len(shape) != 2 or shape[1] != EPOCH_SAMPLES:
        raise ValueError(f'{split}/{group}/{name}: unexpected {shape}')
    if data_path.stat().st_size != np.prod(shape) * 4:
        raise ValueError(f'{data_path}: wrong byte size')
    mm = np.memmap(data_path, dtype=np.float32, mode='r', shape=shape)
    return mm, meta['sig_info']


def labels_for(split, sid, n):
    p = PROCESSED / 'annotations_split' / split / f'{sid}_ann.npy'
    original = np.load(p, allow_pickle=False)
    if original.shape != (n,) or not np.isin(original, [-1, 0, 1, 2, 3, 4]).all():
        raise ValueError(f'Unexpected annotations: {p} {original.shape}')
    result = np.full((n,), -100, dtype=np.int16)
    valid = (original >= 0) & (original <= 4)
    result[valid] = STAGE_MAP[original[valid].astype(np.int64)]
    return result


def quality_mask(split, group, name, sid, n):
    p = PROCESSED / 'quality_masks' / group / name / f'{sid}_valid.npy'
    if not p.exists():
        raise FileNotFoundError(f'Missing quality mask: {p}')
    result = np.load(p, allow_pickle=False)
    if result.shape != (n,):
        raise ValueError(f'Quality mask length mismatch: {p}')
    return result.astype(bool)


def segments(split, sid, n):
    """Preserve original epoch discontinuities if sample positions were saved."""
    position_dir = PROCESSED / 'epoch_start_samples'
    possible = sorted(position_dir.glob(f'{sid}*.npy')) if position_dir.exists() else []
    if not possible:
        if sid == 'S027':
            raise RuntimeError('S027 needs saved original epoch positions for safe segmentation')
        return [(0, n)], False
    if len(possible) != 1:
        raise RuntimeError(f'Ambiguous epoch position files for {sid}: {possible}')
    x = np.load(possible[0], allow_pickle=False).reshape(-1)
    if x.shape != (n,) or not np.issubdtype(x.dtype, np.number):
        raise ValueError(f'Invalid epoch positions: {possible[0]} {x.shape}')
    diff = np.diff(x.astype(np.float64))
    positive = diff[diff > 0]
    if not len(positive):
        raise ValueError(f'Non-monotonic epoch positions: {sid}')
    nominal = np.median(positive)
    # The saved positions can be in samples or seconds; infer a nominal epoch step.
    if not any(abs(nominal - value) <= max(1e-4, value * .01) for value in (30., 3000.)):
        raise ValueError(f'Unexpected epoch interval {sid}: {nominal}')
    cuts = np.flatnonzero(~np.isclose(diff, nominal, rtol=1e-3, atol=1e-3)) + 1
    bounds = [0, *cuts.tolist(), n]
    return list(zip(bounds[:-1], bounds[1:])), True


def split_rows(split, ids, eeg_info):
    rows = []
    labels_by_sid = {}
    for sid in ids:
        if sid not in eeg_info:
            raise KeyError(f'{split}: EEG missing {sid}')
        st, en = map(int, eeg_info[sid]['pos'])
        n = en - st
        y = labels_for(split, sid, n)
        valid_eeg = quality_mask(split, 'EEG', 'C4-M1', sid, n)
        y[~valid_eeg] = -100
        chunks, verified = segments(split, sid, n)
        labels_by_sid[sid] = y
        wi = 0
        for segment_id, (a, b) in enumerate(chunks):
            for offset in range(a, b, N_EPOCHS):
                valid = min(N_EPOCHS, b - offset)
                rows.append({'dataset':'dreamt', 'subject_id':sid,
                             'window_index':wi, 'start_epoch_in_trimmed_recording':offset,
                             'start_epoch':offset, 'start_sec':offset * 30,
                             'valid_epochs':valid, 'segment_id':segment_id,
                             'epoch_positions_verified':verified})
                wi += 1
    return rows, labels_by_sid


def read_ids():
    data = json.loads((DATA / 'dreamt_splits.json').read_text())
    ids = {s: [str(x) for x in data[s]] for s in SPLITS}
    for s in SPLITS:
        if len(ids[s]) != len(set(ids[s])) or not ids[s]:
            raise ValueError(f'Bad split: {s}')
    if len({x for v in ids.values() for x in v}) != sum(map(len, ids.values())):
        raise ValueError('Subject leakage between DREAMT splits')
    return ids


def make_gt_split(split, ids, check_only=False):
    eeg, info = channel_data(split, 'EEG', 'C4-M1')
    rows, labels_by_sid = split_rows(split, ids, info)
    # Consistency against ABDOMEN before allocating files.
    _, resp_info = channel_data(split, 'ABD', 'ABDOMEN')
    for sid in ids:
        if sid not in resp_info:
            raise KeyError(f'ABDOMEN missing {sid}')
        ra, rb = map(int, resp_info[sid]['pos'])
        ea, eb = map(int, info[sid]['pos'])
        if rb-ra != eb-ea:
            raise ValueError(f'{sid}: EEG/ABDOMEN epoch count mismatch')
    print(f'{split}: {len(ids)} subjects, {len(rows)} windows, '
          f'{sum(int(x["valid_epochs"]) for x in rows)} epochs, '
          f'{sum(not r["epoch_positions_verified"] for r in rows)} windows w/o positions', flush=True)
    if check_only:
        return
    dest = GT_ROOT / split
    if dest.exists():
        raise FileExistsError(f'GT output already exists: {dest}')
    dest.mkdir(parents=True)
    count = len(rows)
    gt = np.memmap(dest / 'ground_truth_eeg.mmap', mode='w+',
                   dtype=np.float32, shape=(count, 256, N_EPOCHS))
    y = np.memmap(dest / 'labels.mmap', mode='w+',
                  dtype=np.int16, shape=(count, N_EPOCHS))
    # Match the published CFS Keondo GT *pilot* implementation.
    stats = {}
    for sid in ids:
        a, b = map(int, info[sid]['pos'])
        source = np.asarray(eeg[a:b], dtype=np.float32)
        mu = float(np.mean(source, dtype=np.float64))
        sd = float(np.std(source, dtype=np.float64))
        if not np.isfinite(sd) or sd < 1e-10:
            raise RuntimeError(f'{sid}: zero/invalid EEG variance')
        stats[sid] = mu, sd
    for i, row in enumerate(rows):
        sid = row['subject_id']
        offset, valid = int(row['start_epoch']), int(row['valid_epochs'])
        st, _ = map(int, info[sid]['pos'])
        x = np.zeros((N_EPOCHS, EPOCH_SAMPLES), dtype=np.float32)
        mu, sd = stats[sid]
        x[:valid] = np.clip((np.asarray(eeg[st+offset:st+offset+valid]) - mu)/sd, -10, 10)
        _, power = welch(x, fs=100., window='hann', nperseg=512,
                         noverlap=256, nfft=512, axis=-1, detrend='constant')
        db = 10 * np.log10(np.maximum(power[:, :256], 1e-10))
        spec = np.clip((db + 70) / 80, 0, 1).T.astype(np.float32)
        spec[:, valid:] = 0
        gt[i] = spec
        y[i] = -100
        y[i, :valid] = labels_by_sid[sid][offset:offset+valid]
        if (i+1) % 50 == 0 or i+1 == count:
            print(f'GT {split}: {i+1}/{count}', flush=True)
    gt.flush(); y.flush()
    del gt, y
    with (dest / 'index.jsonl').open('w') as f:
        for row in rows:
            f.write(json.dumps(row) + '\n')
    meta = {'dataset':'dreamt', 'num_samples':count,
            'eeg_shape':[count, 256, N_EPOCHS], 'label_shape':[count, N_EPOCHS],
            'eeg_dtype':'float32', 'label_dtype':'int16',
            'eeg_source':'DREAMT Keondo-preprocessed C4-M1',
            'spectrogram':'CFS Keondo GT pilot Welch fs100 nperseg512 noverlap256',
            'waveform_normalization':'per subject mean/std',
            'db_to_unit_range':[-70, 10],
            'annotation_map':{'0':'Wake','1':'Light','2':'Light','3':'Deep','4':'REM'},
            'padding_ignore_index':-100}
    (dest / 'metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    print('GT saved:', dest, flush=True)


def load_models(seed, device):
    import torch
    from paper_architecture import PaperMaskedRespirationToEEGTransformer
    from src.models.vqgan import VQGAN
    from paper_architecture.evaluate_paper_masked_transformer import load_state_dict
    base = ROOT / f'seed_{seed}'
    tp = base / 'transformer/checkpoint_best_mae.pt'
    vp = base / 'vqgan/checkpoint_best_final.pt'
    transformer_ckpt = torch.load(tp, map_location='cpu', weights_only=False)
    transformer = PaperMaskedRespirationToEEGTransformer(
        **transformer_ckpt['model_config']).to(device)
    transformer.load_state_dict(load_state_dict(transformer_ckpt,
        ['model_state_dict','transformer_state_dict']), strict=True)
    transformer.eval()
    vqgan_ckpt = torch.load(vp, map_location='cpu', weights_only=False)
    vqgan = VQGAN().to(device)
    vqgan.load_state_dict(load_state_dict(vqgan_ckpt,
        ['model_state_dict','vqgan_state_dict','generator_state_dict','state_dict']), strict=True)
    vqgan.eval()
    return transformer, vqgan, tp, vp


def make_synth_split(seed, split, group, channel, batch_size, check_only=False):
    import torch
    from paper_architecture.evaluate_paper_masked_transformer import (
        decode_token_ids, standardize_respiration)
    gt_path = GT_ROOT / split
    metadata = json.loads((gt_path / 'metadata.json').read_text())
    rows = [json.loads(line) for line in (gt_path / 'index.jsonl').read_text().splitlines() if line.strip()]
    count = int(metadata['num_samples'])
    if len(rows) != count:
        raise ValueError('GT index mismatch')
    raw, info = channel_data(split, group, channel)
    grouped = defaultdict(list)
    for i, row in enumerate(rows):
        sid = row['subject_id']
        if sid not in info:
            raise KeyError(f'{sid}: missing {group}/{channel}')
        st, en = map(int, info[sid]['pos'])
        off, valid = int(row['start_epoch']), int(row['valid_epochs'])
        if off < 0 or not 0 < valid <= 512 or off + valid > en-st:
            raise ValueError(f'{sid}: invalid window offset/valid epochs')
        grouped[sid].append((i, off, valid))
    dest = ROOT / f'downstream/dreamt_keondo_pasl_seed{seed}_{channel}/dreamt/{split}'
    print(f'{split}: {count} windows / {len(grouped)} subjects / {group}/{channel}', flush=True)
    if check_only:
        return
    if dest.exists():
        raise FileExistsError(f'Refusing to overwrite {dest}')
    dest.mkdir(parents=True)
    for name in ['labels.mmap','ground_truth_eeg.mmap']:
        (dest / name).symlink_to((gt_path / name).resolve())
    shutil.copyfile(gt_path / 'index.jsonl', dest / 'index.jsonl')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transformer, vqgan, tp, vp = load_models(seed, device)
    generated = np.memmap(dest / 'synthesized_eeg.mmap', mode='w+',
                          dtype=np.float32, shape=(count, 256, N_EPOCHS))
    done = 0
    with torch.inference_mode():
        for si, (sid, windows) in enumerate(grouped.items(), 1):
            st, en = map(int, info[sid]['pos'])
            # Downsample each original contiguous segment independently:
            # never filter across discontinuities in the DREAMT timeline.
            x10_epochs = np.zeros((en-st, 300), dtype=np.float32)
            contiguous, _ = segments(split, sid, en-st)
            for a, b in contiguous:
                segment = np.asarray(raw[st+a:st+b]).reshape(-1)
                values = resample_poly(segment, up=1, down=10).astype(np.float32)
                if values.size != (b-a) * 300:
                    raise RuntimeError(f'{sid}: unexpected resampled size')
                x10_epochs[a:b] = values.reshape(b-a, 300)
            x10 = x10_epochs.reshape(-1)
            for k in range(0, len(windows), batch_size):
                part = windows[k:k+batch_size]
                xb = np.zeros((len(part),64,2400), dtype=np.float32)
                for j, (_, off, valid) in enumerate(part):
                    xb[j].reshape(-1)[:valid*300] = x10[off*300:(off+valid)*300]
                inp = standardize_respiration(torch.from_numpy(xb).to(device))
                with torch.autocast(device_type=device.type, dtype=torch.float16,
                                    enabled=device.type=='cuda'):
                    logits = transformer.predict_from_respiration(inp)
                if tuple(logits.shape[:2]) != (len(part), 512):
                    raise ValueError(f'Bad logits shape {tuple(logits.shape)}')
                ids = logits.argmax(dim=-1).reshape(-1,8,64)
                eeg = decode_token_ids(vqgan, ids)[:,0].float().cpu().numpy()
                if eeg.shape != (len(part),256,512) or not np.isfinite(eeg).all():
                    raise ValueError(f'Bad reconstructed EEG {eeg.shape}')
                for j, (idx, _, _) in enumerate(part):
                    generated[idx] = np.clip(eeg[j], 0., 1.)
                done += len(part)
            if si % 25 == 0 or si == len(grouped):
                generated.flush()
                print(f'SYNTH {split}: {si}/{len(grouped)} subjects, {done}/{count} windows', flush=True)
    if done != count:
        raise RuntimeError(f'Missing synthesis outputs {done}/{count}')
    generated.flush(); del generated
    meta = dict(metadata)
    meta.update(synthesis='PaSL', seed=seed,
                source_channel=f'{group}/{channel}',
                source_respiration_mmap=str(PACKED / split / group / f'{channel}.mmap'),
                transformer_checkpoint=str(tp), vqgan_checkpoint=str(vp),
                spectral_representation='VQGAN-native')
    (dest / 'metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    print('SYNTH complete:', dest, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('check','gt','synth'), required=True)
    p.add_argument('--seed', type=int, choices=(42,0,1))
    p.add_argument('--channel', choices=tuple(CHANNELS), default='ABDOMEN')
    p.add_argument('--split', choices=(*SPLITS, 'all'), default='all')
    p.add_argument('--batch-size', type=int, default=2)
    args = p.parse_args()
    if args.stage=='synth' and args.seed is None:
        p.error('--stage synth requires --seed')
    if args.batch_size < 1:
        p.error('--batch-size must be >= 1')
    requested = SPLITS if args.split == 'all' else (args.split,)
    if args.stage in ('check','gt'):
        ids = read_ids()
        for split in requested:
            make_gt_split(split, ids[split], check_only=args.stage=='check')
    else:
        for split in requested:
            make_synth_split(args.seed, split, *CHANNELS[args.channel],
                             args.batch_size, check_only=False)


if __name__ == '__main__':
    main()
