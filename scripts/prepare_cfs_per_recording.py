"""Prepare CFS test with per-recording EEG z-score (sensitivity analysis)."""
import argparse, json, pickle, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import preprocess_multidataset_vqgan as prep
from scripts import build_transformer_dataset as build

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output-root', type=Path, default=Path('outputs/cfs_external_per_recording'))
    p.add_argument('--vqgan-checkpoint', type=Path, default=Path('outputs/vqgan_multidataset/checkpoint_best.pt'))
    p.add_argument('--normalization', type=Path, default=Path('outputs/vqgan_multidataset_preprocessed/normalization.json'))
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--max-windows', type=int)
    a = p.parse_args()
    if a.output_root.exists(): raise FileExistsError(a.output_root)
    root = Path('/hdd2/kdpark/sleep_datasets/cfs/mmap_signals_MAE_processed')
    eeg_path, abd_path = root/'test/EEG/C4-M1.pickle', root/'test/ABD/ABD.pickle'
    abd_train_path = root/'train/ABD/ABD.pickle'
    def load(path):
        with path.open('rb') as f: return pickle.load(f)
    eeg_meta, abd_meta, abd_train = load(eeg_path), load(abd_path), load(abd_train_path)
    eeg_shape, abd_shape = tuple(eeg_meta['data_shape']), tuple(abd_meta['data_shape'])
    eeg = np.memmap(eeg_path.with_suffix('.mmap'), dtype='float32', mode='r', shape=eeg_shape)
    eeg_ids, abd_ids = eeg_meta['sig_info'], abd_meta['sig_info']
    rows, used = [], set()
    for subject, info in eeg_ids.items():
        if subject not in abd_ids: continue
        es, ee = info['pos']; rs, re = abd_ids[subject]['pos']
        nwin = min((ee-es)//512, (re-rs)//512)
        if nwin <= 0: continue
        n = (ee-es) * eeg_shape[1]
        mean = float(info['stats']['sum']) / n
        var = float(info['stats']['sumsq']) / n - mean*mean
        std = float(np.sqrt(max(var, 1e-12)))
        for wi in range(nwin):
            if a.max_windows is not None and len(rows) >= a.max_windows: break
            x = np.asarray(eeg[es+wi*512:es+(wi+1)*512], dtype=np.float32)
            spec, freqs = prep.make_spectrogram(x, mean, std)
            outdir = a.output_root/'preprocessed/test/cfs_test'; outdir.mkdir(parents=True, exist_ok=True)
            path = outdir/f'cfs_test__{subject}__window{wi:03d}.npz'
            np.savez(path, eeg_spectrogram_db=spec, freqs=freqs, dataset='cfs_test', split='test', subject_id=str(subject), window_index=wi, eeg_mean=mean, eeg_std=std, normalization_source='per_recording')
            rows.append({'path': str(path), 'dataset':'cfs_test', 'subject_id':str(subject), 'window_index':wi, 'normalization_source':'per_recording'})
            used.add(subject)
        if a.max_windows is not None and len(rows) >= a.max_windows: break
    if not rows: raise RuntimeError('No aligned CFS windows found.')
    norm = json.loads(a.normalization.read_text())
    (a.output_root/'preprocessed').mkdir(parents=True, exist_ok=True)
    (a.output_root/'preprocessed/test_manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (a.output_root/'preprocessed/normalization.json').write_text(json.dumps(norm, indent=2))
    # Build transformer respiration mmap/tokens with frozen VQGAN and CFS-train ABD stats.
    build.ABD_PATHS['cfs_train'] = abd_train_path
    build.ABD_PATHS['cfs_test'] = abd_path
    build.RESP_STATS_FROM['cfs_test'] = 'cfs_train'
    stats = build.compute_global_stats(abd_train)
    tr = a.output_root/'transformer_data'; tr.mkdir(parents=True, exist_ok=True)
    (tr/'respiration_stats.json').write_text(json.dumps({'cfs_train':stats,'cfs_test':stats}, indent=2))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build.load_checkpoint(a.vqgan_checkpoint, device)
    build.process_split('test', rows, a.output_root/'preprocessed', tr, model, device, {'cfs_train':stats,'cfs_test':stats}, float(norm['min_db']), float(norm['max_db']), a.batch_size)
    print(f'Prepared {len(rows)} windows from {len(used)} CFS recordings.')
    print(f'Transformer data: {tr}')

if __name__ == '__main__': main()
