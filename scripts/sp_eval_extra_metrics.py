"""Extra metrics for SP v2 runs.

For each SP v2 run dir, loads encoder.pt, runs the AC test split forward,
and computes acc / macro_f1 / average_precision / roc_auc on the test set.
Writes pretrain_extra_metrics.json next to pretrain_metrics.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)

HERE = Path(__file__).resolve().parent
PROJ_ROOT = HERE.parent.parent
V2_SRC = PROJ_ROOT / 'src'
if str(V2_SRC) not in sys.path:
    sys.path.insert(0, str(V2_SRC))

from mprisk.cache.cache_manifest import _can_materialize_entry, _entry_from_row  # noqa: E402
from mprisk.cache.prefill_extract import extract_t0_trajectory  # noqa: E402

CONDITIONS = ('M1', 'M2', 'M12')
COND_IDX = {'M1': 0, 'M2': 1, 'M12': 2}


def load_prompt_ids(prompt_set_path):
    import yaml
    with open(prompt_set_path) as f:
        ps = yaml.safe_load(f)
    return [t['prompt_id'] for t in ps['templates'] if t.get('enabled', True)]


def scan_cache(cache_roots, *, model_key, prompt_ids):
    out = {}
    expected = set(prompt_ids)
    for root in cache_roots:
        if not root.exists():
            continue
        mpath = root / 'manifest.jsonl'
        if not mpath.exists():
            continue
        with open(mpath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get('model_key') != model_key:
                    continue
                cond = row.get('condition')
                if cond not in CONDITIONS:
                    continue
                pid = row.get('prompt_id') or (row.get('metadata') or {}).get('prompt_id')
                if pid is None or pid not in expected:
                    continue
                if not _can_materialize_entry(row):
                    continue
                entry = _entry_from_row(row, cache_root=root)
                slot = out.setdefault(entry.sample_id, {c: {} for c in CONDITIONS})
                slot[cond].setdefault(pid, entry)
    return out


def load_split_assignment(path):
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            split = row.get('representation_split', '')
            for sid in row.get('sample_ids', []):
                out[sid] = split
    return out


def _domain_of(sid):
    return 'gen' if sid.startswith('gen:') else 'natural'


def _filter_domain(sids, domain):
    if domain == 'both':
        return list(sids)
    keep = 'gen' if domain == 'gen_only' else 'natural'
    return [s for s in sids if _domain_of(s) == keep]


def load_sample_types(main_manifest, protocol):
    from mprisk.cache.hidden_state_cache import normalize_protocol
    from mprisk.data.manifests import read_final_manifest
    rows = read_final_manifest(main_manifest)
    return {
        r.sample_id: r.sample_type
        for r in rows
        if normalize_protocol(r.protocol) == normalize_protocol(protocol)
    }


def build_test_tensors(test_pool, sample_types, cache_index, prompt_ids, *, condition):
    chosen_prompt = prompt_ids[0]
    ci = COND_IDX[condition]
    tensors, labels, kept = [], [], []
    for sid in test_pool:
        stype = sample_types.get(sid)
        if stype == 'Conflict':
            label = 1
        elif stype == 'Aligned':
            label = 0
        else:
            continue
        info = cache_index.get(sid)
        if info is None:
            continue
        bundle = []
        ok = True
        for cond in CONDITIONS:
            entry = info[cond].get(chosen_prompt)
            if entry is None:
                ok = False
                break
            try:
                traj = extract_t0_trajectory(entry)
            except Exception:
                ok = False
                break
            bundle.append(traj)
        if not ok:
            continue
        arr = np.stack(bundle, axis=0).astype(np.float32)
        tensors.append(arr[ci])
        labels.append(label)
        kept.append(sid)
    if not tensors:
        return None, None, None
    return (
        torch.from_numpy(np.stack(tensors, axis=0)),
        torch.tensor(labels, dtype=torch.long),
        kept,
    )


def build_test_tensors_with_ratio(test_x, test_y, train_y, *, conflict_ratio, seed):
    """Reproduce train_baseline's conflict-ratio drop on Conflict samples.

    NOTE: conflict_ratio only applies to TRAIN, not test. Test is unchanged.
    But test set order may shift after dropping train samples -- for eval,
    we just use the full test set, since conflict_ratio only affects train.
    """
    return test_x, test_y


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dirs', nargs='+', required=True,
                   help='one or more SP v2 run dirs containing encoder.pt')
    p.add_argument('--cache-roots', nargs='+', required=True)
    p.add_argument('--prompt-set', required=True)
    p.add_argument('--main-manifest', required=True)
    p.add_argument('--split-assignment', required=True)
    p.add_argument('--domain', default='gen_only')
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()

    prompt_ids = load_prompt_ids(args.prompt_set)
    split_of = load_split_assignment(args.split_assignment)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    for run_dir in args.run_dirs:
        run_dir = Path(run_dir)
        meta_path = run_dir / 'pretrain_metrics.json'
        enc_path = run_dir / 'encoder.pt'
        if not meta_path.exists() or not enc_path.exists():
            print(f'[skip] {run_dir}: missing meta/encoder')
            continue
        meta = json.load(open(meta_path))
        model_key = meta['model_key']

        from mprisk_viz.baselines import load_encoder
        encoder, ckpt_meta = load_encoder(enc_path, map_location='cpu')
        encoder.to(device).eval()
        for pp in encoder.parameters():
            pp.requires_grad_(False)

        proto = 'va' if model_key in ('qwen2_5_omni_7b', 'gemma4_12b_it') else 'vt'
        sample_types = load_sample_types(Path(args.main_manifest), protocol=proto)
        cache_index = scan_cache(
            [Path(c) for c in args.cache_roots],
            model_key=model_key, prompt_ids=prompt_ids,
        )

        test_pool_all = [s for s, sp in split_of.items() if sp == 'official_test']
        test_pool = _filter_domain(test_pool_all, args.domain)
        test_x, test_y, _ = build_test_tensors(
            test_pool, sample_types, cache_index, prompt_ids, condition='M12',
        )
        if test_x is None:
            print(f'[skip] {run_dir}: empty test')
            continue
        with torch.no_grad():
            logits = encoder(test_x.to(device)).cpu()
            probs = F.softmax(logits, dim=-1)[:, 1].numpy()
            preds = logits.argmax(dim=-1).numpy()
        labels = test_y.numpy()
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, zero_division=0)
        try:
            ap = average_precision_score(labels, probs)
        except Exception:
            ap = 0.0
        try:
            auc = roc_auc_score(labels, probs)
        except Exception:
            auc = 0.5
        extra = {
            'n_test': int(len(labels)),
            'accuracy': float(acc),
            'macro_f1': float(f1),
            'average_precision': float(ap),
            'roc_auc': float(auc),
            'positive_rate': float(labels.mean()),
        }
        out_path = run_dir / 'pretrain_extra_metrics.json'
        out_path.write_text(json.dumps(extra, indent=2, sort_keys=True))
        print(f'[{run_dir.name}] n={extra["n_test"]} acc={acc:.4f} f1={f1:.4f} ap={ap:.4f} auc={auc:.4f}')


if __name__ == '__main__':
    raise SystemExit(main())
