"""Three-model V2 summary: per-model stats + cross-model comparison."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

# protocol inference: omni -> VA, others -> VT
def infer_protocol(model_key):
    if 'omni' in model_key or 'audio' in model_key:
        return 'VA'
    return 'VT'

def load_summary(model_key, root='outputs/v2'):
    proto = infer_protocol(model_key)
    p = Path(root) / 'state_data' / model_key / proto
    with (p / 'v2_summary.json').open('r', encoding='utf-8') as f:
        summary = json.load(f)
    with (p / 'thresholds.json').open('r', encoding='utf-8') as f:
        th = json.load(f)
    with (p / 'state_patterns.jsonl').open('r', encoding='utf-8') as f:
        rows = [json.loads(l) for l in f]
    return proto, summary, th, rows

def stats_for_split(rows, split, sample_type=None, kappa=None):
    sub = [r for r in rows if r.get('representation_split') == split]
    if sample_type is not None:
        sub = [r for r in sub if r.get('sample_type') == sample_type]
    if not sub:
        return None
    D = np.array([r['D'] for r in sub], dtype=np.float64)
    S = np.array([r['S_mean'] for r in sub], dtype=np.float64)
    R = np.array([r['R'] for r in sub], dtype=np.float64)
    out = {
        'n': len(sub),
        'D_mean': float(D.mean()), 'D_median': float(np.median(D)),
        'S_mean': float(S.mean()),
        'R_mean': float(R.mean()), 'R_abs_mean': float(np.abs(R).mean()),
    }
    if kappa is not None:
        stable = sub if sample_type == 'Aligned' else [r for r in sub if r['S_mean'] <= kappa]
        out['n_stable'] = len(stable)
    return out

def four_pattern_pct(rows, split='official_test'):
    sub = [r for r in rows if r.get('representation_split') == split]
    if not sub:
        return {}
    n = len(sub)
    counts = {}
    for r in sub:
        p = r.get('pattern', 'Unknown')
        counts[p] = counts.get(p, 0) + 1
    return {p: round(100 * c / n, 2) for p, c in counts.items()}, counts, n

def v_lean_pct(rows, th, split='official_test'):
    """V-lean %: stable Conflict with R > +delta (in VT, M1=Visual). For VA, M1=Visual too."""
    kappa = th['kappa']
    test = [r for r in rows if r.get('representation_split') == split]
    stable_conf = [r for r in test if r.get('sample_type') == 'Conflict' and r['S_mean'] <= kappa]
    if not stable_conf:
        return None
    delta = float(np.median([r['delta_i'] for r in rows]))
    v_lean = [r for r in stable_conf if r['R'] > delta]
    ta_lean = [r for r in stable_conf if r['R'] < -delta]
    balanced = [r for r in stable_conf if -delta <= r['R'] <= delta]
    return {
        'n_stable_conflict': len(stable_conf),
        'delta_median': delta,
        'V_lean_pct': round(100 * len(v_lean) / len(stable_conf), 2),
        'T/A_lean_pct': round(100 * len(ta_lean) / len(stable_conf), 2),
        'balanced_pct': round(100 * len(balanced) / len(stable_conf), 2),
    }

def report(model_key):
    proto, summary, th, rows = load_summary(model_key)
    print('=' * 72)
    print(f'MODEL: {model_key}  /  PROTOCOL: {proto}')
    print('=' * 72)
    print(f"thresholds: kappa={th['kappa']:.4f}, tau={th['tau']:.4f}, n_calib={th['n_calibration_rows']}")
    metrics = summary.get('training_metrics', {})
    print(f"train: best_epoch={metrics.get('best_epoch')}, val_bal_acc={metrics.get('best_val_balanced_accuracy_ac', 0):.4f}, stop={metrics.get('stop_reason')}")
    print(f"pattern counts (all rows): {summary.get('pattern_counts', {})}")
    print(f"sample_type counts (all rows): {summary.get('sample_type_counts', {})}")
    print()
    print('--- per split x sample_type D (=d(M1,M2)) stats ---')
    for split in ['official_test', 'aligned_calibration']:
        for st in ['Conflict', 'Aligned']:
            s = stats_for_split(rows, split, st)
            if s:
                print(f'  {split:25s} {st:9s}: n={s["n"]:5d} D_mean={s["D_mean"]:.3f} D_med={s["D_median"]:.3f}')
    print()
    fpct, fcnt, n_total = four_pattern_pct(rows, 'official_test')
    print(f'--- four-pattern distribution (official_test, n={n_total}) ---')
    for p, c in sorted(fcnt.items(), key=lambda x: -x[1]):
        print(f'  {p:12s}: {c:5d} ({fpct[p]:.2f}%)')
    print()
    vl = v_lean_pct(rows, th, 'official_test')
    if vl:
        print(f"--- V-lean (M1-dominant) on stable Conflict: {vl['V_lean_pct']:.2f}% (delta={vl['delta_median']:.4f}) ---")
        print(f"    V-lean={vl['V_lean_pct']:.1f}%, T/A-lean={vl['T/A_lean_pct']:.1f}%, balanced={vl['balanced_pct']:.1f}% (n_stable_conflict={vl['n_stable_conflict']})")
    print()
    return {
        'model_key': model_key,
        'protocol': proto,
        'kappa': th['kappa'], 'tau': th['tau'],
        'best_epoch': metrics.get('best_epoch'),
        'val_bal_acc': metrics.get('best_val_balanced_accuracy_ac', 0),
        'stop_reason': metrics.get('stop_reason'),
        'four_pattern_pct_official_test': fpct,
        'four_pattern_count_official_test': fcnt,
        'v_lean_stats': vl,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+',
                        default=['qwen3_vl_8b', 'internvl3_5_8b', 'qwen2_5_omni_7b'])
    parser.add_argument('--output', default='outputs/v2/three_model_summary.json')
    args = parser.parse_args()
    all_stats = []
    for mk in args.models:
        try:
            s = report(mk)
            all_stats.append(s)
        except FileNotFoundError as e:
            print(f'!! {mk}: {e}', file=sys.stderr)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(all_stats, f, indent=2)
    print(f'\nSaved {args.output}')
    # Cross-model comparison table
    if len(all_stats) >= 2:
        print()
        print('=' * 72)
        print('CROSS-MODEL COMPARISON')
        print('=' * 72)
        print(f'{"model":24s} {"proto":5s} {"kappa":>8s} {"tau":>8s} {"val_acc":>8s} {"V-lean%":>9s} {"Cons%":>7s} {"Dom%":>7s} {"Balc%":>7s} {"Conf%":>7s}')
        for s in all_stats:
            fpct = s['four_pattern_pct_official_test']
            vl = s['v_lean_stats'] or {}
            def _g(p):
                return f'{fpct.get(p, 0):.1f}'
            print(f'{s["model_key"]:24s} {s["protocol"]:5s} {s["kappa"]:>8.4f} {s["tau"]:>8.3f} {s["val_bal_acc"]:>8.4f} {vl.get("V_lean_pct", 0):>9.1f} {_g("Consensus"):>7s} {_g("Dominant"):>7s} {_g("Balanced"):>7s} {_g("Confusion"):>7s}')

if __name__ == '__main__':
    main()
