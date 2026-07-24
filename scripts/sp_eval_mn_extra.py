"""MN head extra metrics for SP v2 runs."""
from __future__ import annotations

import argparse
import json
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


class _MLPHead(torch.nn.Module):
    def __init__(self, in_dim, h1=128, h2=2, dropout=0.1):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 128), torch.nn.GELU(),
            torch.nn.Dropout(0.1), torch.nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.net(x)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dirs', nargs='+', required=True)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    for run_dir in args.run_dirs:
        run_dir = Path(run_dir)
        emb = run_dir / 'embeddings.npz'
        head_pt = run_dir / 'mn_head.pt'
        if not emb.exists() or not head_pt.exists():
            print(f'[skip] {run_dir.name}: missing npz/head')
            continue
        d = np.load(emb, allow_pickle=True)
        reps = torch.from_numpy(d['embeddings']).to(device)
        labels = d['labels'].astype(np.int64)
        test_mask = d['test_mask'].astype(bool)

        head = _MLPHead(int(reps.shape[1]))
        head.net.load_state_dict(torch.load(head_pt, map_location='cpu'))
        head.to(device).eval()
        with torch.no_grad():
            logits = head(reps).cpu()
            probs = F.softmax(logits, dim=-1)[:, 1].numpy()
            preds = logits.argmax(dim=-1).numpy()
        te_labels = labels[test_mask]
        te_preds = preds[test_mask]
        te_probs = probs[test_mask]
        acc = accuracy_score(te_labels, te_preds)
        f1 = f1_score(te_labels, te_preds, zero_division=0)
        try:
            ap = average_precision_score(te_labels, te_probs)
        except Exception:
            ap = 0.0
        try:
            auc = roc_auc_score(te_labels, te_probs)
        except Exception:
            auc = 0.5
        extra = {
            'n_test': int(te_labels.shape[0]),
            'accuracy': float(acc),
            'macro_f1': float(f1),
            'average_precision': float(ap),
            'roc_auc': float(auc),
            'positive_rate': float(te_labels.mean()),
        }
        out = run_dir / 'mn_extra_metrics.json'
        out.write_text(json.dumps(extra, indent=2, sort_keys=True))
        print(f'[{run_dir.name}] MN n={extra["n_test"]} acc={acc:.4f} f1={f1:.4f} ap={ap:.4f} auc={auc:.4f} pr={extra["positive_rate"]:.3f}')


if __name__ == '__main__':
    raise SystemExit(main())
