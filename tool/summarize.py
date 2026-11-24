"""Average the pooled test metrics over the four split seeds (paper protocol, Table 6).

Run tool/test.py for every seed first, then e.g.:
    python tool/summarize.py --mode class --names 1st_resnet50 2st_resnet50 3st_resnet50 4st_resnet50

Each run is scored on checkpoint/<mode>/<name>/prediction/{pred,gt}.txt, pooled over all tasks,
areas and angles of that seed; the table reports the mean over seeds with a t-based 95% CI.
"""
import argparse
import os
import sys

import numpy as np
from scipy.stats import t as t_dist

workspace_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(workspace_path, "tool"))
os.chdir(workspace_path)

from model import paper_metrics  # noqa: E402


def load_rows(pred_dir):
    gt_v, pred_v = [], []
    with open(os.path.join(pred_dir, "gt.txt")) as g, open(os.path.join(pred_dir, "pred.txt")) as p:
        for g_line, p_line in zip(g, p):
            g_s = [x.strip() for x in g_line.split(",", 3)]
            p_s = [x.strip() for x in p_line.split(",", 3)]
            assert g_s[1] == p_s[1] and g_s[3] == p_s[3], "gt.txt / pred.txt rows are not aligned"
            gt_v.append(float(g_s[2]))
            pred_v.append(float(p_s[2]))
    return gt_v, pred_v


def mean_ci(values):
    v = np.asarray(values, dtype=float)
    if len(v) < 2:
        return f"{v.mean():.4f}"
    half = t_dist.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return f"{v.mean():.4f} [{v.mean() - half:.4f}, {v.mean() + half:.4f}]"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="class", choices=["class"])
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--check_root", default="")
    args = parser.parse_args()

    keys = ["mae", "r", "acc0", "acc1", "acc2"]
    per_seed = []
    for name in args.names:
        pred_dir = os.path.join(args.check_root, "checkpoint", args.mode, name, "prediction")
        m = paper_metrics(*load_rows(pred_dir), args.mode)
        per_seed.append(m)
        print(name, f"n={m['n']}", "  ".join(f"{k}={m[k]:.4f}" for k in keys))

    print("MEAN over seeds (95% CI):")
    labels = {"mae": "MAE", "r": "Pearson r", "acc0": "%MAE=0", "acc1": "%MAE<=1", "acc2": "%MAE<=2"}
    for k in keys:
        print(f"  {labels[k]:10s} {mean_ci([m[k] for m in per_seed])}")


if __name__ == "__main__":
    main()
