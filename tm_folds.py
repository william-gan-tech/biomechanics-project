"""
Phase 9 (11/27): folder sets for training Teachable Machine models with one
group of races left out, so each model can be tested on a broadcast it never
saw (Beijing, Calgary, Inzell; decided 11/26).

For each group G, phase9/tm_folds/<kind>/without_<G>/<label>/ holds the
labelled pictures from the other two groups; the pictures from G are the test.

Usage:
    python -m tm_folds --identity
"""

import os
import shutil
import argparse
import pandas as pd

import label_tm as lt

GROUPS = {"pair10_vanderpoel": "beijing", "pair09_bloemen": "beijing", "pair06_bergsma": "beijing",
          "calgary2025_ghiotto": "calgary", "inzell2026_eitrem": "inzell"}


def group_of(image):
    return next(g for p, g in GROUPS.items() if image.startswith(p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--identity", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--all", action="store_true", help="also one set with every group, for the final model")
    a = ap.parse_args()
    cfg = lt.IDENTITY if a.identity else ({"log": lt.LOG, "out": lt.OUT} if a.full else lt.UPPER)
    kind = "identity" if a.identity else ("full" if a.full else "upper")
    d = pd.read_csv(cfg["log"])
    d = d[d.label != "skip"].copy()
    d["group"] = d.image.map(group_of)
    root = os.path.join("phase9/tm_folds", kind)
    shutil.rmtree(root, ignore_errors=True)
    for left_out in ("beijing", "calgary", "inzell") + (("none",) if a.all else ()):
        tr = d[d.group != left_out]
        for _, r in tr.iterrows():
            dst = os.path.join(root, f"without_{left_out}" if left_out != "none" else "all_groups", r.label)
            os.makedirs(dst, exist_ok=True)
            shutil.copy(os.path.join(cfg["out"], r.label, r.image), os.path.join(dst, r.image))
        print(f"without_{left_out}:" if left_out != "none" else "all_groups:", tr.label.value_counts().to_dict(), "| test:", d[d.group == left_out].label.value_counts().to_dict())


if __name__ == "__main__":
    main()
