"""
Phase 9 (11/25): a page for reviewing labelled Teachable Machine pictures
before uploading them, at the researcher's request.

Writes one HTML page (opens in any browser, works offline) with every picture
from a label list, grouped into corner and straight, numbered, with the skater,
frame and what the marks say. A picture where the researcher's label differs
from the marks is outlined in orange. To change one, note its number; the
label list can then be edited and the folders rebuilt with --apply.

11/26: the page first lists every picture whose label differs from the marks,
then the corner and straight pictures that agree with them.

Usage:
    python -m review_tm --upper                         # phase9/review_upper.html
    python -m review_tm                                 # phase9/review_full.html
    python -m review_tm --upper --apply "U12=straight U40=skip"
    python -m review_tm --identity                      # phase9/review_identity.html (11/27)
    python -m review_tm --identity --apply "I12=not_racer"
"""

import os
import html
import shutil
import argparse
import pandas as pd

import label_tm as lt


def sets(upper):
    if upper == "identity":
        return lt.IDENTITY["pool"], lt.IDENTITY["out"], lt.IDENTITY["log"], "I", "phase9/review_identity.html"
    if upper:
        return lt.UPPER["pool"], lt.UPPER["out"], lt.UPPER["log"], "U", "phase9/review_upper.html"
    return lt.POOL, lt.OUT, lt.LOG, "F", "phase9/review_full.html"


def numbered(upper):
    pool, out, log, prefix, page = sets(upper)
    d = pd.read_csv(log)
    p = pd.read_csv(os.path.join(pool, "pool.csv"))
    if upper is False and os.path.exists(os.path.join(lt.MORE["pool"], "pool.csv")):
        p = pd.concat([p, pd.read_csv(os.path.join(lt.MORE["pool"], "pool.csv"))])   # 11/27: second full-body pool
    p = p.drop_duplicates("image").set_index("image")
    d = d[d.label != "skip"].join(p, on="image").reset_index(drop=True)
    d["no"] = [f"{prefix}{i + 1}" for i in range(len(d))]
    return d


def page(upper):
    pool, out, log, prefix, path = sets(upper)
    d = numbered(upper)
    if upper == "identity":
        return identity_page(d, out, path)
    title = "Upper-body pictures" if upper else "Full-body pictures"
    parts = [f"<!doctype html><meta charset='utf-8'><title>{title} review</title>",
             "<style>body{font-family:system-ui,sans-serif;margin:16px;background:#fafaf8;color:#111}"
             "h2{margin-top:28px}.grid{display:flex;flex-wrap:wrap;gap:10px}"
             ".card{width:190px;background:#fff;border:2px solid #ddd;border-radius:6px;padding:4px;font-size:12px}"
             ".card.diff{border-color:#e07b00}.card img{width:100%;height:190px;object-fit:contain;background:#eee}"
             "b{font-size:14px}</style>",
             f"<h1>{title}: review before Teachable Machine</h1>",
             f"<p>{len(d)} pictures ({(d.label == 'corner').sum()} corner, {(d.label == 'straight').sum()} straight). "
             "Each has a number; tell Claude the numbers to change, for example <i>U12=straight, U40=skip</i>. "
             "<b style='color:#e07b00'>Orange outline</b>: your label differs from what the marks say.</p>"]
    sections = [("Your label differs from the marks", d[d.label != d.suggested])] + \
               [(f"Labelled {lab}, same as the marks", d[(d.label == lab) & (d.suggested == lab)]) for lab in ("corner", "straight")]
    for title_, g in sections:
        parts.append(f"<h2>{title_} ({len(g)})</h2><div class='grid'>")
        for _, r in g.iterrows():
            lab = r.label
            src = os.path.relpath(os.path.join(out, lab, r.image), os.path.dirname(path)).replace("\\", "/")
            diff = " diff" if r.suggested != lab else ""
            parts.append(f"<div class='card{diff}'><img src='{html.escape(src)}' loading='lazy'>"
                         f"<b>{r.no}</b> {html.escape(r.skater.split()[-1])}, frame {r.frame}<br>"
                         f"you: <b>{lab}</b> &middot; marks: {r.suggested}</div>")
        parts.append("</div>")
    open(path, "w", encoding="utf-8").write("\n".join(parts))
    print(path, len(d), "pictures")


def identity_page(d, out, path):
    parts = ["<!doctype html><meta charset='utf-8'><title>Skater identity review</title>",
             "<style>body{font-family:system-ui,sans-serif;margin:16px;background:#fafaf8;color:#111}"
             "h2{margin-top:28px}.grid{display:flex;flex-wrap:wrap;gap:10px}"
             ".card{width:170px;background:#fff;border:2px solid #ddd;border-radius:6px;padding:4px;font-size:12px}"
             ".card img{width:100%;height:170px;object-fit:contain;background:#eee}b{font-size:14px}</style>",
             "<h1>Skater identity: review before Teachable Machine</h1>",
             f"<p>{len(d)} pictures ({(d.label == 'racer').sum()} racer, {(d.label == 'not_racer').sum()} not a racer). "
             "Racer: a skater racing in this race. Not a racer: officials, coaches, staff, spectators, skaters warming up. "
             "Tell Claude the numbers to change, for example <i>I12=not_racer, I40=skip</i>.</p>"]
    for lab, title_ in (("not_racer", "Labelled not a racer"), ("racer", "Labelled racer")):
        g = d[d.label == lab]
        parts.append(f"<h2>{title_} ({len(g)})</h2><div class='grid'>")
        for _, r in g.iterrows():
            src = os.path.relpath(os.path.join(out, lab, r.image), os.path.dirname(path)).replace("\\", "/")
            parts.append(f"<div class='card'><img src='{html.escape(src)}' loading='lazy'>"
                         f"<b>{r.no}</b> {html.escape(r.skater)}, frame {r.frame}</div>")
        parts.append("</div>")
    open(path, "w", encoding="utf-8").write("\n".join(parts))
    print(path, len(d), "pictures")


def apply(upper, changes):
    pool, out, log, prefix, _ = sets(upper)
    d = numbered(upper).set_index("no")
    full = pd.read_csv(log)
    for ch in changes.split():
        no, new = ch.split("=")
        img, old = d.loc[no, "image"], d.loc[no, "label"]
        full.loc[full.image == img, "label"] = new
        os.remove(os.path.join(out, old, img))
        if new != "skip":
            os.makedirs(os.path.join(out, new), exist_ok=True)
            src = os.path.join(pool, img)
            if not os.path.exists(src):
                src = os.path.join(lt.MORE["pool"], img)
            shutil.copy(src, os.path.join(out, new, img))
        print(no, img, old, "->", new)
    full.to_csv(log, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upper", action="store_true")
    ap.add_argument("--identity", action="store_true")
    ap.add_argument("--apply")
    args = ap.parse_args()
    kind = "identity" if args.identity else args.upper
    if args.apply:
        apply(kind, args.apply)
    page(kind)


if __name__ == "__main__":
    main()
