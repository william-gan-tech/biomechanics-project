"""
Phase 9 (11/27): test the three Teachable Machine identity models (racer vs
not a racer), each on the group of races it was not trained on (folders from
tm_folds.py). The test pictures are the researcher's labels for that group,
not hand-picked further.

Usage:
    python -m identity_check
"""

import os
import cv2
import numpy as np
import pandas as pd
from ai_edge_litert.interpreter import Interpreter

import label_tm as lt
from tm_folds import group_of


def predict(model_dir, paths):
    it = Interpreter(model_path=os.path.join(model_dir, "model_unquant.tflite"))
    it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    names = [l.split(" ", 1)[1].strip().lower() for l in open(os.path.join(model_dir, "labels.txt"))]
    to_label = ["not_racer" if n.startswith("not") else "racer" for n in names]
    res = []
    for p in paths:
        img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]; s = min(h, w)
        img = img[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
        x = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32) / 127.5 - 1
        it.set_tensor(inp["index"], x[None]); it.invoke()
        res.append(to_label[int(it.get_tensor(out["index"])[0].argmax())])
    return np.array(res)


def main():
    d = pd.read_csv(lt.IDENTITY["log"])
    d = d[d.label != "skip"].copy()
    d["group"] = d.image.map(group_of)
    rows = []
    for g in ("beijing", "calgary", "inzell"):
        t = d[d.group == g]
        pred = predict(f"phase9/tm_model_id_without_{g}", [os.path.join(lt.IDENTITY["out"], r.label, r.image) for r in t.itertuples()])
        y = t.label.values
        r_ok, n_ok = (pred[y == "racer"] == "racer").mean(), (pred[y == "not_racer"] == "not_racer").mean()
        rows.append({"tested on": g, "racers kept": f"{r_ok:.1%} of {(y == 'racer').sum()}",
                     "non-racers caught": f"{n_ok:.1%} of {(y == 'not_racer').sum()}",
                     "average": round((r_ok + n_ok) / 2, 3),
                     "non-racers let through": int((pred[y == "not_racer"] == "racer").sum()),
                     "racers wrongly dropped": int((pred[y == "racer"] == "not_racer").sum())})
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
