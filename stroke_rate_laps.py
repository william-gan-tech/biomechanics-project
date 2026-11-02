"""
Phase 8d (10/31): stroke rate and stroke length lap by lap.

For every stretch the researcher confirmed as the named skater on a
straightaway (stroke_rate/stretch_labels.csv: phase straight, or mixed with
use_from / use_to), the locked 8c counter (stroke_counter.setdowns) finds
the set-downs, and:
    stroke rate   = 60 x FPS / median gap between set-downs (strokes/min);
                    needs at least 3 set-downs
    lap           = from the race clock (clock = frame / FPS - CLOCK_OFFSET,
                    offsets read from the clock in the lap-time sheets) and
                    the skater's cumulative split times
    stroke length = lap speed (400 m / lap time) / (stroke rate / 60), m

Rules fixed 10/31 before any stroke rates were computed (see
docs/CAPABILITIES_PHASE8.md): a skater's early (laps 2-5) vs late (laps
9-12) change in median stroke rate counts only if above 10%; a general
claim needs the same direction in at least 4 of 6 skaters.

Output: stroke_rate/stretch_rates.csv, stroke_rate/skater_summary.csv

Usage:
    python -m stroke_rate_laps
"""

import numpy as np
import pandas as pd

from stroke_counter import setdowns

FPS = 30.0
LABELS = "stroke_rate/stretch_labels.csv"
LAPS = "stroke_rate/lap_times.csv"
CHECKS = "stroke_rate/race_checks.csv"
BAR = 0.10
EARLY, LATE = range(2, 6), range(9, 13)
# race clock = frame / FPS - offset; read from the clock beside the first and
# last split graphic of each pair (they agree within 0.3 s)
CLOCK_OFFSET = {"pair01_kramer": 260.9, "pair02_rijhnen": 752.57, "pair05_roest": 2204.85,
                "pair06_bergsma": 3977.15, "pair09_bloemen": 5486.77, "pair10_vanderpoel": 6009.87}
SKATER = {"pair01_kramer": "Sven Kramer", "pair02_rijhnen": "Felix Rijhnen", "pair05_roest": "Patrick Roest",
          "pair06_bergsma": "Jorrit Bergsma", "pair09_bloemen": "Ted-Jan Bloemen",
          "pair10_vanderpoel": "Nils van der Poel"}


def lap_of(clock, cum):
    for lap, c in enumerate(cum, 1):
        if clock < c:
            return lap
    return None


def main():
    lab = pd.read_csv(LABELS, dtype=str).fillna("")
    laps = pd.read_csv(LAPS)
    rows = []
    for pair, g in lab.groupby("pair"):
        use = g[g.phase.isin(["straight", "mixed"]) & (g.skater != "other")]
        if use.empty:
            continue
        race = pd.read_csv(f"stroke_rate/race_{pair}.csv")
        race = race[race.leg_px.notna()]
        lt = laps[laps.skater == SKATER[pair]].sort_values("lap")
        cum = list(lt.cum_s)
        for _, s in use.iterrows():
            a = int(s.use_from or s.start_frame)
            b = int(s.use_to or s.end_frame)
            sd = setdowns(race[(race.frame >= a) & (race.frame <= b)])
            clock = (a + b) / 2 / FPS - CLOCK_OFFSET[pair]
            lap = lap_of(clock, cum)
            row = {"stretch": s.stretch, "skater": SKATER[pair], "from": a, "to": b,
                   "clock_s": round(clock, 1), "lap": lap, "setdowns": len(sd), "rate_spm": np.nan,
                   "lap_s": np.nan, "length_m": np.nan}
            if len(sd) >= 3 and lap is not None:
                rate = 60 * FPS / np.median(np.diff(sd))
                lap_s = float(lt.lap_s.iloc[lap - 1])
                row.update(rate_spm=round(rate, 1), lap_s=lap_s,
                           length_m=round((400 / lap_s) / (rate / 60), 2))
            rows.append(row)
    r = pd.DataFrame(rows)
    # 10/31: stretches hand-checked after a doubtful counter result (stroke_rate/race_checks.csv,
    # check_<stretch>); the hand rate is kept beside the counter's, and the summary is given both ways
    # 11/1: hand marks are held to the same 3-set-down minimum; a checked stretch with fewer has no
    # rate in the hand-checked version
    r["rate_hand"], r["hand_marks"] = np.nan, np.nan
    try:
        chk = pd.read_csv(CHECKS, dtype=str).fillna("")
    except FileNotFoundError:
        chk = pd.DataFrame(columns=["clip", "setdown_frames"])
    for _, c in chk.iterrows():
        marks = [int(x) for x in c.setdown_frames.split()]
        sid = c["clip"].replace("check_", "")
        if not marks or sid not in set(r.stretch):
            continue
        r.loc[r.stretch == sid, "hand_marks"] = len(marks)
        if len(marks) >= 3:
            r.loc[r.stretch == sid, "rate_hand"] = round(60 * FPS / np.median(np.diff(marks)), 1)
    r.to_csv("stroke_rate/stretch_rates.csv", index=False)
    print(r.to_string(index=False))
    out = []
    for version in ("counter", "hand-checked"):
        g0 = r.copy()
        if version == "hand-checked":
            if g0.hand_marks.isna().all():
                continue
            m = g0.hand_marks.notna()
            g0.loc[m, "rate_spm"] = g0.loc[m, "rate_hand"]
            g0.loc[m, "length_m"] = ((400 / g0.loc[m, "lap_s"]) / (g0.loc[m, "rate_spm"] / 60)).round(2)
        for sk, g in g0.dropna(subset=["rate_spm"]).groupby("skater"):
            e, l = g[g.lap.isin(EARLY)], g[g.lap.isin(LATE)]
            row = {"version": version, "skater": sk, "stretches": len(g), "early_n": len(e), "late_n": len(l),
                   "early_spm": e.rate_spm.median(), "late_spm": l.rate_spm.median(),
                   "early_len_m": e.length_m.median(), "late_len_m": l.length_m.median()}
            if len(e) and len(l):
                row["rate_change"] = round((row["late_spm"] - row["early_spm"]) / row["early_spm"], 3)
                row["len_change"] = round((row["late_len_m"] - row["early_len_m"]) / row["early_len_m"], 3)
                row["rate_real"] = abs(row["rate_change"]) > BAR
                row["len_real"] = abs(row["len_change"]) > BAR
            out.append(row)
    s = pd.DataFrame(out)
    s.to_csv("stroke_rate/skater_summary.csv", index=False)
    print(s.to_string(index=False))


if __name__ == "__main__":
    main()
