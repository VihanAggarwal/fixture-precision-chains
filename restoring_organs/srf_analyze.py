"""Tables for the SRF paper from srf_<part>.jsonl (see srf_transfer.py)."""
import json, os, sys
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
X0, PITCH = -0.26 + 0.008 + 0.020, 0.040     # rs_chain: STOP_FACE + CUBE_H, 2 * CUBE_H
TAUS = (0.004, 0.008, 0.012)


def load(part):
    p = os.path.join(os.path.dirname(__file__), f"srf_{part}.jsonl")
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def contraction(rows):
    r = [d / l for row in rows for d, l in zip(row["devs"], row["landed"]) if l > 0.002]
    return (float(np.median(r)), len(r)) if r else (float("nan"), 0)


def summary(rows):
    fin = np.array([max(r["final"]) for r in rows])
    mean = np.mean([np.mean(r["final"]) for r in rows])
    succ = {t: int((fin <= t).sum()) for t in TAUS}
    c, nc = contraction(rows)
    allstage = np.concatenate([r["devs"] for r in rows])
    return dict(n=len(rows), succ=succ, mean_mm=1000 * mean, max_med_mm=1000 * float(np.median(fin)), contr=c, nc=nc,
                p90_stage_mm=1000 * float(np.percentile(allstage, 90)))


def table(rows, keyf, title):
    g = defaultdict(list)
    for r in rows:
        g[keyf(r)].append(r)
    print(f"\n== {title}")
    print(f"{'setting':<34}{'mode':<9}{'n':>3}  succ@4/8/12mm   mean final  median worst  p90 stage  contraction")
    for k in sorted(g, key=str):
        s = summary(g[k])
        print(f"{str(k[0]):<34}{k[1]:<9}{s['n']:>3}  {s['succ'][0.004]}/{s['succ'][0.008]}/{s['succ'][0.012]:<10}"
              f"{s['mean_mm']:8.1f} mm {s['max_med_mm']:9.1f} mm {s['p90_stage_mm']:8.1f} mm   {s['contr']:.3f} ({s['nc']})")
    return g


def onboard(rows_by_part):
    """Plate x at the end of the final push, read from proprioception, as an estimate of the seated part's x.
    Offset calibrated on default-physics restore rows only, then applied unchanged everywhere."""
    def pairs(rows):
        out = []
        for r in rows:
            if r["mode"] != "restore" or not r.get("stall_x"):
                continue
            trav = r.get("travel") or [None] * len(r["stall_x"])
            for k, (sx, d, tr) in enumerate(zip(r["stall_x"], r["devs"], trav)):
                if sx is None:
                    continue
                off_k = sx - (X0 + k * PITCH)
                # abstain when the push ended before contact (part already at its slot): the plate then sits ~25 mm
                # short and says nothing about the part. Older rows have no travel log, so use the offset itself.
                if (tr is not None and tr < 0.015) or (tr is None and off_k > 0.045):
                    continue
                out.append((off_k, d))
        return out
    cal = pairs([r for r in rows_by_part.get("contact", []) if r["cfg"] == "default"] +
                [r for r in rows_by_part.get("gap", []) if r["bias"] == 0.0])
    if not cal:
        return
    off = float(np.median([a for a, _ in cal]))
    print(f"\n== on-board seat check (plate offset {1000*off:.1f} mm, calibrated on default physics)")
    for name, rows in rows_by_part.items():
        g = defaultdict(list)
        for r in rows:
            key = r.get("cfg") or (f"sd={r.get('sd')} b={r.get('bias')}" if name == "gap" else f"psd={r.get('psd')}")
            g[key].append(r)
        for key in sorted(g):
            pr = pairs(g[key])
            if not pr:
                continue
            est = np.array([abs(a - off) for a, _ in pr]); true = np.array([d for _, d in pr])
            err = np.abs(est - true)
            bad = true > 0.008
            flag = est > 0.008
            print(f"{name:<8}{key:<26} stages {len(pr):3d}  |est-true| median {1000*np.median(err):4.1f} mm  p90 "
                  f"{1000*np.percentile(err, 90):5.1f} mm  bad {bad.sum():2d} caught {int((bad & flag).sum()):2d} "
                  f"false alarms {int((~bad & flag).sum()):2d}")


if __name__ == "__main__":
    parts = {p: load(p) for p in ("gap", "contact", "percep")}
    if parts["gap"]:
        table(parts["gap"], lambda r: (f"sd={1000*r['sd']:.0f}mm bias={1000*r['bias']:.0f}mm", r["mode"]), "policy-side gap (line-up, N=8)")
    if parts["contact"]:
        table(parts["contact"], lambda r: (r["cfg"], r["mode"]), "contact-model shift (line-up, N=4, sd 4 mm)")
    if parts["percep"]:
        table(parts["percep"], lambda r: (f"percep sd={1000*r['psd']:.0f}mm", r["mode"]), "perception noise in restore (N=4, sd 8 mm)")
    onboard({k: v for k, v in parts.items() if v})
