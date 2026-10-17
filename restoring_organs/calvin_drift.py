"""CALVIN drift check: do failed subtasks in pretrained-policy chains follow displaced objects?

Wraps HULC's evaluation loop without editing it: `rollout` is replaced (module-global lookup) by a version that
records scene_obs at the start and end of every subtask, and `evaluate_sequence` is wrapped to mark chain
boundaries. Then `analyse` asks, for subtasks 2..5 of each chain: is failure more likely when a NON-target block
has moved more than DRIFT_M from where the chain started?

scene_obs (24 dims, calvin_env play_table_scene): slider, drawer, button, switch, lightbulb, green light,
then red, blue, pink block (xyz + euler each).
"""
import argparse
import importlib
import json
import os
import sys

import numpy as np

BLOCKS = {"red": slice(6, 9), "blue": slice(12, 15), "pink": slice(18, 21)}
DRIFT_M = 0.02


def install_logger(log_path):
    mod = importlib.import_module("hulc.evaluation.evaluate_policy")
    assert hasattr(mod, "rollout") and hasattr(mod, "evaluate_sequence"), "HULC evaluate_policy layout changed"
    orig_rollout, orig_seq = mod.rollout, mod.evaluate_sequence
    state = {"chain": -1}
    f = open(log_path, "a")

    def evaluate_sequence(*args, **kw):
        state["chain"] += 1
        return orig_seq(*args, **kw)

    def rollout(env, model, task_oracle, subtask, *args, **kw):
        s0 = np.asarray(env.get_obs()["scene_obs"], float).copy()
        ok = orig_rollout(env, model, task_oracle, subtask, *args, **kw)
        s1 = np.asarray(env.get_obs()["scene_obs"], float).copy()
        f.write(json.dumps(dict(chain=state["chain"], subtask=subtask, success=bool(ok),
                                scene_start=s0.tolist(), scene_end=s1.tolist())) + "\n")
        f.flush()
        return ok

    mod.rollout, mod.evaluate_sequence = rollout, evaluate_sequence
    return mod


def analyse(log_path):
    rows = [json.loads(l) for l in open(log_path) if l.strip()]
    chains = {}
    for r in rows:
        chains.setdefault(r["chain"], []).append(r)
    fail_d, fail_n, tot_d, tot_n = 0, 0, 0, 0
    lengths = []
    for ch in chains.values():
        lengths.append(sum(1 for _ in __import__("itertools").takewhile(lambda r: r["success"], ch)))
        start = np.asarray(ch[0]["scene_start"])
        for r in ch[1:]:
            s = np.asarray(r["scene_start"])
            target = [b for b in BLOCKS if b in r["subtask"]]
            moved = max((float(np.linalg.norm(s[sl] - start[sl])) for b, sl in BLOCKS.items() if b not in target),
                        default=0.0)
            drift = moved > DRIFT_M
            if drift:
                tot_d += 1; fail_d += int(not r["success"])
            else:
                tot_n += 1; fail_n += int(not r["success"])
    print(f"chains {len(chains)}, subtask records {len(rows)}, average completed length {np.mean(lengths):.2f}")
    print(f"subtasks 2-5 with a non-target block displaced > {DRIFT_M * 100:.0f} cm since chain start: "
          f"{tot_d} ({tot_d / max(1, tot_d + tot_n):.0%})")
    print(f"  failure rate given drift    {fail_d / max(1, tot_d):.2f}  ({fail_d}/{tot_d})")
    print(f"  failure rate given no drift {fail_n / max(1, tot_n):.2f}  ({fail_n}/{tot_n})")
    all_fail = fail_d + fail_n
    print(f"  share of all later-subtask failures that follow drift: {fail_d / max(1, all_fail):.0%}")
    print("decision rule: port restore to CALVIN only if drift is common AND failure given drift clearly exceeds "
          "failure given no drift")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "analyse"])
    ap.add_argument("--log", required=True)
    ap.add_argument("--num_sequences", type=int, default=200)
    a, rest = ap.parse_known_args()
    if a.mode == "analyse":
        analyse(a.log)
        return
    mod = install_logger(a.log)
    for name in ("NUM_SEQUENCES",):
        if hasattr(mod, name):
            setattr(mod, name, a.num_sequences)
    sys.argv = [sys.argv[0]] + rest
    mod.main()


if __name__ == "__main__":
    main()
