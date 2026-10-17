"""Generate restoring_run_4_learned_gate.ipynb: the learning contribution, a cost-aware learned restore trigger.

Always-restore costs ~400 extra control steps per stage. The learned gate decides, from what the placement policy
itself observes right after a placement (cube positions, stage, neighbours), whether a restore is worth its cost.
  1. log      30 chains (N=4) with a RANDOMISED gate (restore with probability 0.5, propensity recorded)
  2. fit      logistic uplift model P(block ok | features, restore) by inverse-propensity-weighted likelihood
  3. evaluate held-out seeds (N=4): always / never / deviation thresholds / learned at two cost settings /
              random at the learned gate's restore budget
  4. analyse  success vs restore budget, steps per stage, paired McNemar against always-restore and the threshold
Uses the fixed restore (fence check + re-seat) and run 3's trained policy and demos (those cells skip).
"""
import pathlib, shutil, textwrap
import nbformat as nbf
import make_notebook_run3 as R3
from nb_common import HERE, sub, writefile, finish, DRIVE, POLICY, SHUTDOWN

C = []
C.append(nbf.v4.new_markdown_cell(textwrap.dedent("""
# Restoring organs, run 4: learned restore trigger (tag `gate_v1`)

The learning contribution. Always restoring costs about 400 extra control steps per stage. A learned gate decides,
from what the placement policy itself observes after each placement, whether a restore is worth that cost.

1. **Log** (~45 min): 30 chains of 4 stages with a randomised gate (restore with probability 0.5, propensity recorded),
   so the fit is unbiased without counterfactual rollouts.
2. **Fit** (seconds): logistic uplift model of P(block within tolerance | features, restore), inverse-propensity weighted.
3. **Evaluate** (~1.5 h): 10 held-out seeds (N=4), gates `always`, `never`, `dev>8mm`, `dev>4mm`, `learned` at two cost settings,
   and `random` at the learned gate's restore budget.
4. **Analyse**: success vs restore budget and paired tests. **Paste this output.**

The claim to test: the learned gate matches always-restore success with a clearly smaller restore budget, and beats
random restoring at the same budget. The deviation threshold is the honest baseline to beat.

**GPU: T4, about 2.5 to 3 h.** Uses run 3's trained policy and demos (those cells skip). Every chain is saved as it
finishes: after a disconnect, Runtime > Run all again.
""").strip()))
C.append(nbf.v4.new_code_cell(R3.B["install"]))
C.append(nbf.v4.new_code_cell(DRIVE))
C.append(nbf.v4.new_code_cell(writefile("rs_chain.py", R3.B["chain_src"])))
for name in ("rs_geom.py", "rs_pack.py", "rs_pack16.py", "rs_pack_guard.py", "restore_gate.py"):
    C.append(nbf.v4.new_code_cell(writefile(name, (HERE / name).read_text(encoding="utf-8"))))
C.append(nbf.v4.new_code_cell(R3.DEMOS))
C.append(nbf.v4.new_code_cell(R3.TRAIN))
C.append(nbf.v4.new_code_cell(sub(
    POLICY,
    IMPORTS="from rs_chain import Chain, ROW_Y, CUBE_Z, PITCH, X0\nfrom rs_pack import nominal\n"
            "from rs_pack16 import Pack16Chain, N_CELLS\nfrom rs_pack_guard import GuardPack16Chain\nimport restore_gate as G",
    CKDIR="dp_pack3x4",
    CHAIN_CTOR="GuardPack16Chain(seed=7)",
    NOMINAL="def landing_dev(placed, k):          # packing metric: planar distance to the nominal cell\n"
            "    nx, ny = nominal(k); return float(np.hypot(placed[0] - nx, placed[1] - ny))",
    PROBE="[(k, tuple(nominal(k))) for k in (0, 5, 11)]")))

C.append(nbf.v4.new_code_cell(r'''
# ---- 1. log: randomised gate (propensity 0.5), resumable per chain ----
TOL = 0.008                                             # per-cube tolerance, as in runs 1-3
# N=4: the horizon where always-restore completes the block (run 3 and 3b: 6/6) and the bare chain fails (0/6), so final
# success separates the gates. At N=8 always-restore itself ends 0-1/6 on the final-block metric.
N_LOG, LOG_SEEDS, P_LOG = 4, 30, 0.5
LOGF = "gate_logs_v1.json"
place = lambda c, i, t: learned_place(c, i, t, home=True)
logs = [c for c in (load(LOGF) or []) if c.get("tag") == "gate_v1"]
done = {c["seed"] for c in logs}
t0 = time.time()
for s in range(LOG_SEEDS):
    if s in done:
        continue
    ch.reseed(9000 + s); torch.manual_seed(9000 + s); np.random.seed(9000 + s)
    c = G.run_gated(ch, N_LOG, G.gate_random(P_LOG, np.random.default_rng(9000 + s)), place_fn=place)
    c.update(seed=s, n=N_LOG, tag="gate_v1"); logs.append(c); save(LOGF, logs)
    st = c["stages"]
    print(f"log chain {s}: restores {sum(x['restore'] for x in st)}/{N_LOG} final_ok={c['final_success']} "
          f"| {(time.time() - t0) / 60:.0f} min", flush=True)
stages = [x for c in logs for x in c["stages"]]
print(f"{len(logs)} chains, {len(stages)} stages; block ok after restore {np.mean([x['block_ok'] for x in stages if x['restore']]):.2f}"
      f" vs skip {np.mean([x['block_ok'] for x in stages if not x['restore']]):.2f}")
'''.strip()))

C.append(nbf.v4.new_code_cell(r'''
# ---- 2. fit the uplift model (IPW logistic, action interaction) ----
gate = G.LearnedGate().fit(logs)
up = np.array([gate.uplift(x["feat"]) for x in stages])
bad = np.array([x["feat"]["dev"] > TOL for x in stages])
print(f"uplift = P(ok | restore) - P(ok | skip): median {np.median(up):.3f}, landings > 8 mm {np.median(up[bad]) if bad.any() else float('nan'):.3f}, "
      f"landings <= 8 mm {np.median(up[~bad]) if (~bad).any() else float('nan'):.3f}")
for lam in (0.0, 0.05, 0.1, 0.2, 0.3):
    print(f"  lambda {lam:.2f}: would restore {np.mean(up > lam):.2f} of logged stages")
save("gate_model_v1.json", gate.to_json())
'''.strip()))

C.append(nbf.v4.new_code_cell(r'''
# ---- 3. evaluate on held-out seeds, resumable per (gate, seed); every gate sees the same placements per seed ----
N_EVAL, EVAL_SEEDS = 4, 10
EVALF = "gate_eval_v1.json"
ev = [r for r in (load(EVALF) or []) if r.get("tag") == "gate_v1"]
have = {(r["gate"], r["seed"]) for r in ev}
learned = lambda lam: G.LearnedGate.from_json(dict(gate.to_json(), lam=lam))
GATES = [("always", lambda s: G.gate_always), ("never", lambda s: G.gate_never),
         ("dev>8mm", lambda s: G.gate_threshold(TOL)), ("dev>4mm", lambda s: G.gate_threshold(0.004)),
         ("learned_l0.05", lambda s: learned(0.05)), ("learned_l0.15", lambda s: learned(0.15))]
def run_gate(name, make):
    global ev
    for s in range(EVAL_SEEDS):
        if (name, s) in have:
            continue
        ch.reseed(9500 + s); torch.manual_seed(9500 + s); np.random.seed(9500 + s)
        c = G.run_gated(ch, N_EVAL, make(s), place_fn=place)
        c.update(gate=name, seed=s, n=N_EVAL, tag="gate_v1"); ev.append(c); have.add((name, s)); save(EVALF, ev)
        print(f"{name:<14} seed {s} restores {sum(x['restore'] for x in c['stages'])}/{N_EVAL} "
              f"final_ok={c['final_success']} steps/stage {c['steps'] / N_EVAL:.0f}", flush=True)
for name, make in GATES:
    run_gate(name, make)
# random restoring at the learned gate's own budget (the fair control for "the gate knows WHEN")
p_learned = np.mean([x["restore"] for r in ev if r["gate"] == "learned_l0.05" for x in r["stages"]])
print(f"learned_l0.05 restore fraction {p_learned:.2f} -> random gate at the same budget")
run_gate("random_matched", lambda s: G.gate_random(float(p_learned), np.random.default_rng(9500 + s)))
'''.strip()))

C.append(nbf.v4.new_code_cell(r'''
# ---- 4. analysis: success vs restore budget, paired tests (paste this) ----
from scipy.stats import binomtest
ev = [r for r in (load(EVALF) or []) if r.get("tag") == "gate_v1"]
by = {}
for r in ev:
    by.setdefault(r["gate"], []).append(r)
print(f"{'gate':<16}{'final ok':>10}{'restore frac':>14}{'steps/stage':>13}{'stages ok':>11}")
for row in G.pareto(by):
    chains = by[row["gate"]]
    st_ok = np.mean([x["stage_ok"] for c in chains for x in c["stages"]])
    k = sum(c["final_success"] for c in chains)
    print(f"{row['gate']:<16}{f'{k}/{len(chains)}':>10}{row['restore_frac']:>14.2f}{row['steps_per_stage']:>13.0f}{st_ok:>11.2f}")
def paired(a, b):
    A = {r["seed"]: r["final_success"] for r in by.get(a, [])}; B = {r["seed"]: r["final_success"] for r in by.get(b, [])}
    s = sorted(set(A) & set(B)); wa = sum(A[i] and not B[i] for i in s); wb = sum(B[i] and not A[i] for i in s)
    p = binomtest(wa, wa + wb, 0.5).pvalue if wa + wb else 1.0
    print(f"  {a} vs {b}: {len(s)} paired seeds, {a} only {wa}, {b} only {wb}, McNemar p={p:.3f}")
print("\npaired final-block success:")
for a, b in (("learned_l0.05", "always"), ("learned_l0.05", "dev>8mm"), ("learned_l0.05", "random_matched"),
             ("learned_l0.15", "always"), ("dev>8mm", "always")):
    paired(a, b)
print("\nclaim holds if: learned is not worse than always (no significant loss) at a clearly lower restore fraction, "
      "and beats random_matched. With 10 seeds per gate this is a pilot; scale EVAL_SEEDS if the pattern is there.")
'''.strip()))
C.append(nbf.v4.new_code_cell(sub(SHUTDOWN, MANIFEST="manifest_gate_v1.json")))

out = HERE / "notebooks" / "restoring_run_4_learned_gate.ipynb"
out.parent.mkdir(parents=True, exist_ok=True)
finish(C, out)
out.parent.mkdir(parents=True, exist_ok=True)
