"""Learning contribution for the restoring-organs paper: a cost-aware learned restore trigger.

Always-restore costs ~400 extra control steps per stage. The question a learned gate answers: given what the robot
observes right after a placement, is a restore worth its cost for the rest of the chain?

Pipeline
  1. log     run chains with a RANDOMISED gate (restore with probability p, propensity recorded). Every stage logs
             policy-observable features at the decision point and what happened next.
  2. fit     a logistic model of P(stage ends within tolerance | features, action) with an action interaction, fitted
             by inverse-propensity-weighted likelihood on the logged data (off-policy, no counterfactual rollouts).
  3. gate    restore iff P(ok | restore) - P(ok | skip) > lambda. Sweeping lambda traces a success-vs-restore-budget
             curve; the claim is that it dominates always / never / random-at-matched-budget and the best
             single-feature deviation threshold.
Features use only what the placement policy itself observes (cube positions, stage index), plus the neighbourhood of
the current cell: the gate is not given anything the policy could not see.
"""
import json
import sys
import time

import numpy as np

TOL = 0.008
FEATURES = ["dev", "dx", "dy", "stage_frac", "nbr_max_dev", "nbr_mean_dev", "n_bad_nbrs", "row", "col"]


def features(chain, k, n, nominal, cell, cols):
    c = chain.env.cube_pos(k)
    nx, ny = nominal(k)
    r, col = cell(k)
    nbrs = [j for j in range(k) if abs(cell(j)[0] - r) + abs(cell(j)[1] - col) == 1]
    nd = [chain.dev(j) for j in nbrs] or [0.0]
    return dict(dev=float(np.hypot(c[0] - nx, c[1] - ny)), dx=float(c[0] - nx), dy=float(c[1] - ny),
                stage_frac=k / max(1, n - 1), nbr_max_dev=float(max(nd)), nbr_mean_dev=float(np.mean(nd)),
                n_bad_nbrs=int(sum(d > TOL for d in nd)), row=r, col=col)


def run_gated(chain, n, gate, place_fn=None, err_sd=0.0, rng=None, reseat=True):
    """One chain with a restore decision per stage. gate(feat) -> (restore: bool, propensity: float).
    A restore is the fixed restore (fence sanity check in rs_chain) plus the re-seat pass when reseat=True."""
    from rs_chain import CUBE_Z, HOVER
    from rs_pack import cell, nominal, COLS
    chain.cell_of = {k: k for k in range(n)}                 # row-major fill, as in runs 3 and 3b
    chain.y_push_dx = 0.0
    chain.obs = chain.env.reset()
    log = []
    for k in range(n):
        if place_fn is None:
            chain.place_cell(k, k, err_sd)
        else:
            place_fn(chain, k, nominal(k))
        feat = features(chain, k, n, nominal, cell, COLS)
        restore, prop = gate(feat)
        s0 = chain.steps_used
        if restore:
            if reseat and hasattr(chain, "restore_and_reseat"):
                chain.restore_and_reseat(k, k)
            else:
                chain.restore_cell(k, k)
        p = chain.eef(); chain.goto([p[0], p[1], CUBE_Z + HOVER + 0.05], -1, 15); chain.settle(15)
        log.append(dict(k=k, feat=feat, restore=bool(restore), propensity=float(prop),
                        dev_after=float(chain.dev(k)), stage_ok=bool(chain.dev(k) <= TOL),
                        block_ok=bool(all(chain.dev(j) <= TOL for j in range(k + 1))),
                        restore_steps=chain.steps_used - s0))
    final_ok = all(chain.dev(j) <= TOL for j in range(n))
    return dict(stages=log, final_success=bool(final_ok), steps=chain.steps_used)


# ------------------------------------------------------------------ gates
def gate_always(f):
    return True, 1.0


def gate_never(f):
    return False, 1.0


def gate_random(p, rng):
    return lambda f: (bool(rng.random() < p), p)


def gate_threshold(t):
    return lambda f: (f["dev"] > t, 1.0)


def design(feat, action):
    x = np.array([feat[k] for k in FEATURES], float)
    a = float(action)
    return np.concatenate([[1.0], x, [a], a * x])


class LearnedGate:
    """Logistic model with an action interaction, fitted by IPW likelihood; restore iff uplift > lam."""

    def __init__(self, w=None, mu=None, sd=None, lam=0.0):
        self.w, self.mu, self.sd, self.lam = w, mu, sd, lam

    def _x(self, feat, a):
        z = dict(feat)
        for i, k in enumerate(FEATURES):
            z[k] = (feat[k] - self.mu[i]) / self.sd[i]
        return design(z, a)

    def prob(self, feat, a):
        return float(1 / (1 + np.exp(-self._x(feat, a) @ self.w)))

    def uplift(self, feat):
        return self.prob(feat, 1) - self.prob(feat, 0)

    def __call__(self, feat):
        return self.uplift(feat) > self.lam, 1.0

    def fit(self, logs, label="block_ok", l2=1e-2, iters=3000, lr=0.1):
        rows = [s for ch in logs for s in ch["stages"]]
        F = np.array([[s["feat"][k] for k in FEATURES] for s in rows], float)
        self.mu, self.sd = F.mean(0), F.std(0) + 1e-6
        X = np.stack([self._x(s["feat"], s["restore"]) for s in rows])
        y = np.array([float(s[label]) for s in rows])
        wts = np.array([1.0 / (s["propensity"] if s["restore"] else 1.0 - s["propensity"]) for s in rows])
        wts = wts / wts.mean()
        self.w = np.zeros(X.shape[1])
        for _ in range(iters):
            p = 1 / (1 + np.exp(-X @ self.w))
            g = X.T @ ((p - y) * wts) / len(y) + l2 * self.w
            self.w -= lr * g
        return self

    def to_json(self):
        return dict(w=self.w.tolist(), mu=self.mu.tolist(), sd=self.sd.tolist(), lam=self.lam, features=FEATURES)

    @classmethod
    def from_json(cls, d):
        return cls(np.array(d["w"]), np.array(d["mu"]), np.array(d["sd"]), d["lam"])


def pareto(results):
    """results: {gate_name: [chain dicts]} -> rows of (gate, success rate, restore fraction, steps per stage)."""
    out = []
    for name, chains in results.items():
        st = [s for c in chains for s in c["stages"]]
        out.append(dict(gate=name, success=float(np.mean([c["final_success"] for c in chains])),
                        restore_frac=float(np.mean([s["restore"] for s in st])),
                        steps_per_stage=float(np.mean([c["steps"] / len(c["stages"]) for c in chains]))))
    return sorted(out, key=lambda r: r["restore_frac"])


if __name__ == "__main__":
    # local smoke with the scripted placer: log a few randomised chains, fit, compare gates on fresh seeds
    from rs_pack_guard import GuardPack16Chain
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    err_sd = float(sys.argv[2]) if len(sys.argv) > 2 else 0.012
    n_log = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    ch = GuardPack16Chain(seed=0); rng = np.random.default_rng(0); t0 = time.time()
    logs = []
    for s in range(n_log):
        ch.reseed(7000 + s); logs.append(run_gated(ch, n, gate_random(0.5, rng), err_sd=err_sd))
        print(f"logged chain {s}: final {logs[-1]['final_success']} {(time.time() - t0) / 60:.1f} min", flush=True)
    gate = LearnedGate(lam=0.05).fit(logs)
    print("uplift on logged stages:", [round(gate.uplift(st["feat"]), 3) for c in logs for st in c["stages"]])
    res = {}
    for name, g in (("always", gate_always), ("never", gate_never), ("learned", gate)):
        ch.reseed(8000); res[name] = [run_gated(ch, n, g, err_sd=err_sd)]
    for r in pareto(res):
        print(r)
    json.dump(dict(logs=logs, gate=gate.to_json()), open("_restore_gate_smoke.json", "w"))
