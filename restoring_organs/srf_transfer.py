"""Transfer experiments for the Sim-to-Real-to-Field workshop version of the restoring paper.

Four parts, all on the 3-D Panda line-up with the scripted placer (rs_chain.Chain), restore code unchanged:
  gap      policy-side transfer gap: placement scatter sigma and a systematic bias (a calibration offset)
  contact  the restore primitive, tuned in the default simulator, executed under shifted contact physics
  percep   restore driven by a noisy estimate of the part pose instead of simulator state
Every restore row also logs the plate's x position at the end of the final push (proprioception only), so the
onboard check of whether a part seated can be scored afterwards.

Rows are appended one per chain to srf_<part>.jsonl, so a crash loses at most one chain and a rerun resumes.
Usage: python srf_transfer.py gap|contact|percep
"""
import json, os, sys, time
import numpy as np
from rs_chain import Chain, X0, PITCH

PART = sys.argv[1]
OUT = f"srf_{PART}.jsonl"
done = set()
if os.path.exists(OUT):
    for line in open(OUT):
        r = json.loads(line); done.add(r["key"])


class Probe(Chain):
    """Chain that records the plate position at the end of every x push on the pushed part, and can corrupt the
    part poses the restore routine sees (perception noise). Measurement in run() always uses the true pose."""
    percep_sd = 0.0
    _in_restore = False

    def push_axis(self, axis, target, hold, grip, max_steps, **kw):
        x_start = float(self.eef()[0])
        p = super().push_axis(axis, target, hold, grip, max_steps, **kw)
        if axis == 0 and kw.get("cube") is not None and self._in_restore:
            self.last_x_push = float(p[0])
            self.last_travel = x_start - float(p[0])     # < ~25 mm: the push ended before the plate reached the part
        return p

    def restore(self, i, prev_x, **kw):
        self.last_x_push = None; self.last_travel = None
        env = self.env
        true_pos = env.cube_pos
        if self.percep_sd > 0:
            bias = {}
            def noisy(j):
                if j not in bias:            # one perception error per part per restore call (a pose estimate, not jitter)
                    bias[j] = np.r_[self.prng.normal(0, self.percep_sd, 2), 0.0]
                return true_pos(j) + bias[j]
            env.cube_pos = noisy
        self._in_restore = True
        try:
            super().restore(i, prev_x, **kw)
        finally:
            self._in_restore = False
            env.cube_pos = true_pos
        self.stall_log.append(self.last_x_push)
        self.travel_log.append(getattr(self, "last_travel", None))


def chain(ch, n, mode, seed, err_sd, bias=0.0):
    ch.reseed(seed); ch.prng = np.random.default_rng(10_000 + seed); ch.stall_log = []; ch.travel_log = []
    t0 = time.time()
    if bias:
        def place_fn(c, i, target):
            c.place(i, target + bias, err_sd)
        devs = ch.run(n, mode=mode, place_fn=place_fn)
    else:
        devs = ch.run(n, mode=mode, err_sd=err_sd)
    landed = [abs(p[0] - (X0 + k * PITCH)) for k, p in enumerate(ch.placed)]
    return dict(devs=[round(float(d), 5) for d in devs], final=ch.final, landed=[round(float(v), 5) for v in landed],
                stall_x=ch.stall_log if mode == "restore" else None,
                travel=ch.travel_log if mode == "restore" else None, steps=ch.steps_used, sec=round(time.time() - t0, 1))


def emit(key, row):
    row["key"] = key
    with open(OUT, "a") as f:
        f.write(json.dumps(row) + "\n")
    fin = row["final"]
    print(f"{key:<48} final max {1000*max(fin):6.1f} mm  mean {1000*np.mean(fin):5.1f} mm  {row['sec']:6.1f}s", flush=True)


ch = Probe(n_cubes=8, seed=0)
m = ch.env.sim.model._model
base_solref = m.geom_solref.copy(); base_fric = m.geom_friction.copy()
base_cone, base_imp = int(m.opt.cone), float(m.opt.impratio)


def physics(tc=None, mu=1.0, cone=None, imp=None):
    m.geom_solref[:] = base_solref; m.geom_friction[:] = base_fric
    if tc is not None:
        m.geom_solref[:, 0] = tc
    m.geom_friction[:, 0] = base_fric[:, 0] * mu
    m.opt.cone = base_cone if cone is None else cone
    m.opt.impratio = base_imp if imp is None else imp


print(f"default physics: solref tc {base_solref[0,0]}, cone {base_cone}, impratio {base_imp}", flush=True)
MODES = ("none", "restore")

if PART == "gap":
    N, SEEDS = 8, range(5)
    grid = [(sd, 0.0) for sd in (0.002, 0.004, 0.008, 0.012, 0.016)] + [(0.004, b) for b in (0.005, 0.010, 0.020)]
    for seed in SEEDS:                        # seed-major so a partial run still covers every setting
        for sd, b in grid:
            for mode in MODES:
                key = f"gap sd={sd} bias={b} {mode} s{seed}"
                if key in done:
                    continue
                r = chain(ch, N, mode, 100 + seed, sd, b); r.update(part="gap", sd=sd, bias=b, mode=mode, seed=seed, N=N)
                emit(key, r)

elif PART == "contact":
    N, SEEDS = 4, range(4)
    grid = [dict(name="default")]
    grid += [dict(name=f"tc={tc}", tc=tc) for tc in (0.005, 0.01, 0.04)]
    grid += [dict(name=f"mu={mu}", mu=mu) for mu in (0.5, 0.75, 1.5, 2.0)]
    grid += [dict(name="tc=0.01 mu=2.0", tc=0.01, mu=2.0), dict(name="tc=0.04 mu=0.5", tc=0.04, mu=0.5)]
    grid += [dict(name=f"cone={1 - base_cone}", cone=1 - base_cone), dict(name="impratio=1", imp=1.0),
             dict(name="impratio=20", imp=20.0)]
    for seed in SEEDS:
        for g in grid:
            physics(**{k: v for k, v in g.items() if k != "name"})
            for mode in MODES:
                key = f"contact {g['name']} {mode} s{seed}"
                if key in done:
                    continue
                r = chain(ch, N, mode, 900 + seed, 0.004); r.update(part="contact", cfg=g["name"], mode=mode, seed=seed, N=N)
                emit(key, r)
    physics()

elif PART == "percep":
    N, SEEDS = 4, range(4)
    for seed in SEEDS:
        for psd in (0.0, 0.003, 0.006, 0.010):
            key = f"percep psd={psd} restore s{seed}"
            if key in done:
                continue
            ch.percep_sd = psd
            r = chain(ch, N, "restore", 500 + seed, 0.008); r.update(part="percep", psd=psd, mode="restore", seed=seed, N=N)
            emit(key, r)
    ch.percep_sd = 0.0
print("done", flush=True)
