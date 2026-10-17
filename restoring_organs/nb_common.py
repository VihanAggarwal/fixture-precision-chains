"""Shared pieces for the run 1+2 and run 3 notebooks.

Reuses make_notebook.py (install cell, patched rs_chain source with the demo recorder, train cell)
by executing it with the notebook write suppressed, so the existing generators and notebooks are
untouched. All new cell sources below are RAW strings: a backslash-n inside a generated print stays
a literal backslash-n. `finish()` ast-parses every code cell and refuses em-dashes."""
import ast, pathlib
import nbformat as nbf

HERE = pathlib.Path(__file__).parent
EM = "—"


def base():
    g = (HERE / "make_notebook.py").read_text(encoding="utf-8")
    for old, new in (("nbf.write(nb, str(out))", "pass  # write suppressed by nb_common"),
                     ('print(f"wrote {out.name}: {len(C)} cells, all code cells parse")', "pass")):
        assert old in g, old
        g = g.replace(old, new)
    ns = {"__file__": str(HERE / "make_notebook.py"), "__name__": "nb_common_base"}
    exec(compile(g, str(HERE / "make_notebook.py"), "exec"), ns)
    C = ns["C"]
    assert C[1].source.startswith("import os, subprocess, sys") and "pip(" in C[1].source, "install cell moved"
    assert "# ---- 2. train" in C[5].source, "train cell moved"
    return dict(install=C[1].source, chain_src=ns["S"], train=C[5].source)


def sub(src, **kw):
    for k, v in kw.items():
        key = "@@" + k + "@@"
        assert key in src, key
        src = src.replace(key, str(v))
    assert "@@" not in src, src[src.index("@@"):src.index("@@") + 40]
    return src


def writefile(name, text):
    return "%%writefile " + name + "\n" + text


# ---------------------------------------------------------------- Drive (sharp edge 3)
DRIVE = r'''
# ---- Drive: mount, then PROVE the mount is real before touching any Drive path ----
from google.colab import drive
drive.mount("/content/drive")
OUT = "/content/drive/MyDrive/restoring_organs"      # checkpoints, demos, mirrored results
RES = "/content/results"                              # results are written here FIRST, then mirrored to Drive
KNOWN = f"{OUT}/demos.hdf5"                           # a file that is known to exist on a real mount
import os, json, pathlib, time, shutil, numpy as np
if not os.path.isfile(KNOWN):
    raise SystemExit(f"Drive is NOT really mounted ({KNOWN} not found). Run drive.mount('/content/drive', force_remount=True), "
                     "then re-run this cell. Nothing was written.")
os.makedirs(RES, exist_ok=True)

def drive_ok():
    return os.path.isfile(KNOWN)

def mirror(name):
    # copy RES/name to Drive; never fatal, never a rename or delete on Drive
    try:
        if not drive_ok():
            print("WARNING: Drive not mounted, local copy only:", name); return False
        shutil.copyfile(f"{RES}/{name}", f"{OUT}/{name}"); return True
    except Exception as ex:
        print("WARNING: Drive mirror failed for", name, repr(ex), "(local copy kept)"); return False

def save(name, obj):
    tmp = f"{RES}/{name}.tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, f"{RES}/{name}")                  # local disk only
    mirror(name)

def load(name):
    # the longer of the local and Drive copies: a fresh VM only has Drive, an unmirrored save only has local
    best = None
    for p in (f"{RES}/{name}", f"{OUT}/{name}"):
        try:
            v = json.load(open(p))
        except Exception:
            continue
        if best is None or len(v) > len(best):
            best = v
    return best

print("Drive mount verified via", KNOWN)
print("Drive folder:", sorted(p.name for p in pathlib.Path(OUT).iterdir()))
'''.strip()


# ---------------------------------------------------------------- policy loading + learned placement
POLICY = r'''
# ---- load the diffusion policy (DDIM-10 sampling), define the learned placement ----
import torch, glob, re, logging
from collections import deque
from robomimic.utils.file_utils import policy_from_checkpoint
logging.getLogger("robosuite_logs").setLevel(logging.ERROR)
@@IMPORTS@@
EPOCHS = 200
CK_DIR = f"{OUT}/@@CKDIR@@"
CKPT = sorted(glob.glob(f"{CK_DIR}/**/models/*.pth", recursive=True), key=lambda p: int(re.search(r"epoch_(\d+)", p).group(1)))
LAST_EP = int(re.search(r"epoch_(\d+)", CKPT[-1]).group(1)) if CKPT else 0
assert LAST_EP >= EPOCHS, f"need the final epoch {EPOCHS} checkpoint under {CK_DIR}, found epoch {LAST_EP}"
print("using checkpoint:", CKPT[-1])
ck = torch.load(CKPT[-1], map_location="cpu", weights_only=False)
_cfg = json.loads(ck["config"]) if isinstance(ck["config"], str) else ck["config"]
_cfg["algo"]["ddpm"]["enabled"] = False                 # DDIM-10 at inference (7x faster, weights unchanged)
_cfg["algo"]["ddim"]["enabled"] = True
_cfg["algo"]["ddim"]["num_inference_timesteps"] = 10
ck["config"] = json.dumps(_cfg)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy, _ = policy_from_checkpoint(ckpt_dict=ck, device=DEVICE, verbose=False)
FRAME_STACK = int(_cfg["train"].get("frame_stack", 1))
T_POLICY = 240
print("frame_stack =", FRAME_STACK, "| sampler = DDIM-10 | device =", DEVICE)

ch = @@CHAIN_CTOR@@                                     # ONE env for the whole notebook (robosuite leaks envs)
ch.obs = ch.env.reset()
HOME = ch.eef().copy(); HOME_QUAT = np.array(ch.obs["robot0_eef_quat"], float).copy()   # every demo started here
print("home eef:", np.round(HOME, 3), "quat:", np.round(HOME_QUAT, 3))

def learned_place(chain, i, target, home=True):
    # home=True: full-pose homing first (position AND orientation), the policy's training start state.
    # home=False is ONLY for the no-robot-reset arm.
    if home:
        p = chain.eef(); chain.goto([p[0], p[1], HOME[2]], -1, 15)
        chain.home_pose(HOME, HOME_QUAT, -1)
    if isinstance(target, (tuple, list, np.ndarray)):
        tx, ty = float(target[0]), float(target[1])
    else:
        tx, ty = float(target), ROW_Y
    chain.cur_i = i; chain.cur_target = np.array([tx, ty, CUBE_Z]); policy.start_episode()
    hist = deque([chain.obs_dict()] * FRAME_STACK, maxlen=FRAME_STACK)
    for _ in range(T_POLICY):
        ob = {k: np.stack([h[k] for h in hist]) for k in hist[0]} if FRAME_STACK > 1 else hist[-1]
        a = np.clip(np.asarray(policy(ob=ob)), -1, 1)
        chain._step(a)
        hist.append(chain.obs_dict())
    chain.settle(8)

@@NOMINAL@@

# single-stage probe: the learned policy's own error sets the regime
errs = []
for k, tgt in @@PROBE@@:
    ch.obs = ch.env.reset(); learned_place(ch, k, tgt)
    c = ch.env.cube_pos(k); t = tgt if isinstance(tgt, (tuple, list)) else (tgt, ROW_Y)
    errs.append(float(np.hypot(c[0] - t[0], c[1] - t[1])))
print("learned single-stage |error| (m):", " ".join(f"{v:.4f}" for v in errs), f"| mean {np.mean(errs):.4f}")
'''.strip()


# ---------------------------------------------------------------- sweep (resumable per chain)
SWEEP = r'''
# ---- sweep: resumable per chain, keyed on (arm, N, seed) and the sampler TAG ----
N_LIST = @@N_LIST@@; SEEDS = @@SEEDS@@; TOL = 0.008
ARMS = @@ARMS@@            # arm -> (chain mode, full-pose homing before each learned placement)
TAG = "@@TAG@@"; RESULTS = "@@RESULTS@@"   # a NEW tag and file: rows from older runs are ignored, nothing is deleted
EST_S_PER_STAGE = @@EST@@  # T4 seconds per stage, calibrated on the v7 line-up run (90 chains, 83 min)
rows = [r for r in (load(RESULTS) or []) if r.get("sampler") == TAG]
done = {(r["mode"], r["n"], r["seed"]) for r in rows}
# ordered by (N, seed) with every arm inside: an interrupted sweep still holds PAIRED comparisons
todo = [(n, s, arm) for n in N_LIST for s in range(SEEDS) for arm in ARMS if (arm, n, s) not in done]
est = sum(n * EST_S_PER_STAGE[arm] for n, _, arm in todo) / 60
print(f"tag {TAG}: {len(rows)} chains on file, {len(todo)} to run, estimated {est:.0f} min ({est/60:.1f} h) on a T4")
t0 = time.time(); stages_done = 0; stages_total = sum(n for n, _, _ in todo)
for n, s, arm in todo:
    mode, home = ARMS[arm]
    ch.reseed(500 + 17 * n + s)
    torch.manual_seed(100000 + 1000 * n + s); np.random.seed(100000 + 1000 * n + s)   # paired: same sampler seed in every arm
    ch.fence_rejects = 0
    d = ch.run(n, mode=mode, place_fn=lambda c, i, t, _h=home: learned_place(c, i, t, home=_h))
    rows.append(dict(mode=arm, chain_mode=mode, home=home, n=n, seed=s, dev=[float(v) for v in d], placed=ch.placed,
                     final=ch.final, success=bool(np.all(d <= TOL)), final_success=bool(all(v <= TOL for v in ch.final)),
                     steps=ch.steps_used, sampler=TAG, collateral=getattr(ch, "collateral", None),
                     reseated=getattr(ch, "reseated", None), fence_rejects=ch.fence_rejects))
    save(RESULTS, rows)
    stages_done += n; el = (time.time() - t0) / 60
    eta = el / stages_done * (stages_total - stages_done)
    print(f"{arm:<12} N={n:<2} seed={s} ok={rows[-1]['success']} final_ok={rows[-1]['final_success']} "
          f"dev(mm)={' '.join(f'{1000*v:.0f}' for v in d)} | {el:.0f} min, ~{eta:.0f} min left", flush=True)
print("sweep complete:", len(rows), "chains for tag", TAG)
'''.strip()


# ---------------------------------------------------------------- analysis
ANALYSIS = r'''
# ---- analysis: success per arm, censoring-aware per-stage bounds, paired contrasts ----
from scipy.stats import beta as _beta, binomtest
rows = [r for r in (load(RESULTS) or []) if r.get("sampler") == TAG]
ARM_NAMES = [a for a in ARMS if any(r["mode"] == a for r in rows)]
def sel(arm, n): return [r for r in rows if r["mode"] == arm and r["n"] == n]
print(f"{'arm':<12}" + "".join(f"   N={n:<5}" for n in N_LIST) + "  steps/stage")
for a in ARM_NAMES:
    cells = []
    for n in N_LIST:
        rr = sel(a, n); cells.append(f"{sum(r['success'] for r in rr)}/{len(rr)}".rjust(8) + "  ")
    st = np.mean([r["steps"] / r["n"] for r in rows if r["mode"] == a])
    print(f"{a:<12}" + " ".join(cells) + f"  {st:7.0f}")
print("\nfinal success (every cube within tol at the END of the chain):")
for a in ARM_NAMES:
    print(f"  {a:<12}" + "  ".join(f"N={n}: {np.mean([r['final_success'] for r in sel(a, n)]) if sel(a, n) else float('nan'):.2f}" for n in N_LIST))

# censoring-aware per-stage reliability: Clopper-Pearson on observed successes, to the 1/N power, tightest over N
def _cp(k, n, upper, a=0.05):
    if upper: return 1.0 if k == n else float(_beta.ppf(1 - a, k + 1, n - k))
    return 0.0 if k == 0 else float(_beta.ppf(a, k, n - k + 1))
def bounds(arm):
    ks = {n: (int(sum(r["success"] for r in sel(arm, n))), len(sel(arm, n))) for n in N_LIST if sel(arm, n)}
    hi = min(_cp(k, m, True) ** (1 / n) for n, (k, m) in ks.items())
    lo = max(_cp(k, m, False) ** (1 / n) for n, (k, m) in ks.items())
    return lo, hi
print("\nper-stage reliability bounds (95% one-sided, tightest over N):")
B = {a: bounds(a) for a in ARM_NAMES}
for a in ARM_NAMES:
    print(f"  {a:<12} >= {B[a][0]:.3f}   <= {B[a][1]:.3f}")
if "restore" in B:
    others = [a for a in ARM_NAMES if a != "restore"]
    ok = all(B["restore"][0] > B[a][1] for a in others)
    print("\nACCEPT: restore's lower bound exceeds the upper bound of every other arm" if ok and others
          else "\nNOT ACCEPTED on the bounds test (report as is)")

# paired contrasts: same (N, seed) in both arms
CONTRASTS = @@CONTRASTS@@
def _mdev(r): return float(np.mean(r["dev"]))
summary = {"tag": TAG, "success": {a: {str(n): [int(sum(r["success"] for r in sel(a, n))), len(sel(a, n))] for n in N_LIST} for a in ARM_NAMES},
           "bounds": B, "contrasts": {}}
print("\npaired contrasts (A minus B; exact McNemar p on discordant pairs pooled over N; 95% bootstrap CI over pairs):")
rng = np.random.default_rng(0)
for A_, B_, what in CONTRASTS:
    if A_ not in ARM_NAMES or B_ not in ARM_NAMES:
        print(f"  {A_} vs {B_}: missing arm, skipped"); continue
    ia = {(r["n"], r["seed"]): r for r in rows if r["mode"] == A_}; ib = {(r["n"], r["seed"]): r for r in rows if r["mode"] == B_}
    keys = sorted(set(ia) & set(ib))
    if not keys:
        print(f"  {A_} vs {B_}: no paired chains yet"); continue
    ds = np.array([int(ia[k]["success"]) - int(ib[k]["success"]) for k in keys], float)
    dd = np.array([_mdev(ia[k]) - _mdev(ib[k]) for k in keys], float)
    b = int((ds > 0).sum()); c = int((ds < 0).sum())
    p = binomtest(b, b + c, 0.5).pvalue if b + c > 0 else 1.0
    bs = [rng.choice(len(keys), len(keys), replace=True) for _ in range(2000)]
    ci_s = np.percentile([ds[i].mean() for i in bs], [2.5, 97.5]); ci_d = np.percentile([dd[i].mean() for i in bs], [2.5, 97.5])
    per_n = "  ".join(f"N={n}: {np.mean([ds[j] for j, k in enumerate(keys) if k[0] == n]):+.2f}" for n in N_LIST if any(k[0] == n for k in keys))
    print(f"  {what}\n    {A_} - {B_}: success {ds.mean():+.2f} [{ci_s[0]:+.2f},{ci_s[1]:+.2f}]  ({per_n})  discordant {b}:{c}  p={p:.3g}"
          f"\n    mean per-chain deviation {1000*dd.mean():+.1f} mm [{1000*ci_d[0]:+.1f},{1000*ci_d[1]:+.1f}]  over {len(keys)} pairs")
    summary["contrasts"][f"{A_}-{B_}"] = dict(what=what, d_success=float(ds.mean()), ci=ci_s.tolist(), discordant=[b, c], p=float(p),
                                              d_dev_mm=float(1000 * dd.mean()), ci_dev_mm=(1000 * ci_d).tolist(), pairs=len(keys))
save(RESULTS.replace(".json", "_summary.json"), summary)
'''.strip()


# ---------------------------------------------------------------- error recursion
RECURSION = r'''
# ---- error recursion: e_{k+1} = a * e_k + b on the longest chains, per arm ----
# Fit on all consecutive stage pairs pooled over seeds (least squares); CIs by bootstrapping SEEDS (whole chains).
# Fixed point b / (1 - a) exists only when |a| < 1. Claim under test: restore imposes a low fixed point (a near 0,
# small b); the bare chain accumulates (a near 1, or a large b).
# Caveat: along a chain this regression reports a near 1 for ANY smooth ramp, so read it together with the
# within-stage contraction below (deviation after the stage / deviation right after the placement).
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
rows = [r for r in (load(RESULTS) or []) if r.get("sampler") == TAG]
NREC = max(N_LIST)
ARM_NAMES = [a for a in ARMS if any(r["mode"] == a and r["n"] == NREC for r in rows)]
def chains(arm): return [np.asarray(r["dev"], float) for r in rows if r["mode"] == arm and r["n"] == NREC]
def fit(ch_list):
    x = np.concatenate([d[:-1] for d in ch_list]) if ch_list else np.zeros(0)
    y = np.concatenate([d[1:] for d in ch_list]) if ch_list else np.zeros(0)
    if len(x) < 2 or np.ptp(x) == 0:
        return float("nan"), float("nan"), float("nan")
    a, b = np.linalg.lstsq(np.c_[x, np.ones_like(x)], y, rcond=None)[0]
    fp = b / (1 - a) if abs(a) < 1 else float("nan")
    return float(a), float(b), float(fp)
rec = {}; rng = np.random.default_rng(1)
print(f"error recursion on N={NREC} chains (deviations in mm):")
print(f"  {'arm':<12} {'a [95% CI]':<24} {'b mm [95% CI]':<24} {'fixed point mm [95% CI]':<28} P(|a|<1)  seeds")
for arm in ARM_NAMES:
    cl = chains(arm); a, b, fp = fit(cl)
    boot = [fit([cl[i] for i in rng.integers(0, len(cl), len(cl))]) for _ in range(2000)]
    A = np.array([v[0] for v in boot]); Bv = np.array([v[1] for v in boot]); F = np.array([v[2] for v in boot])
    def ci(v, s=1.0):
        v = v[np.isfinite(v)]
        return (s * np.percentile(v, 2.5), s * np.percentile(v, 97.5)) if len(v) else (float("nan"), float("nan"))
    ca, cb, cf = ci(A), ci(Bv, 1000), ci(F, 1000)
    frac = float(np.mean(np.abs(A[np.isfinite(A)]) < 1)) if np.isfinite(A).any() else float("nan")
    print(f"  {arm:<12} {a:+.3f} [{ca[0]:+.3f},{ca[1]:+.3f}]   {1000*b:6.2f} [{cb[0]:6.2f},{cb[1]:6.2f}]   "
          f"{1000*fp:7.2f} [{cf[0]:7.2f},{cf[1]:7.2f}]      {frac:.2f}     {len(cl)}")
    rec[arm] = dict(a=a, a_ci=list(ca), b_mm=1000 * b, b_ci_mm=list(cb), fixed_point_mm=1000 * fp, fp_ci_mm=list(cf), p_abs_a_lt_1=frac, seeds=len(cl))

print(f"\nmean deviation per stage (mm), N={NREC}:")
print(f"  {'arm':<12}" + "".join(f"{k+1:>7}" for k in range(NREC)))
for arm in ARM_NAMES:
    m = np.mean(np.stack(chains(arm)), axis=0) if chains(arm) else np.full(NREC, np.nan)
    print(f"  {arm:<12}" + "".join(f"{1000*v:7.1f}" for v in m)); rec[arm]["mean_dev_mm"] = (1000 * m).tolist()

print("\nwithin-stage contraction, median over stages of dev(after stage) / dev(landing), landings > 2 mm:")
for arm in ARM_NAMES:
    ratios = [r["dev"][k] / landing_dev(r["placed"][k], k) for r in rows if r["mode"] == arm and r["n"] == NREC
              for k in range(NREC) if landing_dev(r["placed"][k], k) > 0.002]
    print(f"  {arm:<12} {np.median(ratios) if ratios else float('nan'):.3f}   ({len(ratios)} stages)")
    rec[arm]["within_stage_median_ratio"] = float(np.median(ratios)) if ratios else None

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
cols = plt.rcParams["axes.prop_cycle"].by_key()["color"]
for j, arm in enumerate(ARM_NAMES):
    cl = chains(arm)
    x = np.concatenate([d[:-1] for d in cl]); y = np.concatenate([d[1:] for d in cl])
    ax[0].scatter(1000 * np.maximum(x, 1e-4), 1000 * np.maximum(y, 1e-4), s=12, alpha=0.5, color=cols[j % len(cols)], label=arm)
    if np.isfinite(rec[arm]["a"]):
        g = np.linspace(max(x.min(), 1e-4), max(x.max(), 2e-4), 50)
        ax[0].plot(1000 * g, 1000 * np.maximum(rec[arm]["a"] * g + rec[arm]["b_mm"] / 1000, 1e-4), color=cols[j % len(cols)], lw=1.5)
    ax[1].plot(range(1, NREC + 1), rec[arm]["mean_dev_mm"], "o-", color=cols[j % len(cols)], label=arm)
lim = [0.1, 400]
ax[0].plot(lim, lim, "k--", lw=0.8, label="e(k+1) = e(k)")
ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("e_k (mm)"); ax[0].set_ylabel("e_(k+1) (mm)")
ax[0].set_title(f"error recursion, N={NREC}"); ax[0].legend(fontsize=8)
ax[1].axhline(1000 * TOL, color="k", ls=":", lw=0.8, label="tolerance")
ax[1].set_yscale("log"); ax[1].set_xlabel("stage"); ax[1].set_ylabel("mean deviation (mm)")
ax[1].set_title("mean deviation vs stage"); ax[1].legend(fontsize=8)
fig.tight_layout()
PNG = RESULTS.replace(".json", "_recursion.png")
fig.savefig(f"{RES}/{PNG}", dpi=150); plt.close(fig)
print("\nfigure:", f"{RES}/{PNG}", "| mirrored to Drive:", mirror(PNG))
save(RESULTS.replace(".json", "_recursion.json"), rec)
'''.strip()


SHUTDOWN = r'''
# ---- save and shut down: mirror every result to Drive, write a manifest, flush ----
rows = [r for r in (load(RESULTS) or []) if r.get("sampler") == TAG]
mirrored = {p.name: mirror(p.name) for p in sorted(pathlib.Path(RES).iterdir()) if p.is_file() and not p.name.endswith(".tmp")}
manifest = {"saved_at": time.strftime("%Y-%m-%d %H:%M"), "tag": TAG, "results": RESULTS, "chains": len(rows),
            "expected_chains": len(N_LIST) * SEEDS * len(ARMS), "checkpoint": CKPT[-1] if CKPT else None, "mirrored": mirrored}
with open(f"{RES}/@@MANIFEST@@", "w") as f:
    json.dump(manifest, f, indent=1)
mirror("@@MANIFEST@@")
print(json.dumps(manifest, indent=1))
if manifest["chains"] < manifest["expected_chains"]:
    print("sweep INCOMPLETE: run the notebook top to bottom again, finished chains are skipped")
from google.colab import drive
drive.flush_and_unmount(); print("Drive flushed, safe to terminate.")
'''.strip()


def finish(cells, out_path):
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
    nb.metadata["accelerator"] = "GPU"
    n_code = 0
    for i, c in enumerate(cells):
        assert EM not in c.source, f"em-dash in cell {i}"
        if c.cell_type != "code":
            continue
        src = c.source.split("\n", 1)[1] if c.source.startswith("%%writefile") else c.source
        try:
            ast.parse(src)
        except SyntaxError as ex:
            raise SystemExit(f"cell {i} does not parse: {ex}\n---\n" + src.splitlines()[max(ex.lineno - 3, 0)] if ex.lineno else str(ex))
        assert "@@" not in src, f"unfilled placeholder in cell {i}"
        n_code += 1
    nbf.write(nb, str(out_path))
    print(f"wrote {out_path}: {len(cells)} cells ({n_code} code), all code cells parse, no em-dashes")
