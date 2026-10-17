"""Generate restoring_colab.ipynb: the GPU gate.

Inlines the locally tested rs_chain.py (with a demo recorder patched in), collects
expert demonstrations, trains a robomimic diffusion policy, then runs the three-arm
chain experiment with the LEARNED policy. Every stage checkpoints to Drive and is
skipped on rerun if its output exists.
"""
import ast, pathlib, textwrap
import nbformat as nbf

SRC = pathlib.Path(__file__).with_name("rs_chain.py").read_text()

# ---- patch a recorder + policy-facing observation into the tested module ----
def patch(s, old, new):
    assert old in s, old[:60]
    return s.replace(old, new, 1)

S = SRC
S = patch(S, "        self.obs = self.env.reset()\n        self.steps_used = 0\n",
          "        self.obs = self.env.reset()\n        self.steps_used = 0\n"
          "        self.rec = None          # list of (obs_dict, action) while recording demos\n"
          "        self.cur_i = 0\n        self.cur_target = np.zeros(3)\n")
S = patch(S, "    # ---- low level ----\n    def eef(self):",
          "    # ---- policy-facing observation (identical for demos and rollouts) ----\n"
          "    def obs_dict(self):\n"
          "        return {'eef_pos': np.array(self.obs['robot0_eef_pos'], np.float32),\n"
          "                'gripper_qpos': np.array(self.obs['robot0_gripper_qpos'], np.float32),\n"
          "                'cube_pos': self.env.cube_pos(self.cur_i).astype(np.float32),\n"
          "                'target': np.asarray(self.cur_target, np.float32)}\n\n"
          "    def _step(self, a):\n"
          "        if self.rec is not None:\n"
          "            self.rec.append((self.obs_dict(), np.array(a, np.float32)))\n"
          "        self.obs, _, _, _ = self.env.step(a)\n"
          "        self.steps_used += 1\n\n"
          "    # ---- low level ----\n    def eef(self):")
S = S.replace("            self.obs, _, _, _ = self.env.step(a)\n            self.steps_used += 1\n",
              "            self._step(a)\n")
S = patch(S, "    def place(self, i, target_x, err_sd=0.0):\n        e = self.env\n        c = e.cube_pos(i)\n",
          "    def place(self, i, target_x, err_sd=0.0):\n        e = self.env\n"
          "        self.cur_i = i; self.cur_target = np.array([target_x, ROW_Y, CUBE_Z])\n"
          "        c = e.cube_pos(i)\n")
assert S.count("self._step(a)") >= 2, "recorder not wired into goto/push_axis"

nb = nbf.v4.new_notebook()
C = []

C.append(nbf.v4.new_markdown_cell(textwrap.dedent("""
# Restoring Organs for Long-Horizon Robot Chains — GPU gate

**Claim under test.** A robot skill chain is an analog pipeline: world-state error after skill *k*
drifts and nothing renormalises it, so success decays as *p*<sup>N</sup>. Inserting a
contact-based *restoring organ* (push the object into its physical references) snaps state onto a
discrete configuration each stage and should change the **exponent** of that decay, not the intercept.

**This notebook** (robosuite Panda + parallel gripper, MuJoCo):
1. collects expert demonstrations of a single-stage *place cube at target x*;
2. trains a **robomimic diffusion policy** on them (a genuinely learned placement policy);
3. runs the sequential line-up chain with that policy under three arms —
   `none` (bare chain), `noop` (matched actuation budget, touches nothing), `restore`
   (closed-finger plate push: seat against the rail, then push into the predecessor/stop; a cube that
   landed on top of the fence or out of the pushable zone is first pushed off / re-picked into it);
4. reports SR vs horizon, the fitted per-stage reliability *p*<sub>eff</sub> with bootstrap CIs,
   and the within-stage contraction factor.

**Acceptance criteria.** *p*<sub>eff</sub>(restore) > *p*<sub>eff</sub>(none) with non-overlapping 95% CIs;
noop ≈ none; within-stage contraction < 1 for restore only. Any other outcome is a negative result and
is reported as such.

**v3 note.** The first GPU run used a grasp-and-slide restore whose push crept at 0.2 mm/step (one jaw sat on the
rail), so it corrected at most ~1 cm per call and the run was NOT ACCEPTED. The v3 primitive was re-validated
locally across placement errors of 2 to 16 cm (residual < 6 mm) before this notebook was regenerated.

**Non-goals.** No VLA (won't fit a T4), no real robot, no irreversible tasks. The rail and stop are the
task's references; the predecessor cube is a general affordance.

**Runtime.** ~5 min demos, ~30–45 min training (T4), ~15 min evaluation. Every stage saves to Drive and
is skipped on rerun if its output exists. Pins: `mujoco==3.3.7` (robosuite 1.5.2 needs `>=3.3.0`; later releases removed `MjData.qM`, which its OSC
controller uses), `robosuite==1.5.2` installed `--no-deps` (its `mink` dep pins `numpy<2`, which has no 3.13 wheels), robomimic from GitHub with `diffusers==0.11.1`.
""").strip()))

C.append(nbf.v4.new_code_cell(textwrap.dedent("""
import os, subprocess, sys
os.environ["MUJOCO_GL"] = "egl"
os.environ["USE_FLAX"] = "0"   # Colab ships jax+flax; diffusers 0.11.1 flax schedulers use the removed jax.random.KeyArray
def pip(*args):
    r = subprocess.run([sys.executable, "-m", "pip", "install", *args], capture_output=True, text=True)
    tail = (r.stdout + r.stderr)[-2500:]
    print(tail)
    if r.returncode != 0:
        raise SystemExit(f"pip failed on {args} -- paste the output above")
# robosuite 1.5.2 requires mujoco>=3.3.0; 3.3.7 is the newest that still has MjData.qM (needed by its OSC controller).
# It also pulls mink==0.0.5, which pins numpy<2 -- and numpy 1.x has no Python 3.13 wheels. mink is only used for
# GR1 whole-body IK and robosuite guards its import, so install robosuite without deps + the runtime deps we need.
pip("numpy>=2", "mujoco==3.3.7", "h5py", "glfw", "pynput", "numba", "scipy", "Pillow", "opencv-python", "termcolor", "tqdm")
pip("--no-deps", "robosuite==1.5.2")
# PyPI robomimic is a 2023 sdist; install master without deps, then the deps we actually use.
# diffusers==0.11.1 + huggingface_hub==0.23.4 is robomimic's own pinned pair (EMAModel(model=...) API).
pip("--no-deps", "git+https://github.com/ARISE-Initiative/robomimic.git")
# diffusers 0.11.1 auto-imports transformers when present; Colab's transformers 5.x needs huggingface_hub>=1.5
# while diffusers 0.11.1 needs <=0.23. They cannot coexist and nothing here uses transformers, so remove it.
r = subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "transformers"], capture_output=True, text=True)
print(r.stdout[-200:] or r.stderr[-200:])
pip("diffusers==0.11.1", "huggingface_hub==0.23.4", "torchvision", "tensorboard", "tensorboardX",
    "imageio", "imageio-ffmpeg", "psutil", "matplotlib")
import mujoco, robosuite, robomimic, torch, diffusers, numpy, logging
logging.getLogger("robosuite_logs").setLevel(logging.ERROR)   # silence per-reset controller-config warnings
print("mujoco", mujoco.__version__, "| robosuite", robosuite.__version__, "| robomimic", robomimic.__version__,
      "| diffusers", diffusers.__version__, "| numpy", numpy.__version__, "| cuda", torch.cuda.is_available())
assert mujoco.__version__.startswith("3.3"), "wrong mujoco: robosuite's OSC needs MjData.qM (removed later)"
from robomimic.algo import diffusion_policy                      # fails fast if the diffusers pin is wrong
# early OSC probe: this is exactly what breaks on an incompatible mujoco
from robosuite.controllers import load_composite_controller_config
env = robosuite.make("Lift", robots="Panda", controller_configs=load_composite_controller_config(controller="BASIC", robot="Panda"),
                     has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False, control_freq=20, horizon=10)
env.reset(); env.step(env.action_spec[0] * 0); env.close()
print("robosuite OSC probe OK -- environment is good")
""").strip()))

C.append(nbf.v4.new_code_cell(textwrap.dedent("""
from google.colab import drive
drive.mount("/content/drive")
OUT = "/content/drive/MyDrive/restoring_organs"
os.makedirs(OUT, exist_ok=True)
import json, pathlib, time, numpy as np
def save(name, obj): json.dump(obj, open(f"{OUT}/{name}", "w"), indent=1)
def load(name):
    p = pathlib.Path(f"{OUT}/{name}"); return json.load(open(p)) if p.exists() else None
print("output dir:", OUT, "| existing:", [p.name for p in pathlib.Path(OUT).iterdir()])
""").strip()))

C.append(nbf.v4.new_code_cell("%%writefile rs_chain.py\n" + S))

C.append(nbf.v4.new_code_cell(textwrap.dedent("""
# ---- 1. expert demonstrations of the single-stage place -> robomimic HDF5 ----
import h5py, numpy as np, time, logging
logging.getLogger("robosuite_logs").setLevel(logging.ERROR)
from rs_chain import Chain, CUBE_Z, ROW_Y, STAGE_Y
DEMO = f"{OUT}/demos.hdf5"; N_DEMO = 150
if pathlib.Path(DEMO).exists():
    print("demos exist, skipping:", DEMO)
else:
    t0 = time.time(); rng = np.random.default_rng(0)
    with h5py.File(DEMO + ".tmp", "w") as f:
        g = f.create_group("data"); total = 0
        for d in range(N_DEMO):
            ch = Chain(n_cubes=8, seed=1000 + d) if d == 0 else ch
            ch.obs = ch.env.reset()
            spawn_x = float(rng.uniform(-0.25, 0.25)); target_x = float(rng.uniform(-0.24, 0.06))
            ch.env.set_cube(0, [spawn_x, STAGE_Y, CUBE_Z]); ch.settle(3)
            ch.rec = []; ch.place(0, target_x, err_sd=0.0); rec = ch.rec; ch.rec = None
            err = abs(ch.env.cube_pos(0)[0] - target_x)
            obs = {k: np.stack([o[k] for o, _ in rec]) for k in rec[0][0]}
            acts = np.stack([a for _, a in rec]); T = len(acts)
            dg = g.create_group(f"demo_{d}"); dg.attrs["num_samples"] = T
            dg.create_dataset("actions", data=acts)
            dg.create_dataset("rewards", data=np.zeros(T, np.float32))
            dg.create_dataset("dones", data=np.zeros(T, np.int64))
            for k, v in obs.items(): dg.create_dataset(f"obs/{k}", data=v)
            total += T
            if d % 25 == 0: print(f"  demo {d}: T={T} place err={err:.4f}  {(time.time()-t0)/60:.1f} min")
        g.attrs["total"] = total
        g.attrs["env_args"] = json.dumps({"env_name": "Lineup", "type": 1, "env_kwargs": {}})
    os.replace(DEMO + ".tmp", DEMO)
    print(f"wrote {N_DEMO} demos, {total} samples, {(time.time()-t0)/60:.1f} min")
""").strip()))

C.append(nbf.v4.new_code_cell(textwrap.dedent("""
# ---- 2. train a robomimic diffusion policy on the demos ----
import robomimic, glob, re
import shutil
EPOCHS = 200
def _ckpts():
    c = sorted(glob.glob(f"{OUT}/dp_run/**/models/*.pth", recursive=True), key=lambda p: int(re.search(r"epoch_(\d+)", p).group(1)))   # numeric: plain sort puts epoch_50 after epoch_200
    return c, (int(re.search(r"epoch_(\d+)", c[-1]).group(1)) if c else 0)
CKPT, LAST_EP = _ckpts()
# Require the FINAL epoch, not merely some checkpoint: an interrupted run leaves epoch 50/100 on Drive and a
# "skip if any checkpoint exists" test then silently evaluates a half-trained policy.
if LAST_EP >= EPOCHS:
    print("training complete, skipping:", CKPT[-1])
else:
    if CKPT:
        print(f"only epoch {LAST_EP} of {EPOCHS} on Drive: training was interrupted, restarting clean")
        shutil.rmtree(f"{OUT}/dp_run", ignore_errors=True)
    tmpl = pathlib.Path(robomimic.__file__).parent / "exps/templates/diffusion_policy.json"
    cfg = json.load(open(tmpl))
    cfg["experiment"]["name"] = "dp_lineup"
    cfg["experiment"]["rollout"]["enabled"] = False          # we evaluate in our own chain
    cfg["experiment"]["save"]["enabled"] = True
    cfg["experiment"]["save"]["every_n_epochs"] = 50
    cfg["experiment"]["validate"] = False
    cfg["experiment"]["logging"]["log_wandb"] = False
    cfg["train"]["data"] = DEMO
    cfg["train"]["output_dir"] = f"{OUT}/dp_run"
    cfg["train"]["num_epochs"] = EPOCHS
    cfg["train"]["batch_size"] = 256
    cfg["train"]["hdf5_cache_mode"] = "all"
    cfg["observation"]["modalities"]["obs"]["low_dim"] = ["eef_pos", "gripper_qpos", "cube_pos", "target"]
    cfg["observation"]["modalities"]["obs"]["rgb"] = []
    cfgp = f"{OUT}/dp_cfg.json"; json.dump(cfg, open(cfgp, "w"), indent=1)
    t0 = time.time()
    r = subprocess.run([sys.executable, "-m", "robomimic.scripts.train", "--config", cfgp],
                       capture_output=True, text=True, input="y\\n")   # answers robomimic's "overwrite? (y/n)" if the run dir exists
    print(r.stdout[-3000:]); print(r.stderr[-2000:])
    CKPT, LAST_EP = _ckpts()
    print(f"trained in {(time.time()-t0)/60:.1f} min; reached epoch {LAST_EP}; checkpoints:", CKPT[-3:])
assert LAST_EP >= EPOCHS, f"training stopped at epoch {LAST_EP}, need {EPOCHS} -- read the log above"
""").strip()))

C.append(nbf.v4.new_code_cell(textwrap.dedent("""
# ---- 3. the chain with the LEARNED policy, three arms ----
import torch
from robomimic.utils.file_utils import policy_from_checkpoint
from rs_chain import Chain, ROW_Y, CUBE_Z, PITCH, X0
import re, glob
CKPT = sorted(glob.glob(f"{OUT}/dp_run/**/models/*.pth", recursive=True), key=lambda p: int(re.search(r"epoch_(\d+)", p).group(1)))
print("using checkpoint:", CKPT[-1])
ck = torch.load(CKPT[-1], map_location="cpu", weights_only=False)
_cfg = json.loads(ck["config"]) if isinstance(ck["config"], str) else ck["config"]
# DDIM-10 sampling at inference: ~7x faster than the trained DDPM-100 schedule (measured 174s -> 25s per
# stage on CPU), and standard practice for diffusion policies. Weights are unaffected.
_cfg["algo"]["ddpm"]["enabled"] = False
_cfg["algo"]["ddim"]["enabled"] = True
_cfg["algo"]["ddim"]["num_inference_timesteps"] = 10
ck["config"] = json.dumps(_cfg)
policy, _ = policy_from_checkpoint(ckpt_dict=ck, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), verbose=False)
FRAME_STACK = int(_cfg["train"].get("frame_stack", 1))
T_POLICY = 240   # ~ expert episode length
print("frame_stack =", FRAME_STACK, "| sampler = DDIM-10")

HOME = None; HOME_QUAT = None   # eef pose at reset: every demo started here, so every policy rollout must too
def learned_place(chain, i, target_x, home=True):
    # robomimic's RolloutPolicy expects obs already stacked to (frame_stack, D);
    # its env wrapper normally does that, so keep the history here.
    from collections import deque
    if home and HOME is not None:                      # precondition of the policy, applied identically in all arms
        p = chain.eef(); chain.goto([p[0], p[1], HOME[2]], -1, 15)
        chain.home_pose(HOME, HOME_QUAT, -1)           # position AND orientation: pushes leave the wrist 5-10 deg off
    chain.cur_i = i; chain.cur_target = np.array([target_x, ROW_Y, CUBE_Z]); policy.start_episode()
    hist = deque([chain.obs_dict()] * FRAME_STACK, maxlen=FRAME_STACK)
    for _ in range(T_POLICY):
        ob = {k: np.stack([h[k] for h in hist]) for k in hist[0]} if FRAME_STACK > 1 else hist[-1]
        a = np.clip(np.asarray(policy(ob=ob)), -1, 1)
        chain._step(a)
        hist.append(chain.obs_dict())
    chain.settle(8)

# single-stage error of the learned policy (this sets the regime)
ch = Chain(n_cubes=8, seed=7); errs = []
ch.obs = ch.env.reset(); HOME = ch.eef().copy(); HOME_QUAT = np.array(ch.obs['robot0_eef_quat'], float).copy(); print('home eef:', np.round(HOME, 3), 'quat:', np.round(HOME_QUAT, 3))
for t in np.linspace(-0.22, 0.04, 10):
    ch.obs = ch.env.reset(); learned_place(ch, 0, float(t)); errs.append(abs(ch.env.cube_pos(0)[0] - t))
print(f"learned policy single-stage |error|: mean {np.mean(errs):.4f}  sd {np.std(errs):.4f}  max {np.max(errs):.4f}")
# start-state check: cube 1 after cube 0 was placed, with and without homing (the v3 run failed here)
for home in (False, True):
    e1 = []
    for s in range(3):
        ch.obs = ch.env.reset(); learned_place(ch, 0, X0, home=False); learned_place(ch, 1, X0 + PITCH, home=home)
        c = ch.env.cube_pos(1); e1.append((abs(c[0] - (X0 + PITCH)), abs(c[1] - ROW_Y)))
    print(f"stage-2 placement {'WITH' if home else 'WITHOUT'} homing: |x err| " + " ".join(f"{a:.3f}" for a, _ in e1) + "  |y err| " + " ".join(f"{b:.3f}" for _, b in e1))

N_LIST = [2, 4, 8]; SEEDS = 10; TOL = 0.008
# v7: full-pose wrist rotations inside the restore (the far-reach leak) and a re-measured noop budget (420 steps/stage);
# the v6 none rows stay valid, noop and restore rerun
TAG = {"none": "ddim10_v6", "noop": "ddim10_v7", "restore": "ddim10_v7"}
rows = [r for r in (load("chain_learned.json") or []) if r.get("sampler") == TAG.get(r.get("mode"))]
done = {(r["mode"], r["n"], r["seed"]) for r in rows}
t0 = time.time()
for mode in ("none", "noop", "restore"):
    for n in N_LIST:
        for s in range(SEEDS):
            if (mode, n, s) in done: continue
            ch.reseed(500 + 17 * n + s)          # ONE env for the whole sweep (robosuite leaks envs)
            d = ch.run(n, mode=mode, place_fn=learned_place)
            rows.append(dict(mode=mode, n=n, seed=s, dev=d.tolist(), success=bool(np.all(d <= TOL)), steps=ch.steps_used, sampler=TAG[mode], placed=ch.placed,
                             final=ch.final, final_success=bool(all(v <= TOL for v in ch.final))))
            save("chain_learned.json", rows)
            print(f"{mode:<8} N={n} seed={s} devs={np.round(d,4)} ok={rows[-1]['success']} {(time.time()-t0)/60:.1f} min")
print("done")
""").strip()))

C.append(nbf.v4.new_code_cell(textwrap.dedent("""
# ---- 4. analysis: does the restoring organ change the exponent? ----
TAG = {"none": "ddim10_v6", "noop": "ddim10_v7", "restore": "ddim10_v7"}
rows = [r for r in load("chain_learned.json") if r.get("sampler") == TAG.get(r.get("mode"))]; MODES = ("none", "noop", "restore")
def peff(mode):
    srs = np.array([np.mean([r["success"] for r in rows if r["mode"] == mode and r["n"] == n]) for n in N_LIST]); ok = srs > 0
    return (float(np.exp(np.polyfit(np.array(N_LIST)[ok], np.log(srs[ok]), 1)[0])) if ok.sum() >= 2 else float("nan")), srs
def boot_peff(mode, B=1500, seed=0):
    rng = np.random.default_rng(seed); per = {n: np.array([r["success"] for r in rows if r["mode"] == mode and r["n"] == n], float) for n in N_LIST}; out = []
    for _ in range(B):
        v = np.array([rng.choice(per[n], len(per[n]), replace=True).mean() for n in N_LIST]); ok = v > 0
        if ok.sum() >= 2: out.append(np.exp(np.polyfit(np.array(N_LIST)[ok], np.log(v[ok]), 1)[0]))
    return np.percentile(out, [2.5, 97.5]) if out else (np.nan, np.nan)
print(f"{'mode':<8}" + "".join(f"  N={n}" for n in N_LIST) + "     p_eff [95% CI]      mean steps")
for m in MODES:
    pe, srs = peff(m); lo, hi = boot_peff(m); st = np.mean([r["steps"] / r["n"] for r in rows if r["mode"] == m])
    print(f"{m:<8}" + "".join(f" {v:5.2f}" for v in srs) + f"   {pe:.4f} [{lo:.4f},{hi:.4f}]   {st:7.0f}/stage")
print("\\nmean |deviation| along the longest chain:")
for m in MODES:
    d = np.array([r["dev"] for r in rows if r["mode"] == m and r["n"] == max(N_LIST)])
    print(f"  {m:<8}", " ".join(f"{v:.4f}" for v in d.mean(axis=0)))
pn, _ = peff("none"); pr, _ = peff("restore")
print(f"\\n50%-success horizon: none N={np.log(0.5)/np.log(pn):.1f}   restore N={np.log(0.5)/np.log(pr) if pr < 1 else float('inf'):.1f}")
# final-row metric (what an assembly is judged on): every cube within TOL at the END of the chain
print("\\nfinal-row success (all cubes within tol at the end):")
for m in MODES:
    fs = [np.mean([r.get("final_success", r["success"]) for r in rows if r["mode"] == m and r["n"] == n]) for n in N_LIST]
    print(f"  {m:<8} " + "  ".join(f"N={n}: {v:.2f}" for n, v in zip(N_LIST, fs)))
# Pre-registered test, made censoring-aware: an arm that reaches 0/10 at some N has no fitted p_eff, only an UPPER
# bound on its per-stage reliability (Clopper-Pearson on the observed successes, taken to the 1/N power). The
# treatment arm gets a LOWER bound the same way. Accept iff the bounds are disjoint in the claimed direction.
from scipy.stats import beta as _beta
def _cp(k, n, upper, a=0.05):
    if upper: return 1.0 if k == n else float(_beta.ppf(1 - a, k + 1, n - k))
    return 0.0 if k == 0 else float(_beta.ppf(a, k, n - k + 1))
def bounds(mode):
    ks = {n: (int(sum(r["success"] for r in rows if r["mode"] == mode and r["n"] == n)), sum(1 for r in rows if r["mode"] == mode and r["n"] == n)) for n in N_LIST}
    hi = min(_cp(k, m, True) ** (1 / n) for n, (k, m) in ks.items())
    lo = max(_cp(k, m, False) ** (1 / n) for n, (k, m) in ks.items())
    return lo, hi
print("\\nper-stage reliability bounds (95% one-sided, tightest over N):")
for m in MODES:
    lo, hi = bounds(m); print(f"  {m:<8} >= {lo:.3f}   <= {hi:.3f}")
r_lo, _ = bounds("restore"); _, n_hi = bounds("none"); _, o_hi = bounds("noop")
verdict = r_lo > n_hi and r_lo > o_hi
print("\\nACCEPT: restore's per-stage reliability lower bound exceeds the upper bounds of both controls" if verdict
      else "NOT ACCEPTED (report as negative)")
save("summary.json", {m: dict(p_eff=peff(m)[0], sr=peff(m)[1].tolist()) for m in MODES})
""").strip()))

C.append(nbf.v4.new_code_cell(textwrap.dedent("""
# ---- 5. save & shut down: everything is already on Drive; this verifies, writes a manifest, and flushes ----
import glob, re
ck = sorted(glob.glob(f"{OUT}/dp_run/**/models/*.pth", recursive=True), key=lambda q: int(re.search(r"epoch_(\d+)", q).group(1)))
rows = load("chain_learned.json") or []
manifest = {"saved_at": time.strftime("%Y-%m-%d %H:%M"), "demos": pathlib.Path(f"{OUT}/demos.hdf5").exists(),
            "checkpoints": [pathlib.Path(c).name for c in ck], "use_checkpoint": ck[-1] if ck else None,
            "chains_done": len([r for r in rows if r.get("sampler") == "ddim10"]), "summary": load("summary.json")}
save("manifest.json", manifest)
for f in sorted(pathlib.Path(OUT).rglob("*")):
    if f.is_file(): print(f"{f.stat().st_size/1e6:8.1f} MB  {f.relative_to(OUT)}")
print(); print("manifest:", json.dumps(manifest, indent=1))
from google.colab import drive
drive.flush_and_unmount(); print("Drive flushed -- safe to terminate. Re-run cells 1-3 next time; 4-6 resume from Drive.")
""").strip()))

nb["cells"] = C
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
nb.metadata["accelerator"] = "GPU"
out = pathlib.Path(__file__).with_name("restoring_colab.ipynb")
nbf.write(nb, str(out))

# syntax-check every code cell (strip the %%writefile magic first)
for i, c in enumerate(C):
    if c.cell_type != "code":
        continue
    src = c.source.split("\n", 1)[1] if c.source.startswith("%%writefile") else c.source
    ast.parse(src)
print(f"wrote {out.name}: {len(C)} cells, all code cells parse")
