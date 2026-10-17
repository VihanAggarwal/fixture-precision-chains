"""Generate restoring_runs_1_2.ipynb: line-up runs 1 and 2 in one paired sweep, on the CURRENT unified
rs_chain.py (geometry-derived restore), reusing the line-up checkpoint on Drive. No retraining."""
import pathlib, shutil, textwrap
import nbformat as nbf
from nb_common import HERE, base, sub, writefile, finish, DRIVE, POLICY, SWEEP, ANALYSIS, RECURSION, SHUTDOWN

B = base()
C = []
C.append(nbf.v4.new_markdown_cell(textwrap.dedent("""
# Restoring organs, line-up runs 1 and 2 (tag `lineup_v8`)

One paired sweep on the current unified code (the restore takes its fences from scene geometry, the same code
that passed gate 2). It reuses the line-up diffusion policy already on Drive (`dp_run`, epoch 200). **No training.**

**Arms** (same seeds and sampler seeds in every arm):

| arm | robot reset before each learned placement | after placement |
|---|---|---|
| `none_nohome` | none (the arm starts where the last stage left it) | nothing |
| `home` | full-pose homing, no contact with the part (robot-state-only reset) | nothing |
| `noop` | full-pose homing | motion-matched no-contact wander, 420 steps |
| `restore` | full-pose homing | geometry-derived restore |

N in {2, 4, 8}, 10 seeds, 120 chains.

**Contrasts.** robot reset = `home` vs `none_nohome`; extra motion = `noop` vs `home`; world restoration =
`restore` vs `noop` and `restore` vs `home`. The error-recursion cell fits e(k+1) = a e(k) + b per arm on the N=8 chains.

**Expected runtime on a T4: about 100 to 120 min** (install 4 min, policy load and probe 3 min, sweep about 100 min,
analysis 1 min). The previous 90-chain run took 83 min; this one adds 30 chains of the cheap `none_nohome` arm.

**How to run.** Runtime > Run all. Every chain is saved as it finishes (to `/content/results`, mirrored to Drive),
so after a disconnect just Run all again: finished chains are skipped. Nothing is ever deleted or renamed.

Cells: 1 install, 2 Drive (checks the mount is real), 3 and 4 write the modules, 5 load policy and probe,
6 sweep, 7 analysis, 8 error recursion and figure, 9 save and unmount.
""").strip()))
C.append(nbf.v4.new_code_cell(B["install"]))
C.append(nbf.v4.new_code_cell(DRIVE))
C.append(nbf.v4.new_code_cell(writefile("rs_chain.py", B["chain_src"])))
C.append(nbf.v4.new_code_cell(writefile("rs_geom.py", (HERE / "rs_geom.py").read_text(encoding="utf-8"))))
C.append(nbf.v4.new_code_cell(sub(
    POLICY,
    IMPORTS="from rs_chain import Chain, ROW_Y, CUBE_Z, PITCH, X0",
    CKDIR="dp_run",
    CHAIN_CTOR="Chain(n_cubes=8, seed=7)",
    NOMINAL="def landing_dev(placed, k):          # line-up metric: x distance to the stop-referenced slot\n"
            "    return abs(float(placed[0]) - (X0 + k * PITCH))",
    PROBE="[(0, float(t)) for t in np.linspace(-0.22, 0.04, 4)]")))
C.append(nbf.v4.new_code_cell(sub(
    SWEEP,
    N_LIST="[2, 4, 8]", SEEDS="10",
    ARMS='{"none_nohome": ("none", False), "home": ("none", True), "noop": ("noop", True), "restore": ("restore", True)}',
    TAG="lineup_v8", RESULTS="chain_lineup_v8.json",
    EST='{"none_nohome": 7.5, "home": 8.0, "noop": 14.0, "restore": 14.0}')))
C.append(nbf.v4.new_code_cell(sub(
    ANALYSIS,
    CONTRASTS='[("home", "none_nohome", "robot reset: full-pose homing vs no reset"),\n'
              '             ("noop", "home", "extra motion: no-contact wander vs homing only"),\n'
              '             ("restore", "noop", "world restoration vs matched motion"),\n'
              '             ("restore", "home", "world restoration vs robot-state reset only")]')))
C.append(nbf.v4.new_code_cell(RECURSION))
C.append(nbf.v4.new_code_cell(sub(SHUTDOWN, MANIFEST="manifest_lineup_v8.json")))

out = HERE / "notebooks" / "restoring_runs_1_2.ipynb"
out.parent.mkdir(parents=True, exist_ok=True)
finish(C, out)
out.parent.mkdir(parents=True, exist_ok=True)
