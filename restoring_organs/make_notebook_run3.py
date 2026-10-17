"""Generate restoring_run_3_horizon.ipynb: horizon sweep on a 3 x 4 packing block (N up to 12) on the 4 x 4 rig.
New demos (300, 2-D targets over all four rows), a new diffusion policy in dp_pack3x4 (final-epoch guard),
then none / noop / restore with full-pose homing, N in {2, 4, 8, 12}, 6 seeds, tag pack3x4_v1."""
import pathlib, shutil, textwrap
import nbformat as nbf
from nb_common import HERE, base, sub, writefile, finish, DRIVE, POLICY, SWEEP, ANALYSIS, RECURSION, SHUTDOWN

B = base()


def rep(s, old, new):
    assert old in s, old[:80]
    return s.replace(old, new)


DEMOS = r'''
# ---- demos: 300 single-stage placements, 2-D targets over the 3 x 4 block ----
import h5py, logging
logging.getLogger("robosuite_logs").setLevel(logging.ERROR)
from rs_chain import CUBE_Z, ROW_Y, PITCH, X0
from rs_pack import COLS
from rs_pack16 import Pack16Chain, staging, N_CELLS, ROWS
BLOCK_ROWS = 3    # rows 0-2 of the 4 x 4 rig: fourth-row restores rake the block (measured locally)
DEMO = f"{OUT}/demos_pack3x4.hdf5"; N_DEMO = 300
LOCAL_DEMO = f"{RES}/demos_pack3x4.hdf5"
X_RANGE = (X0 - 0.01, X0 + (COLS - 1) * PITCH + 0.02)          # block columns plus margin
Y_RANGE = (ROW_Y - 0.01, ROW_Y + (BLOCK_ROWS - 1) * PITCH + 0.02)    # the three block rows plus margin
def demos_valid(p):
    try:
        with h5py.File(p, "r") as f:
            return len(f["data"].keys()) == N_DEMO and int(f["data"].attrs["total"]) > 0
    except Exception:
        return False
if demos_valid(DEMO):
    print("demos complete on Drive, skipping:", DEMO)
else:
    t0 = time.time(); rng = np.random.default_rng(0); errs = []
    dch = Pack16Chain(seed=1000)
    with h5py.File(LOCAL_DEMO, "w") as f:                       # written locally, copied to Drive when complete
        g = f.create_group("data"); total = 0
        for d in range(N_DEMO):
            dch.obs = dch.env.reset()
            # spawn at one of the 16 real staging slots (both staging rows), swapping cube 0 with that slot's cube,
            # so the demo start states cover exactly what the chain will present
            j = int(rng.integers(N_CELLS)); jit = rng.uniform(-0.005, 0.005, 2); sp = staging(j)
            if j != 0:
                dch.env.set_cube(j, staging(0))
            dch.env.set_cube(0, [sp[0] + jit[0], sp[1] + jit[1], CUBE_Z]); dch.settle(3)
            tx = float(rng.uniform(*X_RANGE)); ty = float(rng.uniform(*Y_RANGE))
            dch.rec = []; dch.place_xy(0, tx, ty); rec = dch.rec; dch.rec = None
            c = dch.env.cube_pos(0); errs.append(float(np.hypot(c[0] - tx, c[1] - ty)))
            obs = {k: np.stack([o[k] for o, _ in rec]) for k in rec[0][0]}
            acts = np.stack([a for _, a in rec]); T = len(acts)
            dg = g.create_group(f"demo_{d}"); dg.attrs["num_samples"] = T
            dg.create_dataset("actions", data=acts)
            dg.create_dataset("rewards", data=np.zeros(T, np.float32))
            dg.create_dataset("dones", data=np.zeros(T, np.int64))
            for k, v in obs.items():
                dg.create_dataset(f"obs/{k}", data=v)
            total += T
            if d % 25 == 0:
                print(f"  demo {d}: slot {j} T={T} expert err={errs[-1]:.4f}  {(time.time()-t0)/60:.1f} min", flush=True)
        g.attrs["total"] = total
        g.attrs["env_args"] = json.dumps({"env_name": "Pack3x4", "type": 1, "env_kwargs": {}})
    assert drive_ok(), "Drive dropped during demo collection: re-mount (cell 2) and re-run this cell"
    shutil.copyfile(LOCAL_DEMO, DEMO)
    assert demos_valid(DEMO), "copied demo file failed validation"
    print(f"wrote {N_DEMO} demos, {total} samples, expert err mean {np.mean(errs):.4f} max {np.max(errs):.4f}, {(time.time()-t0)/60:.1f} min")
'''.strip()

TRAIN = B["train"]
TRAIN = rep(TRAIN, "dp_run", "dp_pack3x4")
TRAIN = rep(TRAIN, '"dp_lineup"', '"dp_pack3x4"')
TRAIN = rep(TRAIN, 'cfgp = f"{OUT}/dp_cfg.json"', 'cfgp = f"{OUT}/dp_cfg_pack3x4.json"')
# each robomimic checkpoint is ~1 GB (weights + EMA + optimizer); saving every 50 epochs puts ~6 GB on Drive for nothing,
# since an interrupted run restarts clean. Save only the final epoch (~3 GB with last.pth + last_bak.pth).
TRAIN = rep(TRAIN, 'cfg["experiment"]["save"]["every_n_epochs"] = 50',
            'cfg["experiment"]["save"]["every_n_epochs"] = EPOCHS    # final checkpoint only: each is ~1 GB')
TRAIN = rep(TRAIN, "        shutil.rmtree(", "        assert drive_ok(), 'Drive not mounted: refusing to touch dp_pack3x4'\n        shutil.rmtree(")
assert TRAIN.count("LAST_EP >= EPOCHS") == 2, "final-epoch guard missing"

C = []
C.append(nbf.v4.new_markdown_cell(textwrap.dedent("""
# Restoring organs, run 3: horizon sweep on a 3 x 4 packing block (tag `pack3x4_v1`)

The existing policies cannot go past N=8 (line-up targets leave the training range beyond x=0.06 and the far end is
reach-limited). This notebook packs a **3 x 4 block** (12 cells, pitch 4 cm; rows along +y from the rail, columns
along +x from the stop) on a rig built for 4 x 4: 16 cubes staged in two rows (y = 0.20 and 0.30; with a 2 cm gap the
open gripper lands on the other row) and a stop that spans four rows. The restore is the same geometry-derived code
as gate 2.

**Scope.** Local scripted validation of the full 4 x 4 block showed that restores in the fourth row rake the finished
block (the 4.2 cm plate's push loads the row below; displacements of 2 to 70 mm), so the horizon stops at N=12.

**Steps.** 300 expert demos with 2-D targets over the three rows, a new diffusion policy trained to 200 epochs in
`dp_pack3x4` (the train cell requires the final epoch and restarts clean otherwise), then the sweep:
arms `none`, `noop`, `restore` (all with full-pose homing before every learned placement), N in {2, 4, 8, 12},
6 seeds, 72 chains, 468 learned stages.

**Expected runtime on a T4: about 3 to 4 hours** (install 4 min, demos about 15 min, training about 50 min,
sweep about 2 to 2.5 h). Every chain is saved as it finishes, so a disconnect costs at most one chain:
Runtime > Run all again and finished work (demos, final checkpoint, chains) is skipped. Nothing is ever
deleted or renamed except an INCOMPLETE `dp_pack3x4` training folder, which the train cell restarts.

Cells: 1 install, 2 Drive (checks the mount is real), 3 to 6 write the modules, 7 demos, 8 train,
9 load policy and probe, 10 sweep, 11 analysis, 12 error recursion and figure, 13 save and unmount.
""").strip()))
C.append(nbf.v4.new_code_cell(B["install"]))
C.append(nbf.v4.new_code_cell(DRIVE))
C.append(nbf.v4.new_code_cell(writefile("rs_chain.py", B["chain_src"])))
for name in ("rs_geom.py", "rs_pack.py", "rs_pack16.py"):
    C.append(nbf.v4.new_code_cell(writefile(name, (HERE / name).read_text(encoding="utf-8"))))
C.append(nbf.v4.new_code_cell(DEMOS))
C.append(nbf.v4.new_code_cell(TRAIN))
C.append(nbf.v4.new_code_cell(sub(
    POLICY,
    IMPORTS="from rs_chain import Chain, ROW_Y, CUBE_Z, PITCH, X0\nfrom rs_pack import nominal\nfrom rs_pack16 import Pack16Chain, N_CELLS",
    CKDIR="dp_pack3x4",
    CHAIN_CTOR="Pack16Chain(seed=7)",
    NOMINAL="def landing_dev(placed, k):          # packing metric: planar distance to the nominal cell\n"
            "    nx, ny = nominal(k); return float(np.hypot(placed[0] - nx, placed[1] - ny))",
    PROBE="[(k, tuple(nominal(k))) for k in (0, 5, 11)]")))
C.append(nbf.v4.new_code_cell(sub(
    SWEEP,
    N_LIST="[2, 4, 8, 12]", SEEDS="6",
    ARMS='{"none": ("none", True), "noop": ("noop", True), "restore": ("restore", True)}',
    TAG="pack3x4_v1", RESULTS="chain_pack3x4_v1.json",
    EST='{"none": 11.0, "noop": 18.0, "restore": 20.0}')))
C.append(nbf.v4.new_code_cell(sub(
    ANALYSIS,
    CONTRASTS='[("restore", "none", "restoration vs bare chain"),\n'
              '             ("restore", "noop", "restoration vs matched motion"),\n'
              '             ("noop", "none", "extra motion vs bare chain")]')))
C.append(nbf.v4.new_code_cell(RECURSION))
C.append(nbf.v4.new_code_cell(sub(SHUTDOWN, MANIFEST="manifest_pack3x4_v1.json")))

out = HERE / "notebooks" / "restoring_run_3_horizon.ipynb"
out.parent.mkdir(parents=True, exist_ok=True)
finish(C, out)
out.parent.mkdir(parents=True, exist_ok=True)
