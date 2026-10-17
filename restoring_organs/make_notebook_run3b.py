"""Generate restoring_run_3b_n12_fix.ipynb: the N=12 packing rerun with the diagnosed fixes, same policy and seeds as run 3.

Fixes (both validated locally with the scripted placer, N=12):
  1. fence sanity check in rs_chain.restore: a fence more than half a pitch short of the nominal cell is rejected
     (find_fences returned the face one cell further back when a neighbour was missing: -40 mm seating jumps);
  2. restore_reseat: after each restore, re-seat up to two finished cubes the stage pushed out of tolerance
     (the row-1 end cube was knocked from 3.7 to 13 mm by the row-2 restores beside it).
Arms: restore_v2 (fix 1 only) and restore_reseat (fixes 1 + 2), N in {2, 4, 8, 12}, 6 seeds, tag pack3x4_v2.
Demos and the dp_pack3x4 checkpoint from run 3 are reused (those cells skip when the files are complete), and the
chain and sampler seeds match run 3, so the last cell compares against run 3's none / noop / restore rows pairwise.
"""
import pathlib, shutil, textwrap
import nbformat as nbf
import make_notebook_run3 as R3
from nb_common import HERE, sub, writefile, finish, DRIVE, POLICY, SWEEP, ANALYSIS, RECURSION, SHUTDOWN

C = []
C.append(nbf.v4.new_markdown_cell(textwrap.dedent("""
# Restoring organs, run 3b: N=12 packing with the diagnosed fixes (tag `pack3x4_v2`)

Run 3 collapsed at N=12. Local diagnosis found two causes:
1. **Wrong fence.** When a neighbour was missing or displaced, the fence search returned the face one cell further
   back and the push seated the cube into the hole (errors of exactly 40 mm, one pitch). The restore now rejects any
   fence more than half a pitch short of the target cell and stops at the nominal position.
2. **Knocked-out finished cubes.** Row-2 restores pushed the row-1 end cube from 3.7 mm to 13 mm and nothing re-seated
   it. `restore_reseat` re-seats up to two finished cubes that a stage pushed out of tolerance.

Local scripted check, N=12, 5 mm placement error: final block correct in 3/3 chains, all 36 cubes within 8 mm,
collateral at most 7 mm.

**Arms:** `restore_v2` (fix 1) and `restore_reseat` (fixes 1 + 2), N in {2, 4, 8, 12}, 6 seeds. Same trained policy
(`dp_pack3x4`), demos and seeds as run 3, so the last cell compares pairwise against run 3's none / noop / restore.

**Runtime on a T4: about 2 h** (demo and train cells skip because run 3's files are complete). Resumable per chain.
""").strip()))
C.append(nbf.v4.new_code_cell(R3.B["install"]))
C.append(nbf.v4.new_code_cell(DRIVE))
C.append(nbf.v4.new_code_cell(writefile("rs_chain.py", R3.B["chain_src"])))
for name in ("rs_geom.py", "rs_pack.py", "rs_pack16.py", "rs_pack_guard.py"):
    C.append(nbf.v4.new_code_cell(writefile(name, (HERE / name).read_text(encoding="utf-8"))))
C.append(nbf.v4.new_code_cell(R3.DEMOS))
C.append(nbf.v4.new_code_cell(R3.TRAIN))
C.append(nbf.v4.new_code_cell(sub(
    POLICY,
    IMPORTS="from rs_chain import Chain, ROW_Y, CUBE_Z, PITCH, X0\nfrom rs_pack import nominal\n"
            "from rs_pack16 import Pack16Chain, N_CELLS\nfrom rs_pack_guard import GuardPack16Chain",
    CKDIR="dp_pack3x4",
    CHAIN_CTOR="GuardPack16Chain(seed=7)",
    NOMINAL="def landing_dev(placed, k):          # packing metric: planar distance to the nominal cell\n"
            "    nx, ny = nominal(k); return float(np.hypot(placed[0] - nx, placed[1] - ny))",
    PROBE="[(k, tuple(nominal(k))) for k in (0, 5, 11)]")))
C.append(nbf.v4.new_code_cell(sub(
    SWEEP,
    N_LIST="[2, 4, 8, 12]", SEEDS="6",
    ARMS='{"restore_v2": ("restore", True), "restore_reseat": ("restore_reseat", True)}',
    TAG="pack3x4_v2", RESULTS="chain_pack3x4_v2.json",
    EST='{"restore_v2": 20.0, "restore_reseat": 23.0}')))
C.append(nbf.v4.new_code_cell(sub(
    ANALYSIS,
    CONTRASTS='[("restore_reseat", "restore_v2", "re-seat pass vs fence fix alone")]')))
C.append(nbf.v4.new_code_cell(RECURSION))
C.append(nbf.v4.new_code_cell(r'''
# ---- paired comparison with run 3 (same policy, same chain and sampler seeds): final-block success ----
v1 = [r for r in (load("chain_pack3x4_v1.json") or []) if r.get("sampler") == "pack3x4_v1"]
v2 = [r for r in (load(RESULTS) or []) if r.get("sampler") == TAG]
print(f"{'arm':<16}" + "".join(f"  N={n:<5}" for n in N_LIST) + "   final-block success")
for src, arm in [(v1, "none"), (v1, "noop"), (v1, "restore"), (v2, "restore_v2"), (v2, "restore_reseat")]:
    cells = []
    for n in N_LIST:
        sel = [r for r in src if r["mode"] == arm and r["n"] == n]
        cells.append(f"{sum(r['final_success'] for r in sel)}/{len(sel)}" if sel else "-")
    print(f"{arm:<16}" + "".join(f"  {c:<7}" for c in cells))
'''.strip()))
C.append(nbf.v4.new_code_cell(sub(SHUTDOWN, MANIFEST="manifest_pack3x4_v2.json")))

out = HERE / "notebooks" / "restoring_run_3b_n12_fix.ipynb"
out.parent.mkdir(parents=True, exist_ok=True)
finish(C, out)
out.parent.mkdir(parents=True, exist_ok=True)
