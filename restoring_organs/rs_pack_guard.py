"""Packing restore that does not rake the finished block (fixes for the N=12 far-row collapse), tested locally.

Run 3 (pack3x4_v1) failed at N=12: far-row restores loaded the finished block and shoved it (28-70 mm). Modes:
  restore                 unchanged restore, row-major fill (the failing baseline)
  restore_guard           collateral guard: abort a push when a placed cube moves > 1.5 mm, then re-seat offenders.
                          Measured 0/3: aborts leave cubes unseated.
  restore_retry           guard + retreat and retry at lower force (amplitude x1.0, x0.7, x0.5), re-seating disturbed
                          neighbours corner-first between attempts; a last unguarded gentle push if still unseated
  restore_colmajor        push order that never loads a finished row: fill column by column (row-major leaves the
                          4.2 cm plate overhanging the finished row below by ~1 mm during every x push), and offset the
                          y-push plate 1.5 mm away from the finished column
  restore_colmajor_retry  both
Every stage logs `collateral`: the largest displacement of an already-placed cube caused by that stage.
"""
import json
import sys
import time

import numpy as np
from rs_chain import CUBE_Z, HOVER
from rs_pack import COLS, nominal
from rs_pack16 import Pack16Chain

TOL = 0.008
MAX_RESEAT = 2
BLOCK_ROWS = 3


def fill_order(name, n):
    if name == "col":
        order = [r * COLS + c for c in range(COLS) for r in range(BLOCK_ROWS)]
    else:
        order = list(range(BLOCK_ROWS * COLS))
    return order[:n]


class GuardPack16Chain(Pack16Chain):
    watch = None
    cell_of = {}

    def dev(self, j):
        c = self.env.cube_pos(j)
        nx, ny = nominal(self.cell_of.get(j, j))
        return float(np.hypot(c[0] - nx, c[1] - ny))

    def _positions(self, k):
        return {j: self.env.cube_pos(j).copy() for j in range(k)}

    def _reseat(self, k, ok_before):
        offenders = sorted((j for j in ok_before if self.dev(j) > TOL), key=lambda j: self.cell_of[j])   # corner first
        for j in offenders[:MAX_RESEAT]:
            self.watch = [x for x in range(k + 1) if x != j]
            self.restore_cell(j, self.cell_of[j])
            self.watch = None
        return offenders[:MAX_RESEAT]

    def restore_and_reseat(self, k, cell):
        """Plain restore of cube k, then re-seat (normal force, no abort guard, no retry) any finished cube this stage
        pushed out of tolerance, corner-most first. Diagnosed 2026-09-14: row-2 restores knock the row-1 end cube
        (cell r1c3) from 3.7 to 10.6-13 mm, and nothing ever re-seated it. Returns (max collateral, re-seated cubes)."""
        before = self._positions(k)
        ok_before = [j for j in range(k) if self.dev(j) <= TOL]
        self.restore_cell(k, cell)
        # restore approaches from +x and +y, so a cube whose +x or +y cell is already filled cannot be re-seated
        # without landing the plate on that neighbour: skip it (reviewer finding, 2026-09-14)
        filled = {self.cell_of.get(j, j) for j in range(k + 1)}

        def clear(j):
            c = self.cell_of.get(j, j)
            right = c + 1 if (c + 1) // COLS == c // COLS else None
            return right not in filled and (c + COLS) not in filled
        offenders = sorted((j for j in ok_before if self.dev(j) > TOL and clear(j)),
                           key=lambda j: self.cell_of.get(j, j))[:MAX_RESEAT]
        for j in offenders:
            self.restore_cell(j, self.cell_of.get(j, j))
        moved = max((float(np.linalg.norm(self.env.cube_pos(j)[:2] - p[:2])) for j, p in before.items()), default=0.0)
        return moved, offenders

    def restore_guarded(self, k, retry=False):
        before, cell = self._positions(k), self.cell_of[k]
        ok_before = {j for j in range(k) if self.dev(j) <= TOL}
        reseated = []
        try:
            for scale in ((1.0, 0.7, 0.5) if retry else (1.0,)):
                a0 = self.collateral_aborts
                self.amp_scale, self.watch = scale, list(range(k))
                self.restore_cell(k, cell)
                self.watch = None
                aborted = self.collateral_aborts > a0
                self.amp_scale = 0.7
                reseated += self._reseat(k, ok_before)
                if not aborted and self.dev(k) <= TOL:
                    break
            if retry and self.dev(k) > TOL:            # last resort: gentle, unguarded
                self.amp_scale = 0.5
                self.restore_cell(k, cell)
        finally:
            self.watch, self.amp_scale = None, 1.0
        moved = max((float(np.linalg.norm(self.env.cube_pos(j)[:2] - p[:2])) for j, p in before.items()), default=0.0)
        return moved, reseated

    def run(self, n, mode="none", err_sd=0.0, place_fn=None):
        order = fill_order("col" if "colmajor" in mode else "row", n)
        self.cell_of = {k: order[k] for k in range(n)}
        self.y_push_dx = 0.0015 if "colmajor" in mode else 0.0
        self.obs = self.env.reset(); devs = []; self.placed = []; self.collateral = []; self.reseated = []; self.trace = []
        self.collateral_aborts = 0
        for k in range(n):
            cell = order[k]
            if place_fn is None:
                self.place_cell(k, cell, err_sd)
            else:
                place_fn(self, k, nominal(cell))
            self.placed.append([round(float(v), 4) for v in self.env.cube_pos(k)])
            if mode == "restore_reseat":
                moved, offenders = self.restore_and_reseat(k, cell)
                self.collateral.append(moved)
                self.reseated.append(offenders)
            elif mode in ("restore", "restore_colmajor"):
                before = self._positions(k)
                self.restore_cell(k, cell)
                self.collateral.append(max((float(np.linalg.norm(self.env.cube_pos(j)[:2] - p[:2]))
                                            for j, p in before.items()), default=0.0))
                self.reseated.append([])
            elif mode in ("restore_guard", "restore_retry", "restore_colmajor_retry"):
                moved, res = self.restore_guarded(k, retry="retry" in mode)
                self.collateral.append(moved)
                self.reseated.append(res)
            elif mode == "noop":
                self.noop()
            p = self.eef(); self.goto([p[0], p[1], CUBE_Z + HOVER + 0.05], -1, 15); self.settle(15)
            devs.append(self.dev(k))
            self.trace.append([round(self.dev(j), 4) for j in range(k + 1)])   # every placed cube's error after stage k
        self.final = [round(self.dev(j), 4) for j in range(n)]
        return np.array(devs)


if __name__ == "__main__":
    err_sd = float(sys.argv[1]) if len(sys.argv) > 1 else 0.012
    seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    modes = sys.argv[3].split(",") if len(sys.argv) > 3 else ["restore", "restore_retry", "restore_colmajor"]
    NC = int(sys.argv[4]) if len(sys.argv) > 4 else 12
    tag = sys.argv[5] if len(sys.argv) > 5 else "run"
    ch = GuardPack16Chain(seed=0); rows = []; t0 = time.time()
    for s in range(seeds):
        for mode in modes:
            ch.reseed(5000 + s); d = ch.run(NC, mode=mode, err_sd=err_sd)
            fin = np.array(ch.final)
            rows.append(dict(mode=mode, seed=s, err_sd=err_sd, dev=d.tolist(), final=ch.final, collateral=ch.collateral,
                             reseated=ch.reseated, aborts=ch.collateral_aborts, steps=ch.steps_used,
                             success=bool(np.all(d <= TOL)), final_success=bool(np.all(fin <= TOL))))
            print(f"{mode:<24} s={s} ok={rows[-1]['success']!s:<5} final_ok={rows[-1]['final_success']!s:<5} "
                  f"stages<=8mm {int((d <= TOL).sum())}/{NC} final<=8mm {int((fin <= TOL).sum())}/{NC} "
                  f"max collateral {1000 * max(ch.collateral):.1f} mm aborts {ch.collateral_aborts} "
                  f"steps/stage {ch.steps_used / NC:.0f}  {(time.time() - t0) / 60:.1f} min", flush=True)
            print("   dev (mm):       ", " ".join(f"{1000 * v:.1f}" for v in d), flush=True)
            print("   final (mm):     ", " ".join(f"{1000 * v:.1f}" for v in fin), flush=True)
            print("   collateral (mm):", " ".join(f"{1000 * v:.1f}" for v in ch.collateral), flush=True)
            json.dump(rows, open(f"_pack_fix_{tag}_{err_sd}.json", "w"))
    for mode in modes:
        v = [r for r in rows if r["mode"] == mode]
        print(f"{mode:<24} per-stage ok {sum(r['success'] for r in v)}/{len(v)}  final ok {sum(r['final_success'] for r in v)}/{len(v)}  "
              f"stages<=8mm {np.mean([np.mean(np.array(r['dev']) <= TOL) for r in v]):.2f}  "
              f"final cubes<=8mm {np.mean([np.mean(np.array(r['final']) <= TOL) for r in v]):.2f}  "
              f"max collateral {1000 * max(c for r in v for c in r['collateral']):.1f} mm")
