"""Diagnose the ~40 mm (one cell pitch) jumps in N=12 packing. Per stage, logs the landing offset (dx, dy), the fences
the restore chose (from rs_geom.find_fences) against the fences the target cell should have, re-picks, and the offset
after the stage. A fence error near +/-40 mm means the restore seated the cube against the wrong neighbour.

  python rs_pack_diag.py ERR_SD SEEDS MODES TAG
"""
import json
import sys
import time

import numpy as np
import rs_geom
from rs_chain import CUBE_H, PITCH, RAIL_FACE, STOP_FACE, Chain
from rs_pack import COLS, cell, nominal
from rs_pack_guard import TOL, GuardPack16Chain

_orig_find = rs_geom.find_fences
_orig_repick = Chain.repick


class DiagChain(GuardPack16Chain):
    def restore_cell(self, i, k):
        log = self.stage_log[-1] if self.stage_log else None

        def find(env, cube_index, nominal_xy, half, z_center, **kw):
            f = _orig_find(env, cube_index, nominal_xy, half, z_center, **kw)
            if log is not None and cube_index == log["cube"]:
                log["fences"].append(dict(x=f["x"], y=f["y"]))
            return f

        def repick(chain, j, drop_xy, jaws="y"):
            if log is not None:
                log["repicks"] += 1
            return _orig_repick(chain, j, drop_xy, jaws)

        rs_geom.find_fences = find
        Chain.repick = repick
        try:
            return super().restore_cell(i, k)
        finally:
            rs_geom.find_fences = _orig_find
            Chain.repick = _orig_repick

    def run(self, n, mode="none", err_sd=0.0, place_fn=None):
        self.stage_log = []
        orig_place = self.place_cell

        def place_and_open_log(i, k, e):
            orig_place(i, k, e)
            r, c = cell(k)
            nx, ny = nominal(k)
            p = self.env.cube_pos(i)
            self.stage_log.append(dict(stage=len(self.stage_log), cube=i, cell=k, row=r, col=c,
                                       landing=[float(p[0] - nx), float(p[1] - ny)], fences=[], repicks=0,
                                       expected_fx=float(STOP_FACE if c == 0 else nominal(k - 1)[0] + CUBE_H),
                                       expected_fy=float(RAIL_FACE if r == 0 else nominal(k - COLS)[1] + CUBE_H)))

        self.place_cell = place_and_open_log
        try:
            d = super().run(n, mode=mode, err_sd=err_sd, place_fn=place_fn)
        finally:
            del self.place_cell
        for s in self.stage_log:
            nx, ny = nominal(s["cell"])
            p = self.env.cube_pos(s["cube"])
            s["final"] = [float(p[0] - nx), float(p[1] - ny)]
            s["after_stage_dev"] = float(d[s["stage"]])
            for f in s["fences"]:
                f["x_err"] = None if f["x"] is None else float(f["x"][0] - s["expected_fx"])
                f["y_err"] = None if f["y"] is None else float(f["y"][0] - s["expected_fy"])
        return d


if __name__ == "__main__":
    err_sd = float(sys.argv[1]); seeds = int(sys.argv[2]); modes = sys.argv[3].split(","); tag = sys.argv[4]
    ch = DiagChain(seed=0); out = []; t0 = time.time()
    for s in range(seeds):
        for mode in modes:
            ch.reseed(5000 + s)
            d = ch.run(int(__import__("os").environ.get("NCELLS", "12")), mode=mode, err_sd=err_sd)
            out.append(dict(mode=mode, seed=s, err_sd=err_sd, dev=d.tolist(), final=ch.final, stages=ch.stage_log))
            json.dump(out, open(f"_pack_diag_{tag}.json", "w"), default=str)
            print(f"{mode:<16} err {err_sd} s={s} stages<=8mm {int((d <= TOL).sum())}/12 final<=8mm "
                  f"{int((np.array(ch.final) <= TOL).sum())}/12  {(time.time() - t0) / 60:.1f} min", flush=True)
            for st in ch.stage_log:
                if st["after_stage_dev"] > TOL or max(abs(v) for v in st["final"]) > TOL:
                    fx = [None if f["x_err"] is None else round(1000 * f["x_err"], 1) for f in st["fences"]]
                    fy = [None if f["y_err"] is None else round(1000 * f["y_err"], 1) for f in st["fences"]]
                    print(f"   stage {st['stage']:2d} cell r{st['row']}c{st['col']} landing "
                          f"({1000 * st['landing'][0]:+.0f},{1000 * st['landing'][1]:+.0f}) mm | fence x err {fx} "
                          f"y err {fy} | repicks {st['repicks']} | after stage {1000 * st['after_stage_dev']:.0f} mm "
                          f"| final ({1000 * st['final'][0]:+.0f},{1000 * st['final'][1]:+.0f}) mm", flush=True)
