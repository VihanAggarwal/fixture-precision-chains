"""Gate 2 (second assembly topology): 2-D PACKING on the same rig.
Cubes fill a rows x columns block in row-major order. Cube (r, c) has TWO contact references:
in x the stop (c = 0) or the cube at (r, c-1); in y the rail (r = 0) or the cube at (r-1, c).
Errors therefore propagate along two axes and a failed cube blocks a whole row and column,
which is a different failure topology from the 1-D line-up. Same three arms, same restoring
organ (plate push into both references), same success rule (every cube within TOL of its
nominal cell, per stage and at the end)."""
import numpy as np, json, time, sys
from rs_chain import Chain, X0, ROW_Y, PITCH, CUBE_H, CUBE_Z, HOVER, STOP_FACE, RAIL_FACE, TABLE_Z
COLS = 4


def cell(k):
    return k // COLS, k % COLS                       # row, column


def nominal(k):
    r, c = cell(k)
    return X0 + c * PITCH, ROW_Y + r * PITCH


class PackChain(Chain):
    def seated_xy(self, j):
        p = self.env.cube_pos(j)
        return p if p[2] < CUBE_Z + 0.008 else None

    def refs(self, k):
        """physical reference faces for cube k: x-face of the left neighbour (or stop), y-face of the neighbour below (or rail)."""
        r, c = cell(k)
        fx = STOP_FACE
        if c > 0:
            p = self.seated_xy(k - 1)
            if p is not None and abs(p[1] - nominal(k - 1)[1]) < 0.025:
                fx = float(p[0]) + CUBE_H
        fy = RAIL_FACE
        if r > 0:
            p = self.seated_xy(k - COLS)
            if p is not None and abs(p[0] - nominal(k - COLS)[0]) < 0.025:
                fy = float(p[1]) + CUBE_H
        return fx, fy

    def place_cell(self, i, k, err_sd=0.0):
        e = self.env; c = e.cube_pos(i); nx, ny = nominal(k)
        self.goto([c[0], c[1], CUBE_Z + HOVER], -1, 40)
        self.goto([c[0], c[1], CUBE_Z + 0.003], -1, 35)
        self.goto([c[0], c[1], CUBE_Z + 0.003], +1, 12)
        self.goto([c[0], c[1], CUBE_Z + HOVER], +1, 30)
        tx = nx + err_sd * self.rng.normal(); ty = ny + err_sd * self.rng.normal()
        self.goto([tx, ty, CUBE_Z + HOVER], +1, 45)
        self.goto([tx, ty, CUBE_Z + 0.004], +1, 35)
        self.goto([tx, ty, CUBE_Z + 0.004], -1, 10)
        self.goto([tx, ty, CUBE_Z + HOVER], -1, 25)
        self.settle(8)

    def place_xy(self, i, tx, ty, err_sd=0.0):
        """single-stage expert placement at an explicit (x, y) target: used to record 2-D demos."""
        e = self.env; c = e.cube_pos(i)
        self.cur_i = i; self.cur_target = np.array([tx, ty, CUBE_Z])       # policy-facing target (recorder / rollout)
        self.goto([c[0], c[1], CUBE_Z + HOVER], -1, 40)
        self.goto([c[0], c[1], CUBE_Z + 0.003], -1, 35)
        self.goto([c[0], c[1], CUBE_Z + 0.003], +1, 12)
        self.goto([c[0], c[1], CUBE_Z + HOVER], +1, 30)
        tx = tx + err_sd * self.rng.normal(); ty = ty + err_sd * self.rng.normal()
        self.goto([tx, ty, CUBE_Z + HOVER], +1, 45)
        self.goto([tx, ty, CUBE_Z + 0.004], +1, 35)
        self.goto([tx, ty, CUBE_Z + 0.004], -1, 10)
        self.goto([tx, ty, CUBE_Z + HOVER], -1, 25)
        self.settle(8)

    def in_cell_zone(self, i, fx, fy):
        c = self.env.cube_pos(i)
        return bool(c[2] < CUBE_Z + 0.008 and c[0] > fx + CUBE_H - 0.004 and c[1] > fy + CUBE_H - 0.006
                    and c[1] < ROW_Y + 2 * PITCH + 0.06 and self.cube_upright(i))

    def restore_cell(self, i, k):
        """the line-up restore, unchanged: nominal cell in, fences from geometry."""
        nx, ny = nominal(k)
        self.restore(i, None, nominal_x=nx, nominal_y=ny)

    def dev(self, j):
        c = self.env.cube_pos(j); nx, ny = nominal(j)
        return float(np.hypot(c[0] - nx, c[1] - ny))

    def run(self, n, mode="none", err_sd=0.0, place_fn=None):
        self.obs = self.env.reset(); devs = []; self.placed = []
        for k in range(n):
            if place_fn is None:
                self.place_cell(k, k, err_sd)
            else:
                place_fn(self, k, nominal(k))
            self.placed.append([round(float(v), 4) for v in self.env.cube_pos(k)])
            if mode == "restore":
                self.restore_cell(k, k)
            elif mode == "noop":
                self.noop()
            p = self.eef(); self.goto([p[0], p[1], CUBE_Z + HOVER + 0.05], -1, 15); self.settle(15)
            devs.append(self.dev(k))
        self.final = [round(self.dev(j), 4) for j in range(n)]
        return np.array(devs)


if __name__ == "__main__":
    err_sd = float(sys.argv[1]) if len(sys.argv) > 1 else 0.02
    seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    TOL = 0.008; rows = []; ch = PackChain(n_cubes=8, seed=0); t0 = time.time()
    for mode in ("none", "noop", "restore"):
        for n in (2, 4, 8):
            for s in range(seeds):
                ch.reseed(900 + 37 * n + s); d = ch.run(n, mode=mode, err_sd=err_sd)
                rows.append(dict(mode=mode, n=n, seed=s, err_sd=err_sd, dev=d.tolist(), success=bool(np.all(d <= TOL)),
                                 final=ch.final, final_success=bool(all(v <= TOL for v in ch.final)),
                                 steps=ch.steps_used, placed=ch.placed))
                print(f"{mode:<8} N={n} s={s} devs={np.round(d,4)} ok={rows[-1]['success']} final_ok={rows[-1]['final_success']} {(time.time()-t0)/60:.1f}min", flush=True)
                json.dump(rows, open(f"rs_pack_{err_sd}.json", "w"))
    for mode in ("none", "noop", "restore"):
        srs = [np.mean([r["success"] for r in rows if r["mode"] == mode and r["n"] == n]) for n in (2, 4, 8)]
        fss = [np.mean([r["final_success"] for r in rows if r["mode"] == mode and r["n"] == n]) for n in (2, 4, 8)]
        print(f"{mode:<8} SR N=2/4/8 = {srs[0]:.2f}/{srs[1]:.2f}/{srs[2]:.2f}   final {fss[0]:.2f}/{fss[1]:.2f}/{fss[2]:.2f}")
