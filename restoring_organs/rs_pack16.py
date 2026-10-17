"""Run 3 rig: a 4 x 4 packing block (16 cells, pitch 0.04) so horizons up to N=16 stay inside the
workspace. Rows grow along +y from the rail, columns along +x from the stop (rs_pack.nominal with
COLS=4 already encodes this). Changes against the 2 x 4 rig, all fixtures only:
  * 16 cubes staged in TWO rows (y = STAGE_YS), 8 per row, x from STAGE_X0 in STAGE_DX steps;
  * the stop is lengthened in y so it spans all four rows (the 2 x 4 stop ends at y = 0.008, which
    leaves row 3 of column 0 with no x fence at all).
The restore is unchanged: PackChain.restore_cell -> Chain.restore with fences from rs_geom."""
import sys, time, json
import numpy as np
import rs_chain as _rc
from rs_chain import LineupEnv, CUBE_H, CUBE_Z, TABLE_Z, ROW_Y, PITCH, RAIL_Y, RAIL_HY, RAIL_HZ, STOP_X, STOP_HX
from rs_pack import PackChain, nominal, COLS
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import array_to_string
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv

ROWS = 4
N_CELLS = ROWS * COLS
STAGE_YS = (0.20, 0.30)          # 0.26 was tried: the open y-jaws land on the other staging row (grasp 18 mm high, far slots missed)
STAGE_X0, STAGE_DX, PER_ROW = -0.22, 0.06, 8
STOP_HY = 0.5 * ROWS * PITCH + 0.02                  # spans every row with 2 cm to spare
STOP_CY = ROW_Y + 0.5 * (ROWS - 1) * PITCH


def staging(i):
    return [STAGE_X0 + STAGE_DX * (i % PER_ROW), STAGE_YS[i // PER_ROW], CUBE_Z]


class Pack16Env(LineupEnv):
    def _load_model(self):
        ManipulationEnv._load_model(self)
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)
        arena = TableArena(table_full_size=self.table_full_size, table_friction=self.table_friction,
                           table_offset=self.table_offset)
        arena.set_origin([0, 0, 0])
        self.cubes = [BoxObject(name=f"cube{i}", size=[CUBE_H] * 3, rgba=[0.85, 0.45, 0.2, 1])
                      for i in range(self.n_cubes)]
        self.rail = BoxObject(name="rail", size=[0.32, RAIL_HY, RAIL_HZ], rgba=[0.45, 0.5, 0.55, 1], joints=None)
        self.rail.get_obj().set("pos", array_to_string([0.0, RAIL_Y, TABLE_Z + RAIL_HZ]))
        self.stop = BoxObject(name="stop", size=[STOP_HX, STOP_HY, CUBE_H], rgba=[0.45, 0.5, 0.55, 1], joints=None)
        self.stop.get_obj().set("pos", array_to_string([STOP_X, STOP_CY, CUBE_Z]))
        self.model = ManipulationTask(mujoco_arena=arena, mujoco_robots=[self.robots[0].robot_model],
                                      mujoco_objects=self.cubes + [self.rail, self.stop])

    def _reset_internal(self):
        super()._reset_internal()                        # one-row staging from LineupEnv (qpos only, no stepping)
        for i in range(self.n_cubes):
            self.set_cube(i, staging(i))


class Pack16Chain(PackChain):
    # Measured 2026-09-12: row-3 restores in the full 4 x 4 block rake the block (the 4.2 cm plate's x push loads the
    # row below; the block shifts and springs back, 2-35 mm). Holding the plate 4 mm further +y was tried and made
    # rows 2-3 worse, so it was reverted. Rows 0-2 (12 cells) are the validated horizon; see run 3 notebook.
    def __init__(self, n_cubes=N_CELLS, seed=0):
        # Chain.__init__ builds rs_chain.LineupEnv by name; swap in the 4 x 4 env for the duration of the call
        orig = _rc.LineupEnv
        _rc.LineupEnv = Pack16Env
        try:
            super().__init__(n_cubes=n_cubes, seed=seed)
        finally:
            _rc.LineupEnv = orig


if __name__ == "__main__":
    # local rig validation with the scripted placer: N=16, restore vs none
    err_sd = float(sys.argv[1]) if len(sys.argv) > 1 else 0.004
    seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    modes = sys.argv[3].split(",") if len(sys.argv) > 3 else ["restore", "none"]
    NC = int(sys.argv[4]) if len(sys.argv) > 4 else N_CELLS      # chain length; 12 fills rows 0-2 only
    TOL = 0.008; ch = Pack16Chain(seed=0); t0 = time.time(); rows = []
    ch.obs = ch.env.reset()
    print("staging ok:", all(abs(ch.env.cube_pos(i)[2] - CUBE_Z) < 0.003 for i in range(N_CELLS)), flush=True)
    for s in range(seeds):
        for mode in modes:
            ch.reseed(4000 + s); d = ch.run(NC, mode=mode, err_sd=err_sd)
            fin = np.array(ch.final)
            rows.append(dict(mode=mode, seed=s, dev=d.tolist(), final=ch.final, steps=ch.steps_used, placed=ch.placed))
            print(f"{mode:<8} s={s} steps/stage={ch.steps_used/N_CELLS:.0f} ok={bool(np.all(d <= TOL))} final_ok={bool(np.all(fin <= TOL))} "
                  f"{(time.time()-t0)/60:.1f} min", flush=True)
            print("   dev   (mm, row-major):", " ".join(f"{1000*v:.1f}" for v in d), flush=True)
            print("   final (mm) as 4x4 grid, row 0 = rail:", flush=True)
            for r in range(ROWS):
                print("     row", r, " ".join(f"{1000*v:6.1f}" for v in fin[r * COLS:(r + 1) * COLS]), flush=True)
            json.dump(rows, open(f"_pack16_rig_{err_sd}_{'_'.join(modes)}.json", "w"))
