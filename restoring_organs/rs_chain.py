"""3-D line-up chain in robosuite (Panda + parallel gripper, MuJoCo).

Same experiment as the planar rig, with real grasping:
  stage k: pick cube k from staging, place it at a target x measured relative to
           where cube k-1 ACTUALLY landed (error propagates down the row).
  arms:    none / noop (matched duration, touches nothing) / restore
  restore: closed fingers as a plate; re-pick if out of zone, seat against the rail, push into the fence
           in -x to contact with cube k-1 (cube 0 against a fixed stop).
Requires robosuite==1.5.2 with mujoco==3.2.7 (3.12 removed MjData.qM).
"""
import numpy as np
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import array_to_string
from robosuite.controllers import load_composite_controller_config

CUBE_H = 0.020                     # half-size -> contact pitch 0.040
PITCH = 2 * CUBE_H
TABLE_Z = 0.80
CUBE_Z = TABLE_Z + CUBE_H
RAIL_Y, RAIL_HY, RAIL_HZ = -0.12, 0.008, 0.015   # 3 cm rail: a pushed cube tilts over anything lower (measured 8-14 mm spring-back); recovery grasps use x-jaws so no finger crosses it
RAIL_FACE = RAIL_Y + RAIL_HY
ROW_Y = RAIL_FACE + CUBE_H          # cube y when seated on the rail
STOP_X, STOP_HX = -0.26, 0.008
STOP_FACE = STOP_X + STOP_HX
X0 = STOP_FACE + CUBE_H             # cube 0 when flush against the stop
STAGE_Y = 0.20
HOVER = 0.10


class LineupEnv(ManipulationEnv):
    def __init__(self, n_cubes=8, seed=0):
        self.n_cubes = n_cubes
        self.table_full_size = (0.8, 0.8, 0.05)
        self.table_friction = (1.0, 5e-3, 1e-4)
        self.table_offset = np.array((0, 0, TABLE_Z))
        cfg = load_composite_controller_config(controller="BASIC", robot="Panda")
        super().__init__(robots="Panda", controller_configs=cfg, base_types="default",
                         gripper_types="default", initialization_noise=None,
                         use_camera_obs=False, has_renderer=False, has_offscreen_renderer=False,
                         control_freq=20, horizon=100000, ignore_done=True, hard_reset=False,
                         seed=seed)

    def reward(self, action=None):
        return 0.0

    def _load_model(self):
        super()._load_model()
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)
        arena = TableArena(table_full_size=self.table_full_size, table_friction=self.table_friction,
                           table_offset=self.table_offset)
        arena.set_origin([0, 0, 0])
        self.cubes = [BoxObject(name=f"cube{i}", size=[CUBE_H] * 3, rgba=[0.85, 0.45, 0.2, 1])
                      for i in range(self.n_cubes)]
        self.rail = BoxObject(name="rail", size=[0.32, RAIL_HY, RAIL_HZ], rgba=[0.45, 0.5, 0.55, 1], joints=None)
        self.rail.get_obj().set("pos", array_to_string([0.0, RAIL_Y, TABLE_Z + RAIL_HZ]))
        self.stop = BoxObject(name="stop", size=[STOP_HX, 0.05, CUBE_H], rgba=[0.45, 0.5, 0.55, 1], joints=None)
        self.stop.get_obj().set("pos", array_to_string([STOP_X, ROW_Y, CUBE_Z]))
        self.model = ManipulationTask(mujoco_arena=arena, mujoco_robots=[self.robots[0].robot_model],
                                      mujoco_objects=self.cubes + [self.rail, self.stop])

    def _setup_references(self):
        super()._setup_references()
        self.cube_bid = [self.sim.model.body_name2id(c.root_body) for c in self.cubes]

    def _reset_internal(self):
        super()._reset_internal()
        for i, c in enumerate(self.cubes):
            self.set_cube(i, [-0.22 + 0.06 * i, STAGE_Y, CUBE_Z])

    # ---- helpers ----
    def set_cube(self, i, pos):
        self.sim.data.set_joint_qpos(self.cubes[i].joints[0],
                                     np.concatenate([np.asarray(pos, float), [1, 0, 0, 0]]))
        self.sim.forward()

    def cube_pos(self, i):
        return np.array(self.sim.data.body_xpos[self.cube_bid[i]])


class Chain:
    """Expert, restoring organ, control, and chain runner over a LineupEnv."""

    def __init__(self, n_cubes=8, seed=0):
        self.env = LineupEnv(n_cubes=n_cubes, seed=seed)
        self.rng = np.random.default_rng(seed)
        self.obs = self.env.reset()
        self.steps_used = 0
        self.q_home = np.asarray(self.obs["robot0_eef_quat"], float).copy()   # straight-down wrist: the reference for every rotation

    def reseed(self, seed):
        """Reuse ONE env across runs: robosuite/MuJoCo do not free envs, and
        constructing one per chain exhausts resources after ~18. Only the error
        RNG needs reseeding; the task has no other randomness."""
        self.rng = np.random.default_rng(seed); self.steps_used = 0
        return self

    # ---- low level ----
    def eef(self):
        return np.array(self.obs["robot0_eef_pos"])

    def goto(self, target, grip, steps, gain=8.0):
        target = np.asarray(target, float)
        for _ in range(steps):
            a = np.zeros(self.env.action_dim)
            a[:3] = np.clip(gain * (target - self.eef()), -1, 1)
            a[-1] = grip
            self.obs, _, _, _ = self.env.step(a)
            self.steps_used += 1
        return self.eef()

    def push_axis(self, axis, target, hold, grip, max_steps, amp=0.7, tol=0.0015, cube=None, c_stall_n=5, z_bias=0.0, cube_target=None):
        """Saturated push along one axis (the OSC goal sits a fixed distance ahead,
        so the commanded force is ~kp*amp*0.05 regardless of remaining error), other
        axes held by P-control. Stops on arrival or when contact stalls progress.
        A P-controller cannot do this: its force vanishes as it approaches."""
        hold = np.asarray(hold, float)
        amp = amp * getattr(self, "amp_scale", 1.0)          # retry-at-lower-force hook (1.0 unless a subclass sets it)
        last = self.eef()[axis]; stall = 0
        c0 = self.env.cube_pos(cube)[axis] if cube is not None else None; c_last = c0; c_stall = 0
        # optional collateral guard (off unless a subclass sets self.watch): snapshot the finished parts and abort the
        # push the moment any of them moves, instead of shoving a seated block. Also a sign-crossing arrival test, since
        # a cube driven at full amplitude can jump the +/-1.5 mm window in one control step.
        watch = [j for j in (getattr(self, "watch", None) or []) if j != cube]
        w0 = {j: self.env.cube_pos(j).copy() for j in watch}
        push_sign = np.sign(target - self.eef()[axis]) or 1.0
        for _ in range(max_steps):
            e = self.eef()
            d = target - e[axis]
            if abs(d) < tol:
                break
            if cube_target is not None and abs(self.env.cube_pos(cube)[axis] - cube_target) < 0.0015:   # the cube is seated: stop before overdriving it over the fence
                break
            if cube_target is not None and push_sign * (self.env.cube_pos(cube)[axis] - cube_target) >= -0.0015:
                break                # crossed the target (a full-amplitude push can jump the +/-1.5 mm window above)
            if watch and max(np.abs(self.env.cube_pos(j)[:2] - w0[j][:2]).max() for j in watch) > 0.0015:
                self.collateral_aborts = getattr(self, "collateral_aborts", 0) + 1
                break
            a = np.zeros(self.env.action_dim)
            a[:3] = np.clip(8.0 * (hold - e), -1, 1)
            a[axis] = amp * np.sign(d)
            a[2] = float(np.clip(a[2] + z_bias, -1, 1))
            a[-1] = grip
            self.obs, _, _, _ = self.env.step(a)
            self.steps_used += 1
            stall = stall + 1 if abs(e[axis] - last) < 1e-4 else 0
            last = e[axis]
            if stall >= 14:         # arrested by contact (loose: slow friction creep is not a stall)
                break
            if cube is not None:    # the pushed cube itself has been arrested by a fence: stop before the plate climbs it
                cn = self.env.cube_pos(cube)[axis]
                c_stall = c_stall + 1 if abs(cn - c_last) < 2e-4 else 0; c_last = cn
                if abs(cn - c0) > 0.002 and c_stall >= c_stall_n:
                    break
        for _ in range(6):          # relax: hold position with no push so the contact unloads before the plate leaves
            e = self.eef(); a = np.zeros(self.env.action_dim); a[:3] = np.clip(8.0 * (hold - e), -1, 1); a[axis] = 0.0; a[-1] = grip
            self.obs, _, _, _ = self.env.step(a); self.steps_used += 1
        return self.eef()

    def settle(self, steps=10):
        a = np.zeros(self.env.action_dim); a[-1] = -1
        for _ in range(steps):
            self.obs, _, _, _ = self.env.step(a)
        return self.env

    # ---- placement policy (expert with error; swapped for a learned policy in the notebook) ----
    def place(self, i, target_x, err_sd=0.0):
        e = self.env
        c = e.cube_pos(i)
        self.goto([c[0], c[1], CUBE_Z + HOVER], -1, 40)
        self.goto([c[0], c[1], CUBE_Z + 0.003], -1, 35)
        self.goto([c[0], c[1], CUBE_Z + 0.003], +1, 12)          # close
        self.goto([c[0], c[1], CUBE_Z + HOVER], +1, 30)          # lift
        tx = target_x + err_sd * self.rng.normal()
        self.goto([tx, ROW_Y, CUBE_Z + HOVER], +1, 45)
        self.goto([tx, ROW_Y, CUBE_Z + 0.004], +1, 35)
        self.goto([tx, ROW_Y, CUBE_Z + 0.004], -1, 10)           # open
        self.goto([tx, ROW_Y, CUBE_Z + HOVER], -1, 25)
        self.settle(8)

    # ---- restoring organ ----
    def _seat_x(self, i, ref_face, pad_x, overdrive, z):
        c = self.env.cube_pos(i)
        self.goto([c[0] + CUBE_H + 0.04, c[1], CUBE_Z + HOVER], +1, 22)
        self.goto([c[0] + CUBE_H + 0.04, c[1], z], +1, 25)
        x_over = ref_face + PITCH + pad_x - overdrive
        self.push_axis(0, x_over, [x_over, c[1], z], +1, 80, amp=0.8)
        p = self.eef()
        self.goto([p[0] + 0.05, p[1], CUBE_Z + HOVER], +1, 20)

    def _seat_y(self, i, pad_y, overdrive, z):
        c = self.env.cube_pos(i)
        self.goto([c[0], c[1] + CUBE_H + 0.04, CUBE_Z + HOVER], +1, 22)
        self.goto([c[0], c[1] + CUBE_H + 0.04, z], +1, 25)
        y_over = RAIL_FACE + PITCH + pad_y - overdrive
        self.push_axis(1, y_over, [c[0], y_over, z], +1, 80, amp=0.8)
        p = self.eef()
        self.goto([p[0], p[1] + 0.05, CUBE_Z + HOVER], +1, 20)

    def push_diag(self, target_xy, z, grip, max_steps, amp=0.85, tol=0.002):
        """Saturated push toward a corner along both x and y at once. Stops when
        both axes have arrived or both are arrested by contact."""
        tx, ty = target_xy
        last = self.eef()[:2].copy(); stall = 0
        for _ in range(max_steps):
            e = self.eef()
            dx, dy = tx - e[0], ty - e[1]
            if abs(dx) < tol and abs(dy) < tol:
                break
            a = np.zeros(self.env.action_dim)
            a[0] = amp * np.sign(dx) if abs(dx) >= tol else 0.0
            a[1] = amp * np.sign(dy) if abs(dy) >= tol else 0.0
            a[2] = np.clip(8.0 * (z - e[2]), -1, 1)
            a[-1] = grip
            self.obs, _, _, _ = self.env.step(a)
            self.steps_used += 1
            stall = stall + 1 if np.linalg.norm(e[:2] - last) < 2e-4 else 0
            last = e[:2].copy()
            if stall >= 8:
                break
        return self.eef()

    # ---- restoring organ v3: plate push into the corner, with re-pick recovery ----
    PAD_X = 0.010     # eef site -> closed-finger plate face along x (measured at contact onset)
    PAD_Y = 0.021     # same along y (the fingers' outer faces)
    PAD = PAD_X
    PUSH_Z = CUBE_Z           # plate site height: fingertips on the table
    Z_BIAS = -0.15            # mild downward action during pushes (the plate otherwise climbs the cube face; a hard press shoves cubes over the rail)

    def yaw(self):
        q = self.obs["robot0_eef_quat"]                       # xyzw
        x, y, z, w = q
        return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))

    def home_pose(self, pos, quat, grip=-1, max_steps=60, tol_pos=0.004, tol_rot=0.02):
        """Drive the eef to a full pose (position + orientation). Used to return the robot to the exact
        state the policy was trained from: after a restore the wrist is 5-10 deg off in yaw AND roll,
        which the policy cannot see and which makes its next grasp miss."""
        pos = np.asarray(pos, float); qt = np.asarray(quat, float)
        for _ in range(max_steps):
            e = self.eef(); q = np.asarray(self.obs["robot0_eef_quat"], float)          # xyzw
            # rotation that takes q to qt, as a world-frame axis-angle
            x, y, z, w = q; qi = np.array([-x, -y, -z, w])                             # inverse
            ax, ay, az, aw = qt; bx, by, bz, bw = qi
            d = np.array([aw*bx + ax*bw + ay*bz - az*by, aw*by - ax*bz + ay*bw + az*bx,
                          aw*bz + ax*by - ay*bx + az*bw, aw*bw - ax*bx - ay*by - az*bz])   # qt * q^-1
            if d[3] < 0: d = -d
            ang = 2 * np.arccos(np.clip(d[3], -1, 1)); s = np.sqrt(max(1 - d[3]**2, 1e-12))
            rot = d[:3] / s * ang if ang > 1e-6 else np.zeros(3)
            perr = pos - e
            if np.linalg.norm(perr) < tol_pos and ang < tol_rot:
                break
            a = np.zeros(self.env.action_dim)
            a[:3] = np.clip(8.0 * perr, -1, 1); a[3:6] = np.clip(3.0 * rot, -1, 1); a[-1] = grip
            self.obs, _, _, _ = self.env.step(a)
            self.steps_used += 1
        return self.eef()

    @staticmethod
    def _qmul(a, b):
        ax, ay, az, aw = a; bx, by, bz, bw = b        # xyzw
        return np.array([aw*bx + ax*bw + ay*bz - az*by, aw*by - ax*bz + ay*bw + az*bx,
                         aw*bz + ax*by - ay*bx + az*bw, aw*bw - ax*bx - ay*by - az*bz])

    def quat(self):
        return np.asarray(self.obs["robot0_eef_quat"], float).copy()

    def yawed(self, q, angle):
        """q rotated by `angle` about the world z axis."""
        return self._qmul(np.array([0.0, 0.0, np.sin(angle / 2), np.cos(angle / 2)]), q)

    def set_yaw(self, target, hold, grip, max_steps=45):
        hold = np.asarray(hold, float)
        for _ in range(max_steps):
            err = (target - self.yaw() + np.pi) % (2 * np.pi) - np.pi
            if abs(err) < 0.03:
                break
            a = np.zeros(self.env.action_dim)
            a[:3] = np.clip(8.0 * (hold - self.eef()), -1, 1)
            a[5] = float(np.clip(3.0 * err, -1, 1)); a[-1] = grip
            self.obs, _, _, _ = self.env.step(a)
            self.steps_used += 1

    def cube_upright(self, i):
        m = np.array(self.env.sim.data.body_xmat[self.env.cube_bid[i]]).reshape(3, 3)
        return bool(np.max(np.abs(m[2])) > 0.95)          # some face is up (any yaw)

    def seated_ref(self, upto):
        """x of the rightmost cube among 0..upto-1 that is physically seated in the row (on the table,
        within 2.5 cm of the rail line); None if the row is empty -> the stop is the fence."""
        xs = [float(self.env.cube_pos(j)[0]) for j in range(upto)
              if abs(self.env.cube_pos(j)[1] - ROW_Y) < 0.025 and self.env.cube_pos(j)[2] < CUBE_Z + 0.008]
        return max(xs) if xs else None

    def in_zone(self, i, ref_face, fy=RAIL_FACE):
        """Pushable zone: on the table, in front of the y fence, not on/over the x fence, roughly upright, near its row."""
        c = self.env.cube_pos(i)
        return bool(c[2] < CUBE_Z + 0.008 and fy + CUBE_H - 0.006 < c[1] < fy + CUBE_H + 0.06   # near the row, not still on the staging row
                    and c[0] > ref_face + CUBE_H - 0.004 and self.cube_upright(i))

    def cube_yaw_mod90(self, i):
        m = np.array(self.env.sim.data.body_xmat[self.env.cube_bid[i]]).reshape(3, 3)
        # yaw of whichever horizontal face-normal is closest to +x, wrapped to (-45, 45] degrees
        ax = [m[:, j] for j in range(3) if abs(m[2, j]) < 0.5]     # horizontal body axes
        if not ax:                                                 # balanced on a corner: no axis is near-horizontal
            ax = [m[:, int(np.argmin(np.abs(m[2, :])))]]           # take the least vertical one
        best = max(ax, key=lambda v: abs(v[0]))
        y = np.arctan2(best[1], best[0])
        return float((y + np.pi / 4) % (np.pi / 2) - np.pi / 4)

    def push_off_top(self, i, max_steps=70):
        """Cube resting on top of the fence/predecessor: plate-push it in +x (the open side)
        until it drops onto the table. Plate rides 1 cm above the surface it sits on."""
        c = self.env.cube_pos(i); z = c[2] + 0.010
        ax = c[0] - CUBE_H - self.PAD - 0.02
        self.goto([ax, c[1], CUBE_Z + HOVER], +1, 20)
        self.goto([ax, c[1], z], +1, 22)
        for _ in range(max_steps):
            e = self.eef()
            a = np.zeros(self.env.action_dim)
            a[0] = 0.6; a[1] = float(np.clip(8.0 * (c[1] - e[1]), -1, 1)); a[2] = float(np.clip(8.0 * (z - e[2]), -1, 1)); a[-1] = +1
            self.obs, _, _, _ = self.env.step(a)
            self.steps_used += 1
            if self.env.cube_pos(i)[2] < CUBE_Z + 0.008:
                break
        p = self.eef()
        self.goto([p[0], p[1], CUBE_Z + HOVER], -1, 18)
        self.settle(10)

    def repick(self, i, drop_xy, jaws="y"):
        """Recovery for a cube on the table but out of zone (behind the rail, beside the
        predecessor): grasp with the jaws along x, aligned to the cube's yaw, and drop it
        inside the pushable zone."""
        c = self.env.cube_pos(i); q0 = self.q_home
        hov = [c[0], c[1], CUBE_Z + HOVER]
        self.goto(hov, -1, 35)                                   # far reaches (staging row) track slowly: give the approach time
        if jaws == "x":                                     # beside the fence: jaws along x, aligned to the cube yaw
            self.home_pose(hov, self.yawed(q0, np.pi / 2 + self.cube_yaw_mod90(i)), -1)
        c = self.env.cube_pos(i)
        # grasp height: the normal expert height (tips at the table) unless a y-jaw finger would cross the rail;
        # on the staging row there is no rail, and a raised grasp only pinches the cube's top edges and slips
        near_rail = jaws == "y" and c[1] < ROW_Y + 0.06
        zg = max(c[2] + 0.008, TABLE_Z + 0.038) if near_rail else c[2] + 0.008
        self.goto([c[0], c[1], zg], -1, 40)
        c = self.env.cube_pos(i); self.goto([c[0], c[1], zg], -1, 12)   # re-centre on the cube before closing (an off-centre grasp slips in transit)
        self.goto([c[0], c[1], zg], +1, 12)
        self.goto([c[0], c[1], CUBE_Z + HOVER], +1, 30)
        self.goto([drop_xy[0], drop_xy[1], CUBE_Z + HOVER], +1, 45)
        self.goto([drop_xy[0], drop_xy[1], CUBE_Z + 0.008], +1, 35)
        self.goto([drop_xy[0], drop_xy[1], CUBE_Z + 0.008], -1, 10)
        hov = [drop_xy[0], drop_xy[1], CUBE_Z + HOVER]
        self.goto(hov, -1, 20)
        if jaws == "x":
            self.home_pose(hov, q0, -1)
        self.settle(5)

    def push_x_pressed(self, target, y_hold, z, max_steps, amp=0.8, press=0.25, tol=0.0015):
        """Saturated push along -x while pressing gently toward the rail, so the rail keeps
        the cube square as it slides. Stops on arrival or contact stall."""
        last = self.eef()[0]; stall = 0
        for _ in range(max_steps):
            e = self.eef(); d = target - e[0]
            if abs(d) < tol:
                break
            a = np.zeros(self.env.action_dim)
            a[0] = amp * np.sign(d)
            a[1] = -press
            a[2] = float(np.clip(8.0 * (z - e[2]), -1, 1)); a[-1] = +1
            self.obs, _, _, _ = self.env.step(a)
            self.steps_used += 1
            stall = stall + 1 if abs(e[0] - last) < 1e-4 else 0
            last = e[0]
            if stall >= 14:
                break
        return self.eef()

    def restore(self, i, prev_x, overdrive=0.004, nominal_x=None, nominal_y=None):
        """Restoring organ, 3-D. The closed fingers form a 2 cm plate. Phase 0: if the cube
        is outside the pushable zone (on the fence, behind the rail, tipped), re-pick it into
        the zone. Phase 1: seat it against the rail (plate on its +y face). Phase 2: push it
        along the rail into the fence (stop or predecessor), pressing toward the rail so the
        rail squares it. Every contact is saturated, not P-controlled, and terminates on a
        stall. (The earlier grasp-and-slide version put one jaw on the rail and crept at
        0.2 mm/step: it could correct at most ~1 cm per call.)"""
        e = self.env
        # the fence comes from the scene geometry, not from task bookkeeping: the nearest pushable face in -x
        # at the nominal slot (stop, or whichever cube is seated to the left). Legacy callers without a nominal
        # fall back to the predecessor-based reference.
        ny = ROW_Y if nominal_y is None else nominal_y
        if nominal_x is not None:
            from rs_geom import find_fences
            f = find_fences(e, i, (nominal_x, ny), CUBE_H, CUBE_Z)
            ref_face = f["x"][0] if f["x"] is not None else STOP_FACE
            fy = f["y"][0] if f["y"] is not None else RAIL_FACE
        else:
            ref_face = STOP_FACE if prev_x is None else prev_x + CUBE_H; fy = RAIL_FACE
        seat_y = fy + CUBE_H                                     # canonical y: against the y fence
        # fence sanity check (diagnosed 2026-09-14): when the expected neighbour is missing or displaced, find_fences
        # returns the next face one cell further back (error exactly -40 mm) and the push seats the part into the hole.
        # A fence more than half a pitch short of the nominal cell is not the cell's fence: stop at the nominal instead.
        if nominal_y is not None and seat_y < nominal_y - 0.5 * PITCH:
            seat_y = nominal_y
            fy = seat_y - CUBE_H                                 # in_zone must use the corrected fence too
            self.fence_rejects = getattr(self, "fence_rejects", 0) + 1
        if nominal_x is not None and ref_face + CUBE_H < nominal_x - 0.5 * PITCH:
            ref_face = nominal_x - CUBE_H
            self.fence_rejects = getattr(self, "fence_rejects", 0) + 1
        if not self.in_zone(i, ref_face, fy):
            if e.cube_pos(i)[2] > CUBE_Z + 0.025:                 # sitting on top of another part
                # Lift it off rather than shoving it sideways. A shove only works when the free direction is
                # known (a 1-D row); in a 2-D block the open side depends on which cells are filled, and a fixed
                # +x shove rakes the finished work (4.4 cm of disturbance measured). A lift is direction-agnostic.
                self.repick(i, [ref_face + CUBE_H + 0.05, seat_y + 0.03], jaws="x")
            if not self.in_zone(i, ref_face, fy):                  # on the table but out of zone
                c = e.cube_pos(i)
                # staging row (far from the rail): y-jaws, since x-jaws collide with the next staged cube 2 cm away.
                # near the row: x-jaws, so the 3 cm rail is never under a finger.
                self.repick(i, [ref_face + CUBE_H + 0.05, seat_y + 0.03], jaws="y" if c[1] > seat_y + 0.06 else "x")
        c = e.cube_pos(i); z = self.PUSH_Z
        # phase 1: seat against the rail. The finger BACKS are wedge-shaped (a push with them lands high on the
        # cube, tilts it 17 deg into the rail and it springs back 7 mm on release), so rotate the jaws along x and
        # push with the flat finger SIDES, exactly the surface the x push uses.
        q0 = self.q_home; hov = [c[0], c[1] + CUBE_H + self.PAD_X + 0.025, CUBE_Z + HOVER]   # rotate relative to the STRAIGHT wrist, not the current (possibly bent) one
        self.goto(hov, +1, 20); self.home_pose(hov, self.yawed(q0, np.pi / 2), +1)   # full-pose: yaw-only control leaves pitch/roll errors at far reach
        self.goto(hov, +1, 30)                                   # the rotation displaces the site by ~3 cm: re-centre before descending
        ay = hov[1]
        cx = c[0] + getattr(self, "y_push_dx", 0.0)             # optional plate offset away from a finished neighbour column
        self.goto([cx, ay, z], +1, 30)
        y_t = seat_y + CUBE_H + self.PAD_X - 0.012                # eef backstop 1.2 cm past the seat; the cube-arrival test ends the push
        self.push_axis(1, y_t, [cx, y_t, z], +1, 60, amp=0.6, cube=i, z_bias=self.Z_BIAS, cube_target=seat_y + 0.003)
        if abs(self.cube_yaw_mod90(i)) > np.radians(5):
            p = self.eef(); self.goto([p[0], p[1] + 0.03, z], +1, 10)
            self.push_axis(1, y_t, [cx, y_t, z], +1, 40, amp=0.6, cube=i, c_stall_n=12, z_bias=self.Z_BIAS, cube_target=seat_y + 0.003)
        p = self.eef()
        self.goto([p[0], p[1] + 0.03, z], +1, 10)             # retreat BACK first at push height (a diagonal retreat drags the cube 1 cm)
        hov = [p[0], p[1] + 0.03, CUBE_Z + HOVER]
        self.goto(hov, +1, 15); self.home_pose(hov, q0, +1)                  # straight wrist before the x push
        # phase 2: push along the rail into the fence
        c = e.cube_pos(i); ax = c[0] + CUBE_H + self.PAD_X + 0.025
        self.goto([ax, c[1], CUBE_Z + HOVER], +1, 20)
        self.goto([ax, c[1], z], +1, 22)
        # target the nominal slot measured from the STOP (not the predecessor): the push seats the cube against
        # its predecessor and keeps pushing until the whole row is compacted against the stop, so a predecessor's
        # residual error is removed instead of inherited. Terminates on a genuine stall of the pushed cube.
        slot = ref_face + CUBE_H if nominal_x is None else min(nominal_x, ref_face + CUBE_H)
        x_t = slot + CUBE_H + self.PAD_X - 0.012                  # eef backstop; the cube-arrival test ends the push
        # y is P-held at the SEAT (a few mm past the cube centre): the sliding cube gets a gentle pull onto the rail
        # same mild hold-down as the rail push: a harder press (-0.35) loads the fingertips on the table and the long-row compaction stalls
        self.push_axis(0, x_t, [x_t, seat_y, z], +1, 130, amp=1.0, cube=i, c_stall_n=10, z_bias=self.Z_BIAS, cube_target=slot + 0.003)
        p = self.eef()
        self.goto([p[0] + 0.03, p[1], z], +1, 8)
        self.goto([p[0] + 0.03, p[1], CUBE_Z + HOVER], -1, 18)
        self.settle(8)

    def noop(self, steps=420):   # matched to the v7 restore's measured extra budget (~416 steps/stage without recovery)
        p = self.eef()
        q = steps // 4
        self.goto([p[0], p[1], CUBE_Z + HOVER + 0.05], -1, q)
        self.goto([p[0] + 0.04, p[1], CUBE_Z + HOVER + 0.05], -1, q)
        self.goto([p[0] - 0.04, p[1], CUBE_Z + HOVER + 0.06], -1, q)
        self.goto([p[0], p[1], CUBE_Z + HOVER + 0.05], -1, steps - 3 * q)
        self.settle(8)

    # ---- chain ----
    def run(self, n, mode="none", err_sd=0.0, place_fn=None):
        """place_fn(chain, i, target_x) may replace the expert (learned policy)."""
        self.obs = self.env.reset()
        devs, prev = [], None
        self.placed = []                      # cube pose right after placement, before restore/noop (diagnostics)
        for k in range(n):
            nominal = X0 + k * PITCH
            target = X0 if prev is None else prev + PITCH
            if place_fn is None:
                self.place(k, target, err_sd)
            else:
                place_fn(self, k, target)
            self.placed.append([round(float(v), 4) for v in self.env.cube_pos(k)])
            if mode == "restore":
                self.restore(k, prev, nominal_x=nominal)
            elif mode == "noop":
                self.noop()
            # settle clear before measuring, every mode
            p = self.eef(); self.goto([p[0], p[1], CUBE_Z + HOVER + 0.05], -1, 15); self.settle(15)
            prev = self.seated_ref(k + 1)
            devs.append(abs(float(self.env.cube_pos(k)[0]) - nominal))
        self.final = [round(abs(float(self.env.cube_pos(j)[0]) - (X0 + j * PITCH)), 4) for j in range(n)]   # final-row metric
        return np.array(devs)
