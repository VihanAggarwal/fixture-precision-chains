"""Restore synthesis from scene geometry.

A restoring organ needs a physical reference to push against. Instead of telling the
restore which object is the fence for each stage (the stop for cube 0, the predecessor
for cube k, the rail for row 0, the cube below for row r), this module reads the
collision geometry of the scene and finds, for a given nominal pose, the nearest static
or seated face in the -x and -y directions that the part could be pushed against.
Everything the executor needs (which faces, where) comes out of the MuJoCo model; the
task code only says where the part is supposed to end up.

The same function therefore serves the 1-D line-up (finds stop / predecessor and rail)
and the 2-D packing block (finds left neighbour / stop and lower neighbour / rail).
"""
import numpy as np
import mujoco

BOX = mujoco.mjtGeom.mjGEOM_BOX


def _candidate_geoms(env, exclude_body):
    """box geoms that can act as fences: static fixtures and free cubes, never the robot or the table."""
    m, d = env.sim.model._model, env.sim.data._data
    out = []
    for g in range(m.ngeom):
        if m.geom_type[g] != BOX:
            continue
        b = m.geom_bodyid[g]
        if b == exclude_body:
            continue
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        bname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if "robot" in bname or "gripper" in bname or "mount" in bname or "table" in name:
            continue
        R = d.geom_xmat[g].reshape(3, 3); ext = np.abs(R) @ m.geom_size[g][:3]     # world AABB half-extents
        c = d.geom_xpos[g].copy()
        out.append((name, c - ext, c + ext))
    return out


def find_fences(env, cube_index, nominal_xy, half, z_center, reach=0.30, tol_overlap=0.01):
    """Nearest pushable faces in -x and -y for a part of half-size `half` whose nominal centre is
    `nominal_xy` at height `z_center`. Returns dict(x=(face_x, name) | None, y=(face_y, name) | None).
    A face qualifies if it is at or beyond the part's own face in that direction (so the part can be
    pushed onto it), within `reach`, and overlaps the part's extent on the other two axes."""
    m = env.sim.model._model
    body = m.geom_bodyid[env.sim.model.geom_name2id(f"cube{cube_index}_g0")]
    nx, ny = nominal_xy; zlo, zhi = z_center - half, z_center + half
    best = {"x": None, "y": None}
    for name, lo, hi in _candidate_geoms(env, body):
        if hi[2] - zlo < tol_overlap or zhi - lo[2] < tol_overlap:   # no vertical overlap (a stacked cube, a low kerb)
            continue
        # -x fence: the geom's +x face, left of the part's -x face, overlapping the part in y
        if hi[0] <= nx - half + 0.004 and nx - half - hi[0] <= reach and min(hi[1], ny + half) - max(lo[1], ny - half) >= tol_overlap:
            if best["x"] is None or hi[0] > best["x"][0]:
                best["x"] = (float(hi[0]), name)
        # -y fence: the geom's +y face, below the part's -y face, overlapping the part in x
        if hi[1] <= ny - half + 0.004 and ny - half - hi[1] <= reach and min(hi[0], nx + half) - max(lo[0], nx - half) >= tol_overlap:
            if best["y"] is None or hi[1] > best["y"][0]:
                best["y"] = (float(hi[1]), name)
    return best

def space_free(env, exclude_body, centre, half, tol=0.004):
    """Is the box of half-size `half` centred at `centre` (x, y, z) clear of every other part and
    fixture? Used to decide where a mis-stacked part can be shoved to, instead of assuming a task
    layout: in an open row the far side is free, in a packed block it is not."""
    c = np.asarray(centre, float); lo_q, hi_q = c - half + tol, c + half - tol
    for _, lo, hi in _candidate_geoms(env, exclude_body):
        if np.all(hi > lo_q) and np.all(lo < hi_q):
            return False
    return True
