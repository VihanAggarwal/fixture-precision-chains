# Run 4 scoping: restore on an external long-horizon benchmark

## What any port needs

1. Privileged part poses, to decide when and where to restore, plus collision geometry for fences.
2. The contact primitive (saturated plate push, stall termination, re-pick fallback) in the benchmark's action space.
3. A "canonical" that does not undo the task. External benchmarks change state on purpose (open a drawer), so
   restore may only touch residuals: poses relative to fixtures.
4. The run 1-2 arms: bare, robot-state reset, matched motion, restore.

## CALVIN (PyBullet, Franka, 5-task language chains)

- **Helps:** public pretrained checkpoints (HULC, GR-1, MoDE, among others); chaining and the average-length
  metric are built in; `scene_obs` exposes block poses and drawer/slider joints, so no perception work.
- **Work:** legacy install (Python 3.8-era stack, EGL) 1-2 days; running one pretrained policy in its repo
  environment 2-4 days; re-implementing the push in PyBullet about 1 week (contact tuning does not transfer
  from MuJoCo); choosing residuals and fences (slider interior, drawer walls, table edge; blocks sit free)
  about 1 week; evaluation a few GPU-hours per arm.
- **Total: about 3-4 weeks.**
- **Main risk:** CALVIN failures may be mostly semantic or grasp failures, not geometric drift, leaving restore
  little to fix. First log `scene_obs` along existing HULC or GR-1 chains and measure how often a failed
  subtask follows a displaced block. That check takes about 2 days and decides whether to port.

## FurnitureBench (Isaac Gym, furniture assembly)

- **Fit:** good; parts are pushed into a fixed corner obstacle, literally a fence.
- **Cost:** high. Isaac Gym Preview is deprecated and needs Linux plus a recent NVIDIA driver, so Colab is
  unreliable. Public learned baselines are weak on long tasks, so bare chains may fail for reasons restore
  cannot touch. Insertion contact in PhysX needs its own tuning.
- **Estimate: 5-8 weeks** plus a Linux RTX machine.

## Verdict

CALVIN is clearly more feasible. FurnitureBench only makes sense with a dedicated Linux GPU box.

## A CALVIN-style evaluation in robosuite?

Partly, and cheaply. CALVIN's protocol is random sequences of different subtasks, continued while each succeeds,
scored by average completed length. In robosuite: chain block placement, a corner push, a stack and the
NutAssembly peg on one table; train one task-conditioned diffusion policy; report average length over about 500
sequences per arm. The rig, recorder, training and resumable sweep exist, so this is about 1-2 weeks.

It tests the mechanism across heterogeneous skills but gives no external validity (no language, no third-party
policy, self-built suite); reviewers will read it as an ablation.

**Recommendation:** robosuite multi-skill chain first, then the 2-day CALVIN drift check, and port to CALVIN only
if drift-driven failures are common.
