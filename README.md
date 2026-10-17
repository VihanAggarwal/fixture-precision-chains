# Let the Fixture Carry Precision in Learned Chains

Code, data and paper source for the paper by V. Aggarwal and E. Kuo (CoRL 2026 Workshop on Sim-to-Real-to-Field,
Late-Breaking Track, under review).

When a learned placement policy is chained, each stage's target depends on where the previous part landed, so errors
compound and a policy that transfers twice as badly keeps about a quarter of its horizon. After every placement, we
push the part against a physical reference (a fence, a rail, or a seated neighbour) until contact fixes its pose. The
policy then only has to land within a centimetre-scale capture radius, and the millimetre-scale precision comes from
the fixture.

## Layout

| Path | Contents |
|---|---|
| `restoring_organs/rs_chain.py`, `rs_geom.py` | Panda line-up chain in robosuite (MuJoCo), the scripted placer, and the restore primitive |
| `restoring_organs/rs_pack*.py` | 3x4 packing task |
| `restoring_organs/make_notebook*.py`, `nb_common.py`, `notebooks/` | Colab notebooks for the learned diffusion-policy runs (generated) |
| `restoring_organs/srf_transfer.py` | Transfer sweeps: policy scatter and bias, contact-model shift, plate-stall logging |
| `restoring_organs/srf_analyze.py` | Tables from the sweeps, including the on-board seat check |
| `restoring_organs/srf_gap.jsonl`, `srf_contact.jsonl` | Raw sweep rows (one chain per line) |
| `restoring_organs/paper/` | Result summaries of the learned-policy runs and the workshop paper source (`srf_workshop/`) |

## Running

Needs robosuite 1.5.2 and MuJoCo 3.3.x. Run one MuJoCo process at a time on a small machine.

```
python restoring_organs/srf_transfer.py gap        # or: contact
python restoring_organs/srf_analyze.py
```

Rows are appended one chain at a time, so an interrupted sweep resumes where it stopped.
