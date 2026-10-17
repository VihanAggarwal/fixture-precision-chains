"""Generate calvin_drift_check.ipynb (restoring_organs/notebooks). Unverified install path, so every cell ends in an explicit check."""
import json
import pathlib

HERE = pathlib.Path(__file__).parent
DRIVE = "/content/drive/MyDrive/restoring_organs/calvin"
PY = "/content/venv_calvin/bin/python"


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip()}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip()}


cells = [
    md("""
# CALVIN drift check (restoring organs, external benchmark scoping)
**GPU: T4 or L4.** Rolls out the public pretrained **HULC (D->D)** policy on CALVIN five-subtask chains, logs every
object pose at each subtask boundary, and measures how often a failed subtask follows a displaced (non-target) block.
If drift-driven failures are common, porting the restore primitive to CALVIN is justified.

The install path has not been run end to end yet (HULC is a 2022 stack). Every cell ends in a check: if one fails,
stop and paste its output instead of continuing. Runtime once installed: roughly 1 to 2.5 h for 200 chains.
"""),
    code(f"""
from google.colab import drive
drive.mount('/content/drive')
import os; os.makedirs('{DRIVE}', exist_ok=True); print('ok')
"""),
    code(f"""%%bash
set -e
pip -q install uv
[ -x {PY} ] || uv venv -q /content/venv_calvin -p 3.10
cd /content
[ -d hulc ] || git clone -q --recurse-submodules https://github.com/lukashermann/hulc.git
uv pip install -q --python {PY} "setuptools<58" wheel "numpy<1.24" "pybullet==3.2.5"
cd /content/hulc/calvin_env/tacto && uv pip install -q --python {PY} -e . && cd /content/hulc/calvin_env && uv pip install -q --python {PY} -e .
cd /content/hulc && uv pip install -q --python {PY} -e .
{PY} -c "import pybullet, calvin_env, hulc; print('imports ok')"
"""),
    code(f"""%%bash
set -e
cd /content/hulc/checkpoints && [ -d HULC_D_D ] || sh download_model_weights.sh D
ls /content/hulc/checkpoints/HULC_D_D
cd /content/hulc/dataset && [ -d calvin_debug_dataset ] || sh download_data.sh debug
ls /content/hulc/dataset/calvin_debug_dataset/validation | head
"""),
    code(f"""# smoke: 3 chains; checks the logger sees scene_obs and chain boundaries
%env PYOPENGL_PLATFORM=egl
!cd /content/hulc && {PY} /content/calvin_drift.py run --log /content/smoke.jsonl --num_sequences 3 \\
    --dataset_path /content/hulc/dataset/calvin_debug_dataset --train_folder /content/hulc/checkpoints/HULC_D_D
!{PY} /content/calvin_drift.py analyse --log /content/smoke.jsonl
"""),
    code(f"""# full check: 200 chains, logged to Drive as they finish
!cd /content/hulc && {PY} /content/calvin_drift.py run --log {DRIVE}/drift_hulc_dd.jsonl --num_sequences 200 \\
    --dataset_path /content/hulc/dataset/calvin_debug_dataset --train_folder /content/hulc/checkpoints/HULC_D_D
!{PY} /content/calvin_drift.py analyse --log {DRIVE}/drift_hulc_dd.jsonl
"""),
]
cells.insert(2, code("%%writefile /content/calvin_drift.py\n" + (HERE / "calvin_drift.py").read_text(encoding="utf-8")))
for c in cells:
    assert "—" not in c["source"]
nb = {"cells": cells, "metadata": {"accelerator": "GPU", "kernelspec": {"display_name": "Python 3", "name": "python3"}},
      "nbformat": 4, "nbformat_minor": 0}
out = HERE / "notebooks" / "calvin_drift_check.ipynb"
out.parent.mkdir(parents=True, exist_ok=True)
json.dump(nb, open(out, "w", encoding="utf-8", newline="\n"), indent=1)
print("wrote", out)
