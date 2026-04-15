"""Modal app: Boltz-2 structure prediction for BioDesignBench Phase B.

This is the GPU-side companion to `eval_boltz.py`. The HF Space leaderboard
runs on cpu-basic, so it cannot host Boltz directly; instead it POSTs design
sequences to this Modal app, which spins up an A10G on demand, runs
`boltz predict`, and returns confidence metrics.

Setup (one-time, on a machine with `pip install modal`):

    modal token new                       # if you don't have a token yet
    cd biodesignbench-leaderboard
    modal deploy modal_boltz_app.py

After deploy Modal prints a URL like
    https://<workspace>--bdb-boltz-predict.modal.run

Add that URL plus a shared secret to the HF Space secrets:
    MODAL_BOLTZ_URL  = https://<workspace>--bdb-boltz-predict.modal.run
    MODAL_BOLTZ_TOKEN = <random 32-byte hex>

Cost: A10G is billed per-second, container auto-stops after
`container_idle_timeout` seconds. With one submission per month and
~76 tasks * ~30s = ~38min GPU per submission, expected spend is
well within Modal's free tier.
"""

from __future__ import annotations

import os

import modal

APP_NAME = "bdb-boltz"
ENDPOINT_LABEL = "bdb-boltz-predict"

app = modal.App(APP_NAME)

# Persistent volume for Boltz-2 model weights (~6GB, downloaded on first call)
weights_volume = modal.Volume.from_name(
    "bdb-boltz-weights", create_if_missing=True
)

# Boltz GPU image. Boltz-2 is published on PyPI as `boltz` and pulls a
# CUDA-12 torch wheel automatically.
gpu_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("git", "wget", "build-essential")
    # Boltz-2 (>=2.2) uses NVIDIA cuequivariance for the triangular-multiply
    # kernel and requires CUDA 12.5+. We let pip pick a torch that matches
    # cuequivariance's nvidia-cublas-cu12>=12.5 constraint.
    .pip_install(
        # Match dev's known-working stack: torch 2.10 ships nvidia-cublas-cu12
        # 12.8 which satisfies cuequivariance>=12.5 requirement.
        "torch==2.10.0",
        "boltz==2.2.1",
        "cuequivariance==0.9.0",
        "cuequivariance-torch==0.9.0",
        "cuequivariance-ops-cu12==0.9.0",
        "cuequivariance-ops-torch-cu12==0.9.0",
        "fastapi[standard]",
        "pyyaml",
        "numpy",
    )
    .env(
        {
            "BOLTZ_CACHE": "/weights",
            "TORCH_HOME": "/weights/torch",
            "HF_HOME": "/weights/hf",
        }
    )
)


# ---------------------------------------------------------------------------
#  Internal: write YAMLs, run boltz predict, parse outputs
# ---------------------------------------------------------------------------


def _write_yaml(item: dict) -> str:
    """Render one prediction item to a Boltz YAML string.

    item shape:
        {"name": "task_001",
         "kind": "monomer" | "complex",
         "sequences": ["MKKL...", ...]}    # 1 for monomer, 2 for complex
    """
    seqs = item.get("sequences") or []
    chain_ids = ["A", "B", "C", "D", "E"]
    lines = ["sequences:"]
    for i, seq in enumerate(seqs):
        cid = chain_ids[i] if i < len(chain_ids) else f"X{i}"
        lines.append("  - protein:")
        lines.append(f"      id: {cid}")
        lines.append(f"      sequence: {seq}")
    return "\n".join(lines) + "\n"


def _parse_confidence(pred_dir) -> dict:
    """Parse a Boltz prediction directory into a flat metric dict."""
    import json
    from pathlib import Path

    import numpy as np

    out = {
        "pLDDT": 0.0, "pTM": 0.0, "ipTM": 0.0, "i_pAE": 0.0,
        "success": False,
    }
    pred_dir = Path(pred_dir)

    conf_files = list(pred_dir.rglob("confidence*.json"))
    if conf_files:
        try:
            with open(conf_files[0]) as f:
                c = json.load(f)
            out["pLDDT"] = round(float(c.get("complex_plddt", 0.0)) * 100, 2)
            out["pTM"] = round(float(c.get("ptm", 0.0)), 4)
            out["ipTM"] = round(float(c.get("iptm", 0.0)), 4)
            out["i_pAE"] = round(float(c.get("complex_ipae", 0.0)), 2)
            out["success"] = True
        except Exception:
            pass

    if not out["success"]:
        # Fall back to per-residue plddt npz if confidence.json is missing
        plddt_files = list(pred_dir.rglob("plddt*.npz"))
        if plddt_files:
            try:
                arr = np.load(plddt_files[0])["plddt"]
                out["pLDDT"] = round(float(arr.mean()) * 100, 2)
                out["success"] = True
            except Exception:
                pass

    return out


# ---------------------------------------------------------------------------
#  GPU entry point — single web endpoint handling both monomer and complex
# ---------------------------------------------------------------------------


@app.function(
    image=gpu_image,
    gpu="A10G",
    volumes={"/weights": weights_volume},
    timeout=1800,
    scaledown_window=300,
    secrets=[modal.Secret.from_name("bdb-boltz-shared", required_keys=["TOKEN"])],
)
@modal.fastapi_endpoint(method="POST", label=ENDPOINT_LABEL)
def predict(payload: dict) -> dict:
    """Run Boltz-2 on a list of prediction items.

    Body shape:
        {"token": "<shared secret>",
         "items": [{"name": "...", "kind": "monomer"|"complex",
                    "sequences": [...]}, ...]}

    The list is assembled into a single ``boltz predict`` invocation so
    the model loads only once per call (amortizes ~30s cold start).

    Returns a dict mapping each item's `name` to a metric dict:
        {"pLDDT", "pTM", "ipTM", "i_pAE", "success"}
    """
    import shutil
    import subprocess
    import tempfile
    from pathlib import Path

    expected_token = os.environ.get("TOKEN", "")
    if expected_token and (payload.get("token") or "") != expected_token:
        return {"error": "Unauthorized -- bad MODAL_BOLTZ_TOKEN"}

    items = payload.get("items") or []
    if not items:
        return {"results": {}}

    work = Path(tempfile.mkdtemp(prefix="bdb_boltz_"))
    in_dir = work / "inputs"
    out_dir = work / "out"
    in_dir.mkdir()
    out_dir.mkdir()

    name_to_yaml: dict[str, str] = {}
    for i, item in enumerate(items):
        name = str(item.get("name") or f"item_{i:04d}")
        safe = "".join(c if c.isalnum() else "_" for c in name)[:60]
        yaml_name = f"{i:04d}_{safe}"
        (in_dir / f"{yaml_name}.yaml").write_text(_write_yaml(item))
        name_to_yaml[name] = yaml_name

    cmd = [
        "boltz", "predict",
        str(in_dir),
        "--out_dir", str(out_dir),
        "--cache", "/weights/boltz_cache",
        "--diffusion_samples", "1",
        "--output_format", "pdb",
        "--use_msa_server",
    ]

    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=1700, cwd=str(work),
    )

    # Persist downloaded model weights to the shared volume
    try:
        weights_volume.commit()
    except Exception:
        pass

    if proc.returncode != 0:
        shutil.rmtree(str(work), ignore_errors=True)
        return {
            "error": "boltz predict failed",
            "stderr": proc.stderr[-2000:],
            "stdout": proc.stdout[-2000:],
        }

    # boltz writes outputs to out/boltz_results_inputs/predictions/<name>/
    predictions_root = None
    for p in out_dir.rglob("predictions"):
        if p.is_dir():
            predictions_root = p
            break

    results: dict[str, dict] = {}
    if predictions_root is not None:
        for name, yaml_name in name_to_yaml.items():
            pred_dirs = [
                d for d in predictions_root.iterdir()
                if d.is_dir() and (d.name.startswith(yaml_name) or d.name == yaml_name)
            ]
            if pred_dirs:
                results[name] = _parse_confidence(pred_dirs[0])
            else:
                results[name] = {
                    "pLDDT": 0.0, "pTM": 0.0, "ipTM": 0.0, "i_pAE": 0.0,
                    "success": False, "error": "prediction missing",
                }

    shutil.rmtree(str(work), ignore_errors=True)
    return {"results": results}


# ---------------------------------------------------------------------------
#  CLI smoke test:  modal run modal_boltz_app.py
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main():
    """Quick sanity check — a short ubiquitin-like sequence."""
    import json

    items = [
        {
            "name": "monomer_demo",
            "kind": "monomer",
            "sequences": [
                "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
            ],
        },
    ]
    out = predict.remote(items, authorization="")
    print(json.dumps(out, indent=2))
