"""WS-091 gap 4: key sets of the VLA-JEPA checkpoint load (AUDIT_W1_LOADS.md row F), logged once.

    shared/gate/jetson_run.sh --python journal/papers/jepa-world-model/gaps/vla_jepa_keyset.py <snapshot> <out.json>

Loads exactly as projects/vla-jepa-world-model/jetson/vla_jepa_surprise.load_policy, then compares the module's
state_dict keys with model.safetensors: missing = in the module, not in the file (left at init); unexpected = in the
file, not in the module (dropped by strict=False).
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "projects/vla-jepa-world-model/jetson"))
from vla_jepa_surprise import load_policy  # noqa: E402


def main():
    ckpt, out = Path(sys.argv[1]), Path(sys.argv[2])
    from safetensors import safe_open

    _, pol, _ = load_policy(str(ckpt))
    with safe_open(str(ckpt / "model.safetensors"), "pt") as f:
        file_keys = set(f.keys())
    mod_keys = set(pol.state_dict().keys())
    missing, unexpected = sorted(mod_keys - file_keys), sorted(file_keys - mod_keys)
    wm = [k for k in mod_keys if "world" in k.lower() or "predictor" in k.lower()]
    res = {"checkpoint": str(ckpt), "module_keys": len(mod_keys), "file_keys": len(file_keys),
           "missing": len(missing), "unexpected": len(unexpected), "missing_first": missing[:20],
           "unexpected_first": unexpected[:20], "world_model_keys": len(wm),
           "world_model_keys_missing": len([k for k in wm if k in missing])}
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if not k.endswith("_first")}))


if __name__ == "__main__":
    main()
