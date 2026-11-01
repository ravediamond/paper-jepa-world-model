"""WS-091 gaps 2-3: W2' encoder features and commanded joints on the 69 rollouts, at the frames the gates read.

    W1=1 shared/gate/jetson_run.sh --python journal/papers/jepa-world-model/gaps/extract_controls.py --out <dir>

Per episode (rollouts @ the FACTS.json revision): <out>/w2p/epNNN.npy (T, 768) = frozen facebook/dinov2-small patch
tokens mean-pooled per view, top + wrist concatenated (W2' encoder, as projects/lejepa-identifiability gate0_extract),
filled only at the judge frame (projects/surveyor/results/judge.csv), the mid frame T // 2 and the W1-C1 anchor
(scripts/run_w1c_offline.c1_anchor); NaN elsewhere. <out>/joints/epNNN.npy (T, 6) = the recorded action (commanded
joints) at every frame. Frames: so101_harness EpisodeDataset + w1_live.preprocess (the W1 pipeline), then ImageNet
normalisation for DINOv2.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts"))
from run_w1c_offline import c1_anchor  # noqa: E402
from so101_harness.episodes import TOP, WRIST, EpisodeDataset  # noqa: E402
from so101_harness.w1_live import preprocess  # noqa: E402

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    from huggingface_hub import snapshot_download
    from transformers import AutoModel

    facts = json.loads((ROOT / "data/FACTS.json").read_text())["rollouts"]
    root = Path(snapshot_download(facts["repo"], repo_type="dataset", revision=facts["revision"]))
    labels = pd.read_csv(root / "meta/rollout_labels.csv")
    judge = pd.read_csv(ROOT / "projects/surveyor/results/judge.csv").set_index("episode")
    ds = EpisodeDataset(root)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dino = AutoModel.from_pretrained("facebook/dinov2-small").to(dev).eval()
    (args.out / "w2p").mkdir(parents=True, exist_ok=True)
    (args.out / "joints").mkdir(parents=True, exist_ok=True)
    frames_used = {}
    for r in labels.itertuples():
        ep = int(r.episode_index)
        acts = ds.actions(ep)
        T = len(acts)
        end = T if int(r.reset_start) < 0 else min(int(r.reset_start), T)
        need = {int(judge.loc[ep, "judge_frame"]), T // 2}
        a = c1_anchor(acts[:end, 5])
        if a is not None:
            need.add(int(a))
        h = np.full((T, 768), np.nan, np.float32)
        got = {}
        for (i, top), (_, wrist) in zip(ds.iter_frames(ep, TOP), ds.iter_frames(ep, WRIST)):
            if i in need:
                got[i] = [(preprocess(top) - MEAN) / STD, (preprocess(wrist) - MEAN) / STD]
        with torch.no_grad():
            for i, (t, w) in got.items():
                x = torch.stack([t, w]).to(dev)
                h[i] = dino(pixel_values=x).last_hidden_state[:, 1:].mean(1).reshape(-1).float().cpu().numpy()
        np.save(args.out / "w2p" / f"ep{ep:03d}.npy", h)
        np.save(args.out / "joints" / f"ep{ep:03d}.npy", acts.astype(np.float32))
        frames_used[ep] = sorted(got)
        print(f"ep {ep:3d} T {T} frames {sorted(got)}", flush=True)
    (args.out / "frames_used.json").write_text(json.dumps(frames_used))


if __name__ == "__main__":
    main()
