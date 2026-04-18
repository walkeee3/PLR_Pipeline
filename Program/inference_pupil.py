"""
Pupil Diameter Inference
========================
Given an eye video and a fine-tuned checkpoint, outputs per-frame pupil
diameter predictions. No CSV or labels needed.

The video is expected to already be a cropped eye recording — each frame
is fed directly to the model (resized to 224×224), exactly as during training.

Usage:
    python inference_pupil.py \
        --video   GS_F_07_032-1_left.mp4 \
        --weights finetuned_pupil_model.pth \
        --output  predictions.csv

Output CSV columns:
    frame | timestamp_sec | pred_diameter_px
"""

import argparse
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


# ─────────────────────────────────────────────────────────────────────────────
# 1.  MODEL
# ─────────────────────────────────────────────────────────────────────────────

def build_model(checkpoint_path: str, device: torch.device) -> nn.Module:
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)

    ckpt = torch.load(checkpoint_path, map_location=device)

    if "model_state" in ckpt:
        state = ckpt["model_state"]
    elif "state_dict" in ckpt:
        # PyTorch-Lightning checkpoint — strip "model." prefix
        state = {
            (k[len("model."):] if k.startswith("model.") else k): v
            for k, v in ckpt["state_dict"].items()
        }
    else:
        state = ckpt

    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 2.  DATASET
# ─────────────────────────────────────────────────────────────────────────────

IMG_SIZE      = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class InferenceVideoDataset(Dataset):
    """
    Streams every frame of an eye video.
    No cropping or detection — the full frame is the model input.
    """

    def __init__(self, video_path: str):
        self.video_path = video_path
        cap = cv2.VideoCapture(video_path)
        self.n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps      = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

    def __len__(self) -> int:
        return self.n_frames

    def __getitem__(self, idx: int) -> dict:
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame_bgr = cap.read()
        cap.release()

        if ret:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

        img_t = _TRANSFORM(Image.fromarray(frame_rgb))

        return {
            "img":       img_t,
            "frame":     idx,
            "timestamp": round(idx / self.fps, 4) if self.fps > 0 else 0.0,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 3.  INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(
    video_path: str,
    checkpoint_path: str,
    output_csv: str,
    batch_size: int  = 64,
    num_workers: int = 2,
    device: torch.device = None,
) -> pd.DataFrame:

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] Device    : {device}")
    print(f"[INFO] Video     : {video_path}")
    print(f"[INFO] Weights   : {checkpoint_path}")

    model = build_model(checkpoint_path, device)
    print("[INFO] Model loaded.\n")

    ds     = InferenceVideoDataset(video_path)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"[INFO] Total frames : {len(ds)}")
    print(f"[INFO] FPS          : {ds.fps:.2f}")
    print(f"[INFO] Duration     : {len(ds)/ds.fps:.1f} s\n")

    all_frames, all_ts, all_preds = [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            preds = model(batch["img"].to(device)).squeeze(1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_frames.extend(batch["frame"].numpy().tolist())
            all_ts.extend(batch["timestamp"].numpy().tolist())

    results = pd.DataFrame({
        "frame":            all_frames,
        "timestamp_sec":    all_ts,
        "pred_diameter_px": [round(p, 4) for p in all_preds],
    }).sort_values("frame").reset_index(drop=True)

    results.to_csv(output_csv, index=False)

    print(f"\n[INFO] Diameter stats —"
          f"  mean={results['pred_diameter_px'].mean():.2f} px"
          f"  std={results['pred_diameter_px'].std():.2f} px"
          f"  min={results['pred_diameter_px'].min():.2f} px"
          f"  max={results['pred_diameter_px'].max():.2f} px")
    print(f"[INFO] Predictions saved → {output_csv}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 4.  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Predict pupil diameter for every frame of an eye video."
    )
    p.add_argument("--video",       required=True,  help="Path to input eye video (.mp4).")
    p.add_argument("--weights",     required=True,  help="Path to fine-tuned .pth checkpoint.")
    p.add_argument("--output",      default="predictions_2.csv")
    p.add_argument("--batch_size",  type=int, default=64)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--no_gpu",      action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    device = torch.device("cpu") if args.no_gpu else \
             torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_inference(
        video_path      = args.video,
        checkpoint_path = args.weights,
        output_csv      = args.output,
        batch_size      = args.batch_size,
        num_workers     = args.num_workers,
        device          = device,
    )