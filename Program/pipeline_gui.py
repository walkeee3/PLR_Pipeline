"""
Pupil Analysis Pipeline — GUI
==============================
Run:  python pipeline_gui.py

Expected folder layout (all paths relative to this script's directory,
but configurable in the PATHS section below):

  Videos/                  ← drop uncropped input videos here
  Videos_left/             ← cropped left-eye videos land here
  Videos_right/            ← cropped right-eye videos land here
  Time_Markings/           ← equiluminance CSV files (one or more)
  Models/
      left.pth
      right.pth
  Output/
      <ID>/
          left/
          right/
  Timestamps/              ← timestamp CSV files (one or more)

Program files expected alongside this script:
  crop_videos_lib.py       (extracted from crop_videos.ipynb — generated here)
  inference_pupil.py
  pupil_analysis_v10.py
"""

import tkinter as tk
from tkinter import ttk, messagebox, font as tkfont
import threading
import queue
import os
import re
import sys
import shutil
import glob

# ─────────────────────────────────────────────────────────────────────────────
#  PATH CONFIGURATION  — edit these to match your machine
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))

VIDEOS_IN       = os.path.join(BASE_DIR, "Videos")
VIDEOS_LEFT     = os.path.join(BASE_DIR, "Videos_left")
VIDEOS_RIGHT    = os.path.join(BASE_DIR, "Videos_right")
TIME_MARKINGS   = os.path.join(BASE_DIR, "Time_Markings")
MODELS_DIR      = os.path.join(BASE_DIR, "Models")
OUTPUT_DIR      = os.path.join(BASE_DIR, "Output")
TIMESTAMPS_DIR  = os.path.join(BASE_DIR, "Timestamps")

MODEL_LEFT      = os.path.join(MODELS_DIR, "left.pth")
MODEL_RIGHT     = os.path.join(MODELS_DIR, "right.pth")

VIDEO_EXTS      = ('.mp4', '.avi', '.mov', '.mkv')

# ─────────────────────────────────────────────────────────────────────────────
#  COLOURS / STYLE
# ─────────────────────────────────────────────────────────────────────────────
BG          = "#1E1E2E"
SURFACE     = "#2A2A3E"
ACCENT      = "#7C6FF7"
ACCENT2     = "#56CFE1"
SUCCESS     = "#50FA7B"
WARNING     = "#FFB86C"
ERROR       = "#FF5555"
TEXT        = "#CDD6F4"
TEXT_DIM    = "#6C7086"
MARKED_COL  = "#FF6E6E"

# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def extract_id(filename: str) -> str:
    """
    Extract participant ID from video filename.
    Handles:
      GS_F_08_046-1_something.mp4  ->  GS_F_08_046-1
      OCT_F_18_146_direct.mp4      ->  OCT_F_18_146
    Rule: take first 4 underscore-tokens; stop before any pure-alpha suffix token.
    """
    name = os.path.splitext(os.path.basename(filename))[0]
    # Format 1: XX_X_NN_NNN-N  (with dash-digit)
    m = re.match(r'^([A-Z]{2,3}_[A-Z]_\d{2}_\d{3}-\d)', name)
    if m:
        return m.group(1)
    # Format 2: XXX_X_NN_NNN  (no dash) — take first 4 tokens, drop alpha suffixes
    parts = name.split('_')
    id_parts = []
    for p in parts:
        if len(id_parts) >= 4 and re.fullmatch(r'[A-Za-z]+', p):
            break
        id_parts.append(p)
        if len(id_parts) == 4:
            break
    return '_'.join(id_parts)


def find_timestamps(participant_id: str):
    """Search all CSVs in Timestamps/ for a row whose first column == participant_id."""
    for csv_file in glob.glob(os.path.join(TIMESTAMPS_DIR, "*.csv")):
        try:
            import pandas as pd
            df = pd.read_csv(csv_file, header=None)
            row = df[df.iloc[:, 0] == participant_id]
            if not row.empty:
                ts = row.iloc[0, 1:].dropna().astype(int).tolist()
                return ts, csv_file
        except Exception:
            pass
    return None, None


def find_equiluminance_row(participant_id: str):
    """Search Time_Markings/ CSVs for participant_id in first column."""
    for csv_file in glob.glob(os.path.join(TIME_MARKINGS, "*.csv")):
        try:
            import pandas as pd
            df = pd.read_csv(csv_file, header=None)
            row = df[df.iloc[:, 0] == participant_id]
            if not row.empty:
                return csv_file
        except Exception:
            pass
    return None


def mark_id_folder(participant_id: str):
    """Rename Output/<ID> to Output/<ID>_marked."""
    src = os.path.join(OUTPUT_DIR, participant_id)
    dst = os.path.join(OUTPUT_DIR, f"{participant_id}_marked")
    if os.path.exists(src):
        if os.path.exists(dst):
            shutil.rmtree(dst)
        os.rename(src, dst)
        return dst
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  PIPELINE WORKER  (runs in a background thread)
# ─────────────────────────────────────────────────────────────────────────────

class PipelineWorker:
    def __init__(self, participant_id: str, log_q: queue.Queue,
                 confirm_q: queue.Queue, result_q: queue.Queue):
        self.pid       = participant_id
        self.log_q     = log_q      # str messages → GUI log
        self.confirm_q = confirm_q  # GUI puts True/False here after showing results
        self.result_q  = result_q   # worker puts final status here

    def log(self, msg, level="info"):
        self.log_q.put((level, msg))

    def run(self):
        try:
            self._run()
        except Exception as e:
            import traceback
            self.log(f"UNEXPECTED ERROR: {e}", "error")
            self.log(traceback.format_exc(), "error")
            self.result_q.put(("error", str(e)))

    def _run(self):
        pid = self.pid
        self.log(f"{'='*60}", "sep")
        self.log(f"  Processing: {pid}", "header")
        self.log(f"{'='*60}", "sep")

        # ── 1. Locate input video ─────────────────────────────────────────
        video_files = [
            f for f in os.listdir(VIDEOS_IN)
            if f.lower().endswith(VIDEO_EXTS) and extract_id(f) == pid
        ]
        if not video_files:
            self.log(f"No video found for {pid} in Videos/", "error")
            self.result_q.put(("skip", pid))
            return
        video_path = os.path.join(VIDEOS_IN, video_files[0])
        self.log(f"Input video: {video_files[0]}")

        # ── 2. Find equiluminance CSV ─────────────────────────────────────
        equil_csv = find_equiluminance_row(pid)
        if not equil_csv:
            self.log(f"No equiluminance row found for {pid} in Time_Markings/", "warning")

        # ── 3. Crop video → left + right, timestamps from equil CSV ─────────
        left_out  = os.path.join(VIDEOS_LEFT,  f"{pid}_left.mp4")
        right_out = os.path.join(VIDEOS_RIGHT, f"{pid}_right.mp4")
        timestamps = []

        if os.path.exists(left_out) and os.path.exists(right_out):
            self.log(f"Cropped videos already exist — skipping crop step.")
            # Re-derive timestamps from equiluminance CSV without re-cropping
            if equil_csv:
                try:
                    from crop_videos_lib import get_timestamps_only
                    timestamps = get_timestamps_only(video_path, equil_csv, pid, self.log)
                    self.log(f"Timestamps re-derived: {len(timestamps)} stimuli")
                except Exception as e:
                    self.log(f"Could not re-derive timestamps: {e}", "warning")
        else:
            self.log(f"Cropping video …")
            try:
                from crop_videos_lib import crop_and_extract
                timestamps = crop_and_extract(video_path, equil_csv, pid,
                                              left_out, right_out, self.log)
                self.log(f"Timestamps from equiluminance CSV: {len(timestamps)} stimuli")
            except ImportError:
                self.log("crop_videos_lib not found — attempting raw crop …", "warning")
                self._raw_crop(video_path, left_out, right_out)
            except Exception as e:
                self.log(f"Crop failed: {e}", "error")
                self.result_q.put(("skip", pid))
                return

        if not timestamps:
            self.log(f"No timestamps — analysis plots will be skipped.", "warning")

        # ── 4. Process each eye ───────────────────────────────────────────
        for side in ("left", "right"):
            self._process_eye(pid, side, timestamps)

        self.result_q.put(("done", pid))


    def _raw_crop(self, video_path, left_out, right_out):
        """Fallback: crop using the hardcoded dimensions from crop_videos.ipynb."""
        import cv2
        cap = cv2.VideoCapture(video_path)
        EYE_TOP, EYE_BOTTOM = 340, 640
        EYE_LEFT, EYE_RIGHT = 80, 880
        eye_h = EYE_BOTTOM - EYE_TOP
        eye_w = (EYE_RIGHT - EYE_LEFT) // 2
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        rw = cv2.VideoWriter(right_out, fourcc, 30, (eye_w, eye_h))
        lw = cv2.VideoWriter(left_out,  fourcc, 30, (eye_w, eye_h))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            crop = frame[EYE_TOP:EYE_BOTTOM, EYE_LEFT:EYE_RIGHT]
            mid  = crop.shape[1] // 2
            rw.write(cv2.cvtColor(crop[:, :mid],  cv2.COLOR_BGR2RGB))
            lw.write(cv2.cvtColor(crop[:, mid:],  cv2.COLOR_BGR2RGB))
        rw.release(); lw.release(); cap.release()
        self.log(f"Raw crop complete → {os.path.basename(left_out)}, {os.path.basename(right_out)}")

    def _process_eye(self, pid: str, side: str, timestamps: list):
        self.log(f"\n── {side.upper()} eye ──────────────────────────────", "subheader")

        # Check model
        model_path = MODEL_LEFT if side == "left" else MODEL_RIGHT
        if not os.path.exists(model_path):
            self.log(f"Model not found: {model_path}", "warning")
            self.log(f"Skipping {side} eye.")
            return

        # Check cropped video
        video_path = os.path.join(
            VIDEOS_LEFT if side == "left" else VIDEOS_RIGHT,
            f"{pid}_{side}.mp4"
        )
        if not os.path.exists(video_path):
            self.log(f"Cropped video not found: {video_path}", "warning")
            return

        # Output folder
        out_dir = os.path.join(OUTPUT_DIR, pid, side)
        os.makedirs(out_dir, exist_ok=True)

        # ── 5a. Inference ─────────────────────────────────────────────────
        pred_csv = os.path.join(out_dir, f"{pid}_{side}_pred.csv")
        if os.path.exists(pred_csv):
            self.log(f"Prediction CSV exists — skipping inference.")
        else:
            self.log(f"Running inference …")
            try:
                import torch
                sys.path.insert(0, BASE_DIR)
                from inference_pupil import run_inference
                import torch
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.log(f"  Device: {device}")
                run_inference(
                    video_path=video_path,
                    checkpoint_path=model_path,
                    output_csv=pred_csv,
                    batch_size=64,
                    num_workers=0,
                    device=device,
                )
                self.log(f"  Predictions saved → {os.path.basename(pred_csv)}", "success")
            except Exception as e:
                self.log(f"Inference failed: {e}", "error")
                return

        # ── 5b. Pupil analysis ────────────────────────────────────────────
        if not timestamps:
            self.log(f"No timestamps — skipping analysis plots.", "warning")
        else:
            self.log(f"Running pupil analysis …")
            try:
                self._run_analysis(pid, side, pred_csv, timestamps, out_dir)
            except Exception as e:
                import traceback
                self.log(f"Analysis failed: {e}", "error")
                self.log(traceback.format_exc(), "error")
                return

        # ── 5c. Show results + await confirmation ─────────────────────────
        self.log(f"Showing results for {pid} {side} eye …", "success")
        # Signal GUI to display results and wait
        self.log_q.put(("show_results", {
            "pid": pid, "side": side, "out_dir": out_dir,
            "pred_csv": pred_csv, "timestamps": timestamps,
        }))
        # Block until user confirms
        confirmed = self.confirm_q.get()
        if confirmed == "mark":
            self.log(f"Marked {pid} as problematic.", "warning")
            mark_id_folder(pid)
        else:
            self.log(f"Confirmed {pid} {side} — moving on.", "success")

    def _run_analysis(self, pid, side, pred_csv, timestamps, out_dir):
        import numpy as np
        import pandas as pd
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from scipy.signal import medfilt
        from scipy import signal as sp_signal

        # ── Import analysis functions from pupil_analysis_v10 ─────────────
        sys.path.insert(0, BASE_DIR)
        from pupil_analysis_v10 import (
            preprocess, adaptive_threshold2,
            compute_all_mca_mcv,
            save_signal_figure,
            save_signed_mca_figure,
            save_detection_windows_figure,
            PRED_COLOR, STIM_COLOR, CONS_COLOR, DIL_COLOR,
            PT_A_COLOR, PT_B_COLOR, MCV_COLOR,
        )

        pred_df   = pd.read_csv(pred_csv)
        pred_raw  = pred_df["pred_diameter_px"].values.astype(float)
        pred_conf = np.ones(len(pred_raw))

        ts = [t for t in timestamps if t < len(pred_raw)]

        pred_pre    = preprocess(pred_raw, pred_conf)
        pred_t2     = adaptive_threshold2(pred_raw)
        pred_pre_t2 = adaptive_threshold2(pred_pre)

        # Signal overview — no colours (don't know stimulus order)
        save_signal_figure(
            pred_raw, ts,
            f"{pid} ({side})  –  Predicted  [Raw]",
            PRED_COLOR, "Predicted (raw)",
            os.path.join(out_dir, "signal_raw.png"),
        )
        save_signal_figure(
            pred_pre, ts,
            f"{pid} ({side})  –  Predicted  [Preprocessed]",
            PRED_COLOR, "Predicted (preprocessed)",
            os.path.join(out_dir, "signal_pre.png"),
        )

        # MCA / MCV
        pred_raw_mca, pred_raw_mcv, pred_raw_detail = compute_all_mca_mcv(
            pred_raw, ts, pred_t2, signal_type='pred')
        pred_pre_mca, pred_pre_mcv, pred_pre_detail = compute_all_mca_mcv(
            pred_pre, ts, pred_pre_t2, signal_type='pred')

        # Signed MCA
        save_signed_mca_figure(
            pred_raw_mca, pred_pre_mca,
            f"Predicted — {pid} ({side})",
            PRED_COLOR,
            os.path.join(out_dir, "signed_mca.png"),
        )

        # Detection windows (preprocessed only)
        save_detection_windows_figure(
            pred_pre, pred_pre_detail, pred_pre_mca,
            PRED_COLOR,
            f"Predicted Preprocessed — {pid} ({side})",
            os.path.join(out_dir, "detection_windows_pre.png"),
        )

        # CSV
        rows = []
        for i, (lbl, _) in enumerate(pred_raw_mca):
            rows.append({
                "stimulus":     lbl,
                "pred_raw_mca": pred_raw_mca[i][1],
                "pred_raw_mcv": pred_raw_mcv[i][1],
                "pred_pre_mca": pred_pre_mca[i][1],
                "pred_pre_mcv": pred_pre_mcv[i][1],
            })
        pd.DataFrame(rows).to_csv(
            os.path.join(out_dir, "mca_mcv_results.csv"), index=False)

        self.log(f"  Analysis outputs saved to {out_dir}", "success")


# ─────────────────────────────────────────────────────────────────────────────
#  GUI  APPLICATION
# ─────────────────────────────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Pupil Analysis Pipeline")
        self.configure(bg=BG)
        self.resizable(True, True)
        self.minsize(900, 650)

        self._log_q     = queue.Queue()
        self._confirm_q = queue.Queue()
        self._result_q  = queue.Queue()
        self._pending_results = None   # dict from show_results event
        self._current_pid = None

        self._build_ui()
        self._poll()

    # ── UI BUILD ──────────────────────────────────────────────────────────

    def _build_ui(self):
        # Title bar
        hdr = tk.Frame(self, bg=ACCENT, pady=8)
        hdr.pack(fill="x")
        tk.Label(hdr, text="Pupil Analysis Pipeline",
                 bg=ACCENT, fg="white",
                 font=("Segoe UI", 16, "bold")).pack()
        tk.Label(hdr, text="Automated crop → inference → analysis",
                 bg=ACCENT, fg="#D0D0FF",
                 font=("Segoe UI", 9)).pack()

        # Main body
        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=16, pady=10)

        # Left panel: video list + controls
        left = tk.Frame(body, bg=SURFACE, bd=0, relief="flat", width=260)
        left.pack(side="left", fill="y", padx=(0, 10))
        left.pack_propagate(False)

        tk.Label(left, text="Videos in queue", bg=SURFACE, fg=ACCENT2,
                 font=("Segoe UI", 10, "bold")).pack(pady=(10, 4))

        # Listbox
        lb_frame = tk.Frame(left, bg=SURFACE)
        lb_frame.pack(fill="both", expand=True, padx=8)
        scrollbar = tk.Scrollbar(lb_frame)
        scrollbar.pack(side="right", fill="y")
        self._listbox = tk.Listbox(
            lb_frame, bg="#1A1A2E", fg=TEXT,
            selectbackground=ACCENT, selectforeground="white",
            font=("Consolas", 9), bd=0, highlightthickness=0,
            yscrollcommand=scrollbar.set,
            activestyle="none",
        )
        self._listbox.pack(fill="both", expand=True)
        scrollbar.config(command=self._listbox.yview)

        # Buttons
        btn_frame = tk.Frame(left, bg=SURFACE)
        btn_frame.pack(fill="x", padx=8, pady=8)

        self._btn_refresh = self._make_button(
            btn_frame, "⟳  Refresh", self._refresh_videos, ACCENT)
        self._btn_refresh.pack(fill="x", pady=2)

        self._btn_start = self._make_button(
            btn_frame, "▶  Process All", self._start_pipeline, SUCCESS)
        self._btn_start.pack(fill="x", pady=2)

        # Right panel: log
        right = tk.Frame(body, bg=SURFACE)
        right.pack(side="left", fill="both", expand=True)

        tk.Label(right, text="Log", bg=SURFACE, fg=ACCENT2,
                 font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=8, pady=(8, 2))

        log_frame = tk.Frame(right, bg=SURFACE)
        log_frame.pack(fill="both", expand=True, padx=8, pady=(0, 4))
        log_scroll = tk.Scrollbar(log_frame)
        log_scroll.pack(side="right", fill="y")
        self._log = tk.Text(
            log_frame, bg="#13131F", fg=TEXT,
            font=("Consolas", 9), bd=0, highlightthickness=0,
            state="disabled", wrap="word",
            yscrollcommand=log_scroll.set,
        )
        self._log.pack(fill="both", expand=True)
        log_scroll.config(command=self._log.yview)

        # Tag colours
        self._log.tag_config("info",      foreground=TEXT)
        self._log.tag_config("success",   foreground=SUCCESS)
        self._log.tag_config("warning",   foreground=WARNING)
        self._log.tag_config("error",     foreground=ERROR)
        self._log.tag_config("header",    foreground=ACCENT,  font=("Consolas", 9, "bold"))
        self._log.tag_config("subheader", foreground=ACCENT2)
        self._log.tag_config("sep",       foreground=TEXT_DIM)

        # Bottom: confirmation bar (hidden until needed)
        self._confirm_bar = tk.Frame(self, bg="#2D1B4E", pady=6)
        self._confirm_lbl = tk.Label(
            self._confirm_bar,
            text="", bg="#2D1B4E", fg="white",
            font=("Segoe UI", 10))
        self._confirm_lbl.pack(side="left", padx=16)

        btn_row = tk.Frame(self._confirm_bar, bg="#2D1B4E")
        btn_row.pack(side="right", padx=16)

        self._btn_view = self._make_button(
            btn_row, "📂 Open Folder", self._open_folder, ACCENT2, width=14)
        self._btn_view.pack(side="left", padx=4)

        self._btn_mark = self._make_button(
            btn_row, "⚑ Mark as Bad", self._mark_bad, MARKED_COL, width=14)
        self._btn_mark.pack(side="left", padx=4)

        self._btn_ok = self._make_button(
            btn_row, "✓  Confirm & Next", self._confirm_ok, SUCCESS, width=16)
        self._btn_ok.pack(side="left", padx=4)

        # Status bar
        self._status_var = tk.StringVar(value="Ready — drop videos in the Videos/ folder")
        status = tk.Label(self, textvariable=self._status_var,
                          bg="#13131F", fg=TEXT_DIM,
                          font=("Segoe UI", 8), anchor="w", padx=10, pady=4)
        status.pack(fill="x", side="bottom")

        self._refresh_videos()

    def _make_button(self, parent, text, cmd, color, width=None):
        kw = dict(bg=color, fg="white", font=("Segoe UI", 9, "bold"),
                  relief="flat", bd=0, padx=10, pady=6,
                  cursor="hand2", activebackground=color,
                  activeforeground="white", command=cmd)
        if width:
            kw["width"] = width
        return tk.Button(parent, text=text, **kw)

    # ── VIDEO MANAGEMENT ─────────────────────────────────────────────────

    def _refresh_videos(self):
        self._listbox.delete(0, "end")
        if not os.path.isdir(VIDEOS_IN):
            os.makedirs(VIDEOS_IN, exist_ok=True)
        videos = sorted([
            f for f in os.listdir(VIDEOS_IN)
            if f.lower().endswith(VIDEO_EXTS)
        ])
        for v in videos:
            pid = extract_id(v)
            self._listbox.insert("end", f"  {pid}  ({v})")
        count = len(videos)
        self._status_var.set(
            f"{count} video{'s' if count != 1 else ''} found in Videos/")

    def _get_video_ids(self):
        ids = []
        for f in sorted(os.listdir(VIDEOS_IN)):
            if f.lower().endswith(VIDEO_EXTS):
                ids.append(extract_id(f))
        return ids

    # ── PIPELINE START ────────────────────────────────────────────────────

    def _start_pipeline(self):
        ids = self._get_video_ids()
        if not ids:
            messagebox.showinfo("No videos",
                                "No videos found in the Videos/ folder.\n"
                                "Please add videos and click Refresh.")
            return
        self._btn_start.config(state="disabled")
        self._btn_refresh.config(state="disabled")
        self._log_write("Pipeline started.", "success")
        self._queue = list(ids)
        self._process_next()

    def _process_next(self):
        if not self._queue:
            self._log_write("\n✓  All videos processed.", "success")
            self._btn_start.config(state="normal")
            self._btn_refresh.config(state="normal")
            self._status_var.set("Pipeline complete.")
            return

        pid = self._queue.pop(0)
        self._current_pid = pid
        self._status_var.set(f"Processing: {pid}")

        worker = PipelineWorker(pid, self._log_q, self._confirm_q, self._result_q)
        t = threading.Thread(target=worker.run, daemon=True)
        t.start()

    # ── POLLING ───────────────────────────────────────────────────────────

    def _poll(self):
        # Drain log queue
        try:
            while True:
                item = self._log_q.get_nowait()
                level, msg = item
                if level == "show_results":
                    self._pending_results = msg
                    self._show_confirm_bar(msg)
                else:
                    self._log_write(msg, level)
        except queue.Empty:
            pass

        # Check result queue
        try:
            status, pid = self._result_q.get_nowait()
            self._log_write(f"\nFinished {pid}: {status}", "success" if status=="done" else "warning")
            self._process_next()
        except queue.Empty:
            pass

        self.after(100, self._poll)

    # ── LOG HELPERS ───────────────────────────────────────────────────────

    def _log_write(self, msg: str, level: str = "info"):
        self._log.config(state="normal")
        self._log.insert("end", msg + "\n", level)
        self._log.see("end")
        self._log.config(state="disabled")

    # ── CONFIRMATION BAR ─────────────────────────────────────────────────

    def _show_confirm_bar(self, info: dict):
        pid  = info["pid"]
        side = info["side"]
        self._confirm_lbl.config(
            text=f"Results ready for  {pid}  [{side.upper()} eye] — review plots then confirm or mark.")
        self._confirm_bar.pack(fill="x", side="bottom", before=self.nametowidget(
            self.pack_slaves()[-1]))

    def _hide_confirm_bar(self):
        self._confirm_bar.pack_forget()

    def _confirm_ok(self):
        self._hide_confirm_bar()
        self._pending_results = None
        self._confirm_q.put("ok")

    def _mark_bad(self):
        self._hide_confirm_bar()
        res = self._pending_results
        self._pending_results = None
        self._confirm_q.put("mark")

    def _open_folder(self):
        if self._pending_results:
            out = self._pending_results.get("out_dir", OUTPUT_DIR)
        else:
            out = OUTPUT_DIR
        # Cross-platform open
        import subprocess
        if sys.platform == "win32":
            os.startfile(out)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", out])
        else:
            subprocess.Popen(["xdg-open", out])


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Create required folders if missing
    for d in (VIDEOS_IN, VIDEOS_LEFT, VIDEOS_RIGHT,
              TIME_MARKINGS, MODELS_DIR, OUTPUT_DIR, TIMESTAMPS_DIR):
        os.makedirs(d, exist_ok=True)

    app = App()
    app.mainloop()