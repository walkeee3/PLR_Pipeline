"""
crop_videos_lib.py
==================
Functions extracted from crop_videos.ipynb for use by the pipeline GUI.
Tesseract path is auto-detected for Linux/WSL (no hardcoded Windows path).
"""

import cv2
import re
import os
import csv
import pandas as pd
import pytesseract

# ── Auto-detect Tesseract path ────────────────────────────────────────────────
import shutil as _shutil
_tess = _shutil.which("tesseract")
if _tess:
    pytesseract.pytesseract.tesseract_cmd = _tess
else:
    # Windows fallback
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# ─────────────────────────────────────────────────────────────────────────────
#  TIME NORMALISATION  (verbatim from notebook)
# ─────────────────────────────────────────────────────────────────────────────

def normalize_ocr_time(raw):
    if not raw:
        return None
    raw = raw.strip()
    raw = re.sub(r"[^0-9:]", "", raw)
    parts = re.findall(r"\d+", raw)
    if len(parts) < 3:
        return None
    h, mm, ss = parts[-3], parts[-2], parts[-1]
    try:
        mm_i = int(mm)
        ss_i = int(ss)
        if mm_i > 59 or ss_i > 59:
            return None
        return f"{mm_i:02d}:{ss_i:02d}"
    except Exception:
        return None


def normalize_csv_time(raw):
    if not raw:
        return None
    raw = raw.strip()
    raw = re.sub(r'\d{4}-\d{2}-\d{2}', '', raw)
    raw = re.sub(r"[^0-9:\.]", "", raw)
    parts = raw.split(":")
    if len(parts) != 2:
        return None
    try:
        mm = int(parts[0])
        sec_float = float(parts[1])
        ss = round(sec_float)
        mm += ss // 60
        ss = ss % 60
        return f"{mm:02d}:{ss:02d}"
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  OCR + CROP  (func from notebook)
# ─────────────────────────────────────────────────────────────────────────────

def func(vid_path, left_out, right_out, log_fn=print):
    """
    Crop video into left/right eye streams and extract OCR clock times.
    Returns (clocks, dict_seconds).
    """
    cap = cv2.VideoCapture(vid_path)
    clocks = []
    dict_seconds = {}

    ret, frame = cap.read()
    if not ret:
        log_fn("Could not read video.", "error")
        return [], {}

    EYE_TOP, EYE_BOTTOM = 340, 640
    EYE_LEFT, EYE_RIGHT = 80, 880
    eye_h = EYE_BOTTOM - EYE_TOP
    eye_w = (EYE_RIGHT - EYE_LEFT) // 2

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    right_writer = cv2.VideoWriter(right_out, fourcc, 30, (eye_w, eye_h))
    left_writer  = cv2.VideoWriter(left_out,  fourcc, 30, (eye_w, eye_h))

    frame_idx = 0
    prev_time_norm = None
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def run_ocr(crop):
        g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        p = cv2.resize(g, None, fx=8, fy=8)
        p = cv2.GaussianBlur(p, (3, 3), 0)
        _, p = cv2.threshold(p, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        raw = pytesseract.image_to_string(
            p, config="--psm 7 -c tessedit_char_whitelist=0123456789:")
        return normalize_ocr_time(raw)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        eyes_crop = frame[EYE_TOP:EYE_BOTTOM, EYE_LEFT:EYE_RIGHT]
        mid = eyes_crop.shape[1] // 2
        right_writer.write(cv2.cvtColor(eyes_crop[:, :mid],  cv2.COLOR_BGR2RGB))
        left_writer.write(cv2.cvtColor(eyes_crop[:, mid:], cv2.COLOR_BGR2RGB))

        clock_crop_1 = frame[1030:1050, 1810:1870]
        clock_crop_2 = frame[1030:1050, 1757:1817]

        if frame_idx % 3 == 0:
            norm = run_ocr(clock_crop_1)
            if not norm:
                norm = run_ocr(clock_crop_2)
            if norm and norm != prev_time_norm:
                clocks.append((frame_idx, norm))
                dict_seconds[len(clocks) - 1] = frame_idx
                prev_time_norm = norm

        frame_idx += 1
        if frame_idx % 500 == 0:
            log_fn(f"  Cropping … {frame_idx}/{total_frames} frames")

    right_writer.release()
    left_writer.release()
    cap.release()
    return clocks, dict_seconds


# ─────────────────────────────────────────────────────────────────────────────
#  SWITCH-CLOCK MATCHING  (func2 from notebook)
# ─────────────────────────────────────────────────────────────────────────────

def func2(vid_path, clocks, dict_seconds, path, equil_csv_path, log_fn=print):
    """
    Match equiluminance switch clocks to video frame numbers.
    Returns list of color_switch_frames (frame offsets from first switch).
    """
    data_df = pd.read_csv(equil_csv_path, header=None)
    match = data_df[data_df[0] == path]
    if match.empty:
        log_fn(f"Warning: no row found for '{path}' in equiluminance CSV.", "warning")
        return []

    row = match.iloc[0]
    # Drop NaN cells, keep only valid string values, skip first 4 metadata columns
    clocks_list = [str(v) for v in row.iloc[4:] if pd.notna(v)]

    switch_clocks = [
        normalize_csv_time(re.sub(r'\d{4}-\d{2}-\d{2}', '', x))
        for x in clocks_list
    ]
    switch_clocks = [x for x in switch_clocks if x]

    detected_times = {c[1]: c[0] for c in clocks}

    frame_ests = []
    for sw in switch_clocks:
        if sw in detected_times:
            frame_ests.append(detected_times[sw])
            continue
        sw_m, sw_s = map(int, sw.split(":"))
        sw_total = sw_m * 60 + sw_s
        best_frame = None
        best_diff = float("inf")
        for det_time, det_frame in detected_times.items():
            d_m, d_s = map(int, det_time.split(":"))
            diff = abs(sw_total - (d_m * 60 + d_s))
            if diff < best_diff:
                best_diff = diff
                best_frame = det_frame
        if best_frame is not None:
            frame_ests.append(best_frame)

    clock_switch_frames = [frame_ests[0]]
    for i in range(1, len(frame_ests)):
        clock_switch_frames.append(frame_ests[i])

    start_frame = clock_switch_frames[0]
    color_switch_frames = [f for f in clock_switch_frames]
    return color_switch_frames


# ─────────────────────────────────────────────────────────────────────────────
#  HIGH-LEVEL ENTRY POINT  (called by pipeline_gui.py)
# ─────────────────────────────────────────────────────────────────────────────

def crop_and_extract(video_path, equil_csv_path, participant_id,
                     left_out, right_out, log_fn=print):
    """
    Full pipeline step:
      1. Crop video into left/right eye streams
      2. Extract clock times via OCR
      3. Match to equiluminance switch clocks
    Returns color_switch_frames list (or empty list if equil_csv_path is None).
    """
    log_fn(f"  Cropping & extracting clocks from {os.path.basename(video_path)} …")
    clocks, dict_seconds = func(video_path, left_out, right_out, log_fn)
    log_fn(f"  Extracted {len(clocks)} clock readings.")

    if equil_csv_path and os.path.exists(equil_csv_path):
        log_fn(f"  Matching equiluminance switch clocks …")
        frames = func2(video_path, clocks, dict_seconds,
                       participant_id, equil_csv_path, log_fn)
        log_fn(f"  Found {len(frames)} switch frames.")
        return frames
    else:
        log_fn(f"  No equiluminance CSV — skipping switch-clock matching.", "warning")
        return []


def get_timestamps_only(video_path, equil_csv_path, participant_id, log_fn=print):
    """
    Re-derive timestamps from an already-cropped video without writing output files.
    Runs OCR on the original video to get clock readings, then matches against
    the equiluminance CSV. Returns color_switch_frames list.
    """
    import tempfile, os
    log_fn(f"  Re-running OCR to derive timestamps (no re-crop) …")
    cap = __import__('cv2').VideoCapture(video_path)
    clocks = []
    dict_seconds = {}
    prev_time_norm = None
    frame_idx = 0
    total = int(cap.get(__import__('cv2').CAP_PROP_FRAME_COUNT))

    def run_ocr(crop):
        import cv2
        g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        p = cv2.resize(g, None, fx=8, fy=8)
        p = cv2.GaussianBlur(p, (3,3), 0)
        _, p = cv2.threshold(p, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        raw = pytesseract.image_to_string(p, config="--psm 7 -c tessedit_char_whitelist=0123456789:")
        return normalize_ocr_time(raw)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % 3 == 0:
            crop1 = frame[1030:1050, 1810:1870]
            crop2 = frame[1030:1050, 1757:1817]
            norm = run_ocr(crop1) or run_ocr(crop2)
            if norm and norm != prev_time_norm:
                clocks.append((frame_idx, norm))
                dict_seconds[len(clocks)-1] = frame_idx
                prev_time_norm = norm
        frame_idx += 1
        if frame_idx % 500 == 0:
            log_fn(f"  OCR … {frame_idx}/{total} frames")

    cap.release()
    log_fn(f"  OCR complete: {len(clocks)} clock readings.")

    if equil_csv_path and os.path.exists(equil_csv_path):
        return func2(video_path, clocks, dict_seconds, participant_id, equil_csv_path, log_fn)
    return []