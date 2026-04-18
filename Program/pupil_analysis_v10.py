"""
Pupil Diameter Analysis Script  –  v10
======================================
Usage:
    python pupil_analysis_v10.py

You will be prompted for:
  - Participant ID  (e.g. GS_F_08_046-1)
  - Side            (left / right)

Expects files in the same directory as this script:
  - <ID>_<side>.csv        → predicted diameter   (frame, timestamp_sec, pred_diameter_px)
  - <ID>_<side>_gt.csv     → ground truth         (Frame, Valid, diameter_px, confidence, …)
  - timestamps.csv          → stimulus frame nums  (first col = ID, rest = frame numbers)

Outputs saved to  ./output_<ID>_<side>_v10/ :
  - raw_pred.png                 – Predicted raw signal, all stimuli marked
  - raw_gt.png                   – GT raw signal, all stimuli marked
  - preprocessed_pred.png        – Predicted after preprocessing (with background colors)
  - preprocessed_gt.png          – GT after preprocessing (with background colors)
  - mca_mcv_pred.png             – Unsigned MCA & MCV bar charts for Pred (raw vs pre)
  - mca_mcv_gt.png               – Unsigned MCA & MCV bar charts for GT  (raw vs pre)
  - signed_mca_pred.png          – Signed MCA for Pred (+constriction / −dilation)
  - signed_mca_gt.png            – Signed MCA for GT   (+constriction / −dilation)
  - detection_windows_pred_raw.png   – Per-stimulus window with a, b, and MCV slope marked
  - detection_windows_pred_pre.png   – Per-stimulus window with a, b, and MCV slope marked
  - detection_windows_gt_raw.png     – Per-stimulus window with a, b, and MCV slope marked
  - detection_windows_gt_pre.png     – Per-stimulus window with a, b, and MCV slope marked
  - mca_mcv_results.csv          – All numeric results (MCA is signed)
"""

import os
import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.signal import medfilt
from scipy import signal


# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
THRESHOLD1 = 100
THRESHOLD4 = 1.5

PRED_COLOR  = "#2196F3"
GT_COLOR    = "#FF5722"
STIM_COLOR  = "#4CAF50"
CONS_COLOR  = "#1565C0"
DIL_COLOR   = "#E53935"

# Detection-window annotation colours
PT_A_COLOR    = "#1B5E20"   # dark green  — point a  (transition start)
PT_B_COLOR    = "#B71C1C"   # dark red    — point b  (transition end / MCA end)
MCV_COLOR     = "#FF6F00"   # amber       — MCV 5-frame max slope line
WINDOW_COLOR  = "#E3F2FD"   # light blue window background


# ─────────────────────────────────────────────────────────────────────────────
#  PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def interpolate_array(array, confidence):
    interpolated_array = []
    last_high_confidence_index = 0
    for i in range(len(array)):
        if confidence[i] >= 0.95:
            last_high_confidence_index = i
            interpolated_array.append(array[i])
        else:
            j = last_high_confidence_index
            k = i + 1
            while k < len(array) and confidence[k] < 0.9:
                k += 1
            if j >= 0 and k < len(array):
                interpolated_array.append(
                    array[j] + (array[k] - array[j]) / (k - j))
            else:
                interpolated_array.append(array[i])
    return interpolated_array


def smoothen(data):
    b, a = signal.butter(5, 0.1, 'low')
    return signal.filtfilt(b, a, data)


def preprocess(diameter_array, confidence_array):
    arr = interpolate_array(list(diameter_array), list(confidence_array))
    arr = medfilt(arr)
    arr = smoothen(arr)
    return np.array(arr, dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
#  CORE DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def get_derivatives_local(window_array):
    n = len(window_array)
    if n < 2:
        return np.array([], dtype=int)
    interp_idx = np.linspace(0, n - 1, n * 10)
    interp_arr = np.interp(interp_idx, np.arange(n), window_array)
    grad = np.gradient(interp_arr)
    sign_changes = np.where(np.diff(np.sign(grad)))[0]
    return np.round(interp_idx[sign_changes + 1]).astype(int)


def find_closest_element(arr, target):
    arr = arr.tolist()
    if not arr:
        return None
    low, high = 0, len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    if low > len(arr) - 1:
        return high
    if high < 0:
        return low
    return low if abs(arr[low] - target) < abs(arr[high] - target) else high


def _find_transition_local(window, derivatives, direction, threshold2):
    try:
        if direction == 'drop':
            a_local = int(np.argmax(window))
            b_local = int(np.argmin(window))
            while a_local > 0 and window[a_local] <= window[a_local - 1]:
                a_local -= 1
            while b_local < len(window) - 2 and window[b_local] >= window[b_local + 1]:
                b_local += 1
        else:
            a_local = int(np.argmin(window))
            b_local = int(np.argmax(window))
            while a_local > 0 and window[a_local] >= window[a_local - 1]:
                a_local -= 1
            while b_local < len(window) - 2 and window[b_local] <= window[b_local + 1]:
                b_local += 1

        n = len(window)
        if a_local > 0 and len(derivatives) >= 1:
            idx_a  = find_closest_element(derivatives, a_local)
            a_local = derivatives[idx_a]
        else:
            idx_a = None          

        if b_local < n - 1 and len(derivatives) >= 1:
            idx_b  = find_closest_element(derivatives, b_local)
            b_local = derivatives[idx_b]
        else:
            idx_b = None          

        if idx_a is not None:
            idx_b_limit = idx_b if idx_b is not None else len(derivatives)
            while (idx_a + 2 < len(derivatives) and
                   idx_a + 2 < idx_b_limit and
                   abs(window[derivatives[idx_a + 2]] - window[a_local]) < THRESHOLD4):
                idx_a  += 2
                a_local = derivatives[idx_a]

        if idx_b is not None:
            idx_a_limit = idx_a if idx_a is not None else -1
            while (idx_b - 2 >= 0 and
                   idx_b - 2 > idx_a_limit and
                   abs(window[derivatives[idx_b - 2]] - window[b_local]) < THRESHOLD4):
                idx_b  -= 2
                b_local = derivatives[idx_b]

        diff = b_local - a_local
        amp  = (window[a_local] - window[b_local] if direction == 'drop'
                else window[b_local] - window[a_local])

        if not (0 < diff < THRESHOLD1 and amp > threshold2):
            return -1, 0
        return a_local, diff

    except Exception:
        return -1, 0


def adaptive_threshold2(signal_array):
    valid = np.asarray(signal_array, dtype=float)
    valid = valid[valid > 0]
    if len(valid) == 0:
        return 1.0
    return max(0.02 * (valid.max() - valid.min()), 0.1)


# ─────────────────────────────────────────────────────────────────────────────
#  SIGNED DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def _detect_signed_with_points(window, derivatives, threshold2):
    for direction in ('drop', 'rise'):
        local_a, width = _find_transition_local(window, derivatives, direction, threshold2)
        if local_a >= 0:
            local_b  = local_a + width

            if direction == 'drop':
                raw_amp    = window[local_a] - window[local_b]
                signed_mca = float(raw_amp)
            else:
                raw_amp    = window[local_b] - window[local_a]
                signed_mca = float(-raw_amp)

            max_slope = -1.0
            best_k = local_a
            best_k_end = min(local_a + 5, len(window) - 1)
            
            for k in range(local_a, local_b):
                k_end = min(k + 5, len(window) - 1)
                if k_end > k:
                    slope = abs(window[k] - window[k_end]) / (k_end - k)
                    if slope > max_slope:
                        max_slope = slope
                        best_k = k
                        best_k_end = k_end
            
            if max_slope == -1.0:
                max_slope = float(abs(window[local_a] - window[best_k_end]) / 5)

            mcv = float(max_slope)
            return signed_mca, mcv, direction, local_a, local_b, best_k, best_k_end

    return "Not Found", "Not Found", None, None, None, None, None


# ─────────────────────────────────────────────────────────────────────────────
#  PER-SIGNAL WRAPPERS
# ─────────────────────────────────────────────────────────────────────────────

def _run_pred_window(full_signal, idx1, idx2, threshold2):
    idx1 = max(0, idx1)
    idx2 = min(len(full_signal), idx2)
    window = np.array(full_signal[idx1:idx2], dtype=float)

    if len(window) < 10:
        return "Not Found", "Not Found", None, None, None, None, None, window

    invalid = window <= 0
    if invalid.all():
        return "Not Found", "Not Found", None, None, None, None, None, window
    if invalid.any():
        xp = np.where(~invalid)[0]
        fp = window[~invalid]
        window[invalid] = np.interp(np.where(invalid)[0], xp, fp)

    derivatives = get_derivatives_local(window)
    if len(derivatives) < 2:
        return "Not Found", "Not Found", None, None, None, None, None, window

    mca, mcv, direction, la, lb, l_mcv_start, l_mcv_end = _detect_signed_with_points(
        window, derivatives, threshold2)

    ga          = idx1 + la          if la          is not None else None
    gb          = idx1 + lb          if lb          is not None else None
    g_mcv_start = idx1 + l_mcv_start if l_mcv_start is not None else None
    g_mcv_end   = idx1 + l_mcv_end   if l_mcv_end   is not None else None

    return mca, mcv, direction, ga, gb, g_mcv_start, g_mcv_end, window


def _run_gt_window(full_signal, idx1, idx2, threshold2):
    idx1 = max(0, idx1)
    idx2 = min(len(full_signal), idx2)
    window = np.array(full_signal[idx1:idx2], dtype=float)

    if len(window) < 10:
        return "Not Found", "Not Found", None, None, None, None, None, window

    if np.sum(window > 0) < len(window) * 0.5:
        return "Not Found", "Not Found", None, None, None, None, None, window

    derivatives = get_derivatives_local(window)
    if len(derivatives) < 2:
        return "Not Found", "Not Found", None, None, None, None, None, window

    mca, mcv, direction, la, lb, l_mcv_start, l_mcv_end = _detect_signed_with_points(
        window, derivatives, threshold2)

    ga          = idx1 + la          if la          is not None else None
    gb          = idx1 + lb          if lb          is not None else None
    g_mcv_start = idx1 + l_mcv_start if l_mcv_start is not None else None
    g_mcv_end   = idx1 + l_mcv_end   if l_mcv_end   is not None else None

    return mca, mcv, direction, ga, gb, g_mcv_start, g_mcv_end, window


# ─────────────────────────────────────────────────────────────────────────────
#  COMPUTE ALL STIMULI (UPDATED BASELINE LOGIC)
# ─────────────────────────────────────────────────────────────────────────────

def compute_all_mca_mcv(signal_array, timestamps, threshold2, signal_type='pred'):
    run_fn = _run_pred_window if signal_type == 'pred' else _run_gt_window

    mca_list, mcv_list, detail_list = [], [], []
    
    if not timestamps:
        return mca_list, mcv_list, detail_list

    # 1. Calculate Baseline completely independently
    first_ts = timestamps[0]
    base_start = max(0, first_ts - 100)
    base_slice = signal_array[base_start:first_ts]
    
    if len(base_slice) > 0:
        base = float(np.mean(base_slice))
    else:
        # Fallback if no frames exist before the first timestamp
        base = float(signal_array[0]) if len(signal_array) > 0 else 0.0
        
    mca_list.append(("baseline", base))
    mcv_list.append(("baseline", base))
    detail_list.append(("baseline", None, None, None, None, None, base_start, first_ts, None))

    # 2. Evaluate all timestamps as stimuli
    for i, index in enumerate(timestamps):
        idx1 = max(0, index - 20)
        idx2 = min(len(signal_array), index + 100)
        
        mca, mcv, direction, ga, gb, g_mcv_start, g_mcv_end, _ = run_fn(
            signal_array, idx1, idx2, threshold2)
            
        lbl = f"stim_{i+1}"
        mca_list.append((lbl, mca))
        mcv_list.append((lbl, mcv))
        detail_list.append((lbl, direction, ga, gb, g_mcv_start, g_mcv_end, idx1, idx2, mcv))

    return mca_list, mcv_list, detail_list


# ─────────────────────────────────────────────────────────────────────────────
#  PLOTTING — SIGNAL OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────

def save_signal_figure(signal_array, timestamps, title, color, label, out_path, bg_colors=None):
    fig, ax = plt.subplots(figsize=(16, 4))
    fig.suptitle(title, fontsize=12, fontweight="bold")
    
    if bg_colors:
        span_edges = [0] + timestamps + [len(signal_array)]
        for i in range(len(span_edges) - 1):
            start = span_edges[i]
            end = span_edges[i+1]
            c = bg_colors[i] if i < len(bg_colors) else "white"
            ax.axvspan(start, end, facecolor=c, alpha=0.15, zorder=0)

    ax.plot(np.arange(len(signal_array)), signal_array,
            color=color, linewidth=0.8, label=label, zorder=2)
            
    first = True
    for t in timestamps:
        ax.axvline(t, color=STIM_COLOR, linewidth=0.9, alpha=0.75,
                   linestyle="--",
                   label="Stimulus" if first else "_nolegend_", zorder=3)
        first = False
        
    ax.set_ylabel("Diameter (px)")
    ax.set_xlabel("Frame")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3, zorder=1)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  PLOTTING — UNSIGNED MCA/MCV BARS
# ─────────────────────────────────────────────────────────────────────────────

def save_mca_mcv_figure_single(mca_raw, mcv_raw, mca_pre, mcv_pre,
                                color, signal_label, out_path):
    labels = [lbl for lbl, _ in mca_raw if lbl != "baseline"]
    n = len(labels)
    x = np.arange(n)

    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    fig.suptitle(f"MCA & MCV — {signal_label}  (Raw vs Preprocessed)",
                 fontsize=13, fontweight="bold")

    panels = [
        (axes[0, 0], "MCA – Raw",          mca_raw, "MCA magnitude (px)"),
        (axes[0, 1], "MCV – Raw",           mcv_raw, "MCV (px/frame)"),
        (axes[1, 0], "MCA – Preprocessed",  mca_pre, "MCA magnitude (px)"),
        (axes[1, 1], "MCV – Preprocessed",  mcv_pre, "MCV (px/frame)"),
    ]

    for ax, title, data_list, ylabel in panels:
        vals = [abs(v) if isinstance(v, float) else np.nan
                for lbl, v in data_list if lbl != "baseline"]
        bars = ax.bar(x, vals, color=color, alpha=0.85,
                      edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"{val:.2f}", ha="center", va="bottom",
                        fontsize=6.5, color="black")
            else:
                ax.text(bar.get_x() + bar.get_width() / 2, 0.05,
                        "NF", ha="center", va="bottom",
                        fontsize=6.5, color="grey")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3, axis="y")
        found = sum(1 for v in vals if not np.isnan(v))
        ax.set_xlabel(f"Found: {found}/{n}", fontsize=8, color="dimgrey")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  PLOTTING — SIGNED MCA
# ─────────────────────────────────────────────────────────────────────────────

def save_signed_mca_figure(mca_raw, mca_pre, signal_label, signal_color, out_path):
    labels = [lbl for lbl, _ in mca_raw if lbl != "baseline"]
    n = len(labels)
    x = np.arange(n)

    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    fig.suptitle(f"Signed MCA — {signal_label}\n(+) Constriction  |  (−) Dilation",
                 fontsize=13, fontweight="bold")

    for ax, data_list, stage in [(axes[0], mca_raw, "Raw"),
                                  (axes[1], mca_pre, "Preprocessed")]:
        vals = [v if isinstance(v, float) else np.nan
                for lbl, v in data_list if lbl != "baseline"]
        bar_colors = [
            "lightgrey" if np.isnan(v) else CONS_COLOR if v >= 0 else DIL_COLOR
            for v in vals
        ]
        plot_vals = [v if not np.isnan(v) else 0.0 for v in vals]
        bars = ax.bar(x, plot_vals, color=bar_colors,
                      alpha=0.85, edgecolor="white", linewidth=0.5)

        for bar, val, pv in zip(bars, vals, plot_vals):
            if np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, 0.15,
                        "NF", ha="center", va="bottom", fontsize=6, color="grey")
            else:
                offset = 0.15 if pv >= 0 else -0.15
                va     = "bottom" if pv >= 0 else "top"
                ax.text(bar.get_x() + bar.get_width() / 2, pv + offset,
                        f"{val:.2f}", ha="center", va=va,
                        fontsize=6.5, color="black")

        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel("MCA (px)  [signed]")
        ax.set_title(stage, fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.25, axis="y")
        n_c  = sum(1 for v in vals if isinstance(v, float) and v > 0)
        n_d  = sum(1 for v in vals if isinstance(v, float) and v < 0)
        n_nf = sum(1 for v in vals if np.isnan(v))
        ax.set_xlabel(
            f"Constrictions: {n_c}  |  Dilations: {n_d}  |  Not Found: {n_nf}",
            fontsize=8, color="dimgrey")
        legend_els = [
            mpatches.Patch(facecolor=CONS_COLOR, alpha=0.85, label="Constriction (+)"),
            mpatches.Patch(facecolor=DIL_COLOR,  alpha=0.85, label="Dilation (−)"),
            mpatches.Patch(facecolor="lightgrey",             label="Not Found"),
        ]
        ax.legend(handles=legend_els, loc="upper right", fontsize=8)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  PLOTTING — DETECTION WINDOW INSPECTION
# ─────────────────────────────────────────────────────────────────────────────

def save_detection_windows_figure(signal_array, detail_list, mca_list,
                                  signal_color, signal_label, out_path):
    stim_details = [(lbl, d, m) for (lbl, *d), (_, m)
                    in zip(detail_list, mca_list) if lbl != "baseline"]

    entries = []
    for i in range(len(detail_list)):
        lbl, direction, ga, gb, g_mcv_start, g_mcv_end, idx1, idx2, mcv_val = detail_list[i]
        if lbl == "baseline":
            continue
        mca_val = mca_list[i][1]
        entries.append((lbl, direction, ga, gb, g_mcv_start, g_mcv_end, idx1, idx2, mca_val, mcv_val))

    n_stim = len(entries)
    if n_stim == 0:
        return

    ncols = 4
    nrows = math.ceil(n_stim / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 4.5, nrows * 3.2),
                             squeeze=False)
    fig.suptitle(
        f"Detection Windows — {signal_label}\n"
        f"● a (green) = transition start   ● b (red) = transition end   "
        f"● Line (amber) = 5-frame Max Slope (MCV)",
        fontsize=11, fontweight="bold"
    )

    legend_handles = [
        mpatches.Patch(color=PT_A_COLOR,  label="a  (MCA start)"),
        mpatches.Patch(color=PT_B_COLOR,  label="b  (MCA end)"),
        mpatches.Patch(color=MCV_COLOR,   label="MCV window (max slope)"),
    ]

    for idx, (lbl, direction, ga, gb, g_mcv_start, g_mcv_end, idx1, idx2, mca_val, mcv_val) in enumerate(entries):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        frames = np.arange(idx1, idx2)
        window_sig = signal_array[idx1:idx2]

        found = isinstance(mca_val, float)
        bg    = WINDOW_COLOR if found else "#FAFAFA"
        ax.set_facecolor(bg)

        ax.plot(frames, window_sig, color=signal_color,
                linewidth=1.0, zorder=2)

        if found:
            ax.axvline(ga, color=PT_A_COLOR, linewidth=1.5,
                       linestyle="-", zorder=3, alpha=0.9)
            ax.scatter([ga], [signal_array[ga]], color=PT_A_COLOR,
                       s=50, zorder=5, label="a")

            ax.axvline(gb, color=PT_B_COLOR, linewidth=1.5,
                       linestyle="-", zorder=3, alpha=0.9)
            ax.scatter([gb], [signal_array[gb]], color=PT_B_COLOR,
                       s=50, zorder=5, label="b")

            ya = signal_array[ga]
            yb = signal_array[gb]
            y_mid  = (ya + yb) / 2
            x_mid  = (ga + gb) / 2
            ax.annotate(
                "", xy=(gb, yb), xytext=(ga, ya),
                arrowprops=dict(arrowstyle="<->", color="dimgrey",
                                lw=1.2, shrinkA=0, shrinkB=0),
                zorder=4
            )
            sign_str = "+" if mca_val >= 0 else "−"
            dir_str  = "C" if direction == "drop" else "D"
            ax.text(x_mid + 1, y_mid,
                    f"MCA={sign_str}{abs(mca_val):.2f}\n({dir_str})",
                    fontsize=6, color="dimgrey",
                    va="center", ha="left", zorder=6)

            if g_mcv_start is not None and g_mcv_end is not None:
                ax.axvline(g_mcv_start, color=MCV_COLOR, linewidth=1.2, linestyle="--", zorder=3, alpha=0.7)
                ax.axvline(g_mcv_end, color=MCV_COLOR, linewidth=1.2, linestyle="--", zorder=3, alpha=0.7)
                
                ax.scatter([g_mcv_start, g_mcv_end], [signal_array[g_mcv_start], signal_array[g_mcv_end]],
                           color=MCV_COLOR, s=40, zorder=5, marker="D")
                
                ax.plot([g_mcv_start, g_mcv_end], [signal_array[g_mcv_start], signal_array[g_mcv_end]], 
                        color=MCV_COLOR, linewidth=2, zorder=4)

                y_mcv_mid = (signal_array[g_mcv_start] + signal_array[g_mcv_end]) / 2
                x_mcv_mid = (g_mcv_start + g_mcv_end) / 2
                
                ax.annotate(
                    f"MCV={mcv_val:.3f}",
                    xy=(x_mcv_mid, y_mcv_mid),
                    xytext=(x_mcv_mid + 2, y_mcv_mid + (signal_array[ga] - y_mcv_mid) * 0.6),
                    fontsize=5.5, color=MCV_COLOR,
                    arrowprops=dict(arrowstyle="->", color=MCV_COLOR,
                                    lw=0.8, shrinkA=0, shrinkB=2),
                    zorder=6
                )

        else:
            ax.text(0.5, 0.5, "Not Found",
                    transform=ax.transAxes,
                    ha="center", va="center",
                    fontsize=9, color="grey", style="italic")

        ax.set_title(lbl, fontsize=8, fontweight="bold",
                     color="black" if found else "grey")
        ax.set_xlabel("Frame", fontsize=6)
        ax.set_ylabel("Diameter (px)", fontsize=6)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.25)

    for idx in range(n_stim, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.legend(handles=legend_handles, loc="upper right",
               fontsize=8, framealpha=0.9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    participant_id = input("Enter Participant ID (e.g. GS_F_08_046-1): ").strip()
    side = input("Enter side (left / right): ").strip().lower()
    if side not in ("left", "right"):
        print("Side must be 'left' or 'right'.")
        sys.exit(1)

    base_dir  = os.path.dirname(os.path.abspath(__file__))
    pred_path = os.path.join(base_dir, f"{participant_id}_{side}.csv")
    gt_path   = os.path.join(base_dir, f"{participant_id}_{side}_gt.csv")
    ts_path   = os.path.join(base_dir, "timestamps.csv")

    for p in (pred_path, gt_path, ts_path):
        if not os.path.exists(p):
            print(f"ERROR: File not found: {p}")
            sys.exit(1)

    print("\nLoading data …")
    pred_df = pd.read_csv(pred_path)
    gt_df   = pd.read_csv(gt_path)
    ts_df   = pd.read_csv(ts_path, header=None)

    ts_row = ts_df[ts_df.iloc[:, 0] == participant_id]
    if ts_row.empty:
        print(f"ERROR: '{participant_id}' not found in timestamps.csv")
        sys.exit(1)
    timestamps = ts_row.iloc[0, 1:].dropna().astype(int).tolist()
    print(f"  {len(timestamps)} stimulus timestamps: {timestamps}")

    pred_raw  = pred_df["pred_diameter_px"].values.astype(float)
    gt_diam   = gt_df["diameter_px"].values.astype(float)
    gt_conf   = (gt_df["confidence"].values.astype(float)
                 if "confidence" in gt_df.columns else np.ones(len(gt_diam)))
    pred_conf = np.ones(len(pred_raw))

    n          = min(len(pred_raw), len(gt_diam))
    pred_raw   = pred_raw[:n];  pred_conf = pred_conf[:n]
    gt_diam    = gt_diam[:n];   gt_conf   = gt_conf[:n]
    timestamps = [t for t in timestamps if t < n]

    print("Preprocessing …")
    pred_pre = preprocess(pred_raw, pred_conf)
    gt_pre   = preprocess(gt_diam,  gt_conf)

    pred_t2     = adaptive_threshold2(pred_raw)
    gt_t2       = adaptive_threshold2(gt_diam)
    pred_pre_t2 = adaptive_threshold2(pred_pre)
    gt_pre_t2   = adaptive_threshold2(gt_pre)
    print(f"  Adaptive threshold2 →  pred_raw: {pred_t2:.3f}  gt_raw: {gt_t2:.3f}  "
          f"pred_pre: {pred_pre_t2:.3f}  gt_pre: {gt_pre_t2:.3f}")

    out_dir = os.path.join(base_dir, f"output_{participant_id}_{side}_v10")
    os.makedirs(out_dir, exist_ok=True)
    print(f"  Output folder: {out_dir}")

    bg_colors_seq = [
        "gray", "red", "gray", "green", "gray", "blue", "gray", "yellow", 
        "gray", "black", "gray", "white", "gray", "purple", "gray", "indigo"
    ]

    print("\nGenerating signal plots …")
    save_signal_figure(pred_raw, timestamps,
        f"{participant_id} ({side})  –  Predicted  [Raw]",
        PRED_COLOR, "Predicted",
        os.path.join(out_dir, "raw_pred.png"))
        
    save_signal_figure(gt_diam, timestamps,
        f"{participant_id} ({side})  –  Ground Truth  [Raw]",
        GT_COLOR, "Ground Truth",
        os.path.join(out_dir, "raw_gt.png"))
        
    save_signal_figure(pred_pre, timestamps,
        f"{participant_id} ({side})  –  Predicted  [Preprocessed]",
        PRED_COLOR, "Predicted (preprocessed)",
        os.path.join(out_dir, "preprocessed_pred.png"), bg_colors=bg_colors_seq)
        
    save_signal_figure(gt_pre, timestamps,
        f"{participant_id} ({side})  –  Ground Truth  [Preprocessed]",
        GT_COLOR, "Ground Truth (preprocessed)",
        os.path.join(out_dir, "preprocessed_gt.png"), bg_colors=bg_colors_seq)

    print("\nComputing MCA & MCV …")
    pred_raw_mca, pred_raw_mcv, pred_raw_detail = compute_all_mca_mcv(
        pred_raw, timestamps, pred_t2, signal_type='pred')
    pred_pre_mca, pred_pre_mcv, pred_pre_detail = compute_all_mca_mcv(
        pred_pre, timestamps, pred_pre_t2, signal_type='pred')
    gt_raw_mca, gt_raw_mcv, gt_raw_detail       = compute_all_mca_mcv(
        gt_diam, timestamps, gt_t2, signal_type='gt')
    gt_pre_mca, gt_pre_mcv, gt_pre_detail       = compute_all_mca_mcv(
        gt_pre, timestamps, gt_pre_t2, signal_type='gt')

    print("\nGenerating MCA/MCV bar charts …")
    save_mca_mcv_figure_single(
        pred_raw_mca, pred_raw_mcv, pred_pre_mca, pred_pre_mcv,
        PRED_COLOR, f"Predicted — {participant_id} ({side})",
        os.path.join(out_dir, "mca_mcv_pred.png"))
    save_mca_mcv_figure_single(
        gt_raw_mca, gt_raw_mcv, gt_pre_mca, gt_pre_mcv,
        GT_COLOR, f"Ground Truth — {participant_id} ({side})",
        os.path.join(out_dir, "mca_mcv_gt.png"))

    print("\nGenerating signed MCA plots …")
    save_signed_mca_figure(
        pred_raw_mca, pred_pre_mca,
        f"Predicted — {participant_id} ({side})",
        PRED_COLOR, os.path.join(out_dir, "signed_mca_pred.png"))
    save_signed_mca_figure(
        gt_raw_mca, gt_pre_mca,
        f"Ground Truth — {participant_id} ({side})",
        GT_COLOR, os.path.join(out_dir, "signed_mca_gt.png"))

    print("\nGenerating detection window plots …")
    save_detection_windows_figure(
        pred_raw, pred_raw_detail, pred_raw_mca,
        PRED_COLOR,
        f"Predicted Raw — {participant_id} ({side})",
        os.path.join(out_dir, "detection_windows_pred_raw.png"))
    save_detection_windows_figure(
        pred_pre, pred_pre_detail, pred_pre_mca,
        PRED_COLOR,
        f"Predicted Preprocessed — {participant_id} ({side})",
        os.path.join(out_dir, "detection_windows_pred_pre.png"))
    save_detection_windows_figure(
        gt_diam, gt_raw_detail, gt_raw_mca,
        GT_COLOR,
        f"Ground Truth Raw — {participant_id} ({side})",
        os.path.join(out_dir, "detection_windows_gt_raw.png"))
    save_detection_windows_figure(
        gt_pre, gt_pre_detail, gt_pre_mca,
        GT_COLOR,
        f"Ground Truth Preprocessed — {participant_id} ({side})",
        os.path.join(out_dir, "detection_windows_gt_pre.png"))

    rows = []
    for i, (label, _) in enumerate(pred_raw_mca):
        rows.append({
            "stimulus":     label,
            "pred_raw_mca": pred_raw_mca[i][1],
            "pred_raw_mcv": pred_raw_mcv[i][1],
            "pred_pre_mca": pred_pre_mca[i][1],
            "pred_pre_mcv": pred_pre_mcv[i][1],
            "gt_raw_mca":   gt_raw_mca[i][1],
            "gt_raw_mcv":   gt_raw_mcv[i][1],
            "gt_pre_mca":   gt_pre_mca[i][1],
            "gt_pre_mcv":   gt_pre_mcv[i][1],
        })
    csv_path = os.path.join(out_dir, "mca_mcv_results.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    def fmt(v):
        return f"{v:+8.3f}" if isinstance(v, float) else f"{'NF':>8}"

    data_map = {
        "pred_raw_mca": pred_raw_mca, "pred_pre_mca": pred_pre_mca,
        "pred_raw_mcv": pred_raw_mcv, "pred_pre_mcv": pred_pre_mcv,
        "gt_raw_mca":   gt_raw_mca,   "gt_pre_mca":   gt_pre_mca,
        "gt_raw_mcv":   gt_raw_mcv,   "gt_pre_mcv":   gt_pre_mcv,
    }

    for metric, keys in [
        ("MCA (signed: + constriction, − dilation)",
         ("pred_raw_mca", "pred_pre_mca", "gt_raw_mca", "gt_pre_mca")),
        ("MCV",
         ("pred_raw_mcv", "pred_pre_mcv", "gt_raw_mcv", "gt_pre_mcv")),
    ]:
        print(f"\n{'='*70}\n  {metric}\n{'─'*70}")
        print(f"{'Stimulus':<12}  {'Pred-Raw':>9}  {'Pred-Pre':>9}  "
              f"{'GT-Raw':>9}  {'GT-Pre':>9}")
        print("─" * 70)
        for i, (lbl, _) in enumerate(data_map[keys[0]]):
            vals = "  ".join(fmt(data_map[k][i][1]) for k in keys)
            print(f"{lbl:<12}  {vals}")

    # Notice the total length is now simply len(timestamps) instead of len - 1
    total = len(timestamps)
    for stage, mk, gk in [("Raw", "pred_raw_mca", "gt_raw_mca"),
                           ("Pre", "pred_pre_mca", "gt_pre_mca")]:
        pf  = sum(1 for l, v in data_map[mk] if l != "baseline" and isinstance(v, float))
        gf  = sum(1 for l, v in data_map[gk] if l != "baseline" and isinstance(v, float))
        pc  = sum(1 for l, v in data_map[mk] if l != "baseline" and isinstance(v, float) and v > 0)
        pd_ = sum(1 for l, v in data_map[mk] if l != "baseline" and isinstance(v, float) and v < 0)
        gc  = sum(1 for l, v in data_map[gk] if l != "baseline" and isinstance(v, float) and v > 0)
        gd  = sum(1 for l, v in data_map[gk] if l != "baseline" and isinstance(v, float) and v < 0)
        print(f"\n[{stage}]  Pred: {pf}/{total} found "
              f"({pc} constrictions, {pd_} dilations)   |   "
              f"GT: {gf}/{total} found ({gc} constrictions, {gd} dilations)")

    print(f"\nAll outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()