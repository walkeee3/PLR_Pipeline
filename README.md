# Pupil Analysis Pipeline — Setup & Usage

## Folder structure
Place everything in one working directory:

```
working_dir/
├── pipeline_gui.py          ← main GUI (run this)
├── crop_videos_lib.py       ← crop & OCR helper
├── inference_pupil.py       ← your inference script
├── pupil_analysis_v10.py    ← your analysis script
│
├── Videos/                  ← DROP INPUT VIDEOS HERE
├── Videos_left/             ← auto-created (cropped left eye)
├── Videos_right/            ← auto-created (cropped right eye)
├── Time_Markings/           ← put equiluminance CSV(s) here
├── Models/
│   ├── left.pth
│   └── right.pth
├── Timestamps/              ← put timestamp CSV(s) here
└── Output/                  ← results appear here
```

## Install dependencies (WSL / Linux)
```bash
sudo apt update && sudo apt install -y tesseract-ocr python3-tk
pip install opencv-python pytesseract pandas numpy scipy matplotlib torch torchvision tqdm pillow
```

## Run
```bash
python pipeline_gui.py
```

## Workflow
1. Drop `.mp4` (or `.avi`/`.mov`/`.mkv`) files into `Videos/`
2. Click **⟳ Refresh** — videos will appear in the left panel
3. Click **▶ Process All**

For each video the pipeline will:
- Extract the participant ID from the filename (e.g. `GS_F_08_046-1`)
- Crop the video into left and right eye streams
- Run inference with the matching model
- Run pupil analysis and save plots
- **Pause and show a confirmation bar** — click **📂 Open Folder** to
  review the plots, then either:
  - **✓ Confirm & Next** — move to the next eye / next participant
  - **⚑ Mark as Bad** — rename the output folder to `<ID>_marked`
    and continue

## ID extraction rule
Filenames are expected to start with the ID pattern:
```
XX_X_NN_NNN-N   e.g.  GS_F_08_046-1
```
followed by any suffix, e.g. `GS_F_08_046-1_session2.mp4`

## Output per participant
```
Output/
└── GS_F_08_046-1/
    ├── left/
    │   ├── GS_F_08_046-1_left_pred.csv
    │   ├── signal_raw.png
    │   ├── signal_pre.png
    │   ├── signed_mca.png
    │   ├── detection_windows_pre.png
    │   └── mca_mcv_results.csv
    └── right/
        └── (same structure)
```
