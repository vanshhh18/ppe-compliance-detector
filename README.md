# 🦺 PPE Compliance Detection System

A computer vision system that detects personal protective equipment (PPE) on construction site workers and flags safety violations in real time. Built with YOLOv8 and a custom rule-based compliance engine.

## Problem

Workplace safety compliance monitoring is a real, recurring cost and risk for construction and industrial sites — missing PPE (helmets, vests, gloves, boots, goggles) is a leading cause of preventable workplace injuries. This project automates PPE detection from images/video and flags non-compliant workers.

## Demo

![Detection Result](outputs/annotated_image1003.jpg)

*Detected PPE items with bounding boxes, alongside a per-worker compliance report. You can see more results inside output folder*

## Approach

1. **Object Detection** — Fine-tuned a YOLOv8m model on a construction-site PPE dataset to detect: `helmet`, `gloves`, `vest`, `boots`, `goggles`, and `Person`.
2. **Class imbalance handling** — The original dataset also included direct "missing-item" labels (e.g., `no_helmet`, `no_boots`), but several of these had as few as 4-56 training instances — too few to learn reliably. These were removed from training after diagnosing the imbalance via per-class mAP analysis.
3. **Rule-based compliance engine** — Instead of relying on unreliable minority-class predictions, missing PPE is inferred geometrically: each detected `Person` bounding box is divided into regions (head, torso, hands, feet), and the engine checks whether the corresponding PPE item overlaps that region. If not, it's flagged as a violation.
4. **Interactive demo** — A Streamlit app for uploading images/video and viewing detection + compliance results live.

## Results

| Class | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|
| helmet | 0.79 | 0.43 |
| gloves | 0.77 | 0.38 |
| vest | 0.82 | 0.49 |
| boots | 0.79 | 0.43 |
| goggles | 0.76 | 0.36 |
| Person | 0.89 | 0.48 |
| **Overall** | **0.80** | **0.43** |

## Tech Stack

- **Model:** YOLOv8m (Ultralytics), transfer learning
- **Training:** Google Colab (T4 GPU), 832px resolution, class-imbalance-aware augmentation (mosaic, HSV jitter, rotation)
- **Compliance Logic:** Custom Python geometric bounding-box overlap engine
- **Frontend:** Streamlit
- **Language:** Python

## Project Structure

```
ppe-detection-project/
├── models/
│   └── best.pt              # trained YOLOv8m weights
├── data/
│   └── test_images/         # sample test images
├── outputs/                 # sample annotated demo outputs
├── compliance_check.py      # core detection + compliance logic
├── app.py                   # Streamlit interactive demo
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone https://github.com/vanshhh18/ppe-detection.git
cd ppe-detection

python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate # Mac/Linux

pip install -r requirements.txt
```

## Usage

**Run compliance check on test images:**
```bash
python compliance_check.py
```

**Launch the interactive Streamlit demo:**
```bash
streamlit run app.py
```

## Known Limitations

- The rule-based compliance engine assumes a roughly upright pose; accuracy degrades for crouching, angled, or partially-occluded workers, since region checks are based on fixed percentage zones of the person's bounding box.
- Classes with limited training data (e.g., goggles) have moderately lower precision than well-represented classes.
- Video processing runs at reduced frame sampling (every 5th frame) for performance on CPU-only inference.

## Future Improvements

- Replace fixed-region compliance logic with pose-estimation-based keypoint matching for better robustness to worker posture.
- Expand training data for underrepresented classes.
- Deploy as a FastAPI inference endpoint for integration with live camera feeds.

## License

MIT
