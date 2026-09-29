<div align="center">

# 3D Markerless Gait Analysis

**Clinical gait metrics from a single depth camera, no markers**

Detects a subject with YOLOv8, tracks 133 keypoints with MMPose, reconstructs
them in 3D from RealSense depth, and extracts sagittal-plane hip, knee and ankle
kinematics — with Kalman/RTS smoothing and automatic PDF reporting.

`Python` · `YOLOv8` · `MMPose` · `RealSense D435i` · `Kalman/RTS` · `Modbus TCP`

**Semester project — ENISo, Mechatronics Department, 2025–2026**

</div>

---

## Results

| Metric | Value |
|:--|:--|
| Keypoints tracked | 133 (MMPose HRNet) |
| Joint angles | hip, knee, ankle — sagittal plane |
| Following distance | 3.0 m, closed loop over Modbus TCP |
| Smoothing | Kalman filter + RTS smoother |
| Output | PDF report, gait curves, CSV |

Sample outputs — gait curves, joint angle plots and generated reports — are in
[`Output example/`](Output%20example).

---

## Method

**1 · Detection and keypoints**
YOLOv8 locates the subject; MMPose (HRNet) extracts 133 body keypoints from the
RGB frame.

**2 · 3D reconstruction**
Keypoints are projected into 3D using the pinhole camera model and the
RealSense depth stream aligned to colour.

**3 · Sagittal alignment**
The 3D skeleton is rotated to align with the walking direction, so joint angles
are measured in the anatomical sagittal plane regardless of camera placement.

**4 · Filtering**
A Rauch–Tung–Striebel smoother removes depth sensor noise from joint
trajectories — necessary because raw depth jitter dominates the angle signal.

**5 · Gait normalisation**
Steps are segmented and normalised to a 0–100 % gait cycle for standard
clinical comparison.

**6 · Autonomous following**
A Modbus TCP controller keeps the robot at 3.0 m so the subject stays in frame
throughout the walk.

---

## Usage

Requires an Intel RealSense D435i on USB 3.0 and, for the following mode, a
Modbus TCP-capable base. Set the robot IP in `robot/follow_controller.py`.

```bash
python main.py
```

| Key | Action |
|:--|:--|
| `r` | Start recording — only if a subject is detected |
| `s` | Stop recording and run the analysis |

Results are written to `curves/`: PDF report, gait graphs and CSV data.

---

## Setup

```bash
git clone https://github.com/yahyabenturkia/Gait-Analysis-3D.git
cd Gait-Analysis-3D

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

A CUDA-capable GPU is recommended for real-time MMPose inference. Install
PyTorch matching your CUDA version from
[pytorch.org](https://pytorch.org/get-started/locally/).

---

## Repository
```
camera/ RealSense alignment and acquisition
vision/ 3D reconstruction, RTS smoothing, biomechanical maths
robot/ Modbus following controller
utils/ YOLOv8 person detection
visualisation/ gait curve and ROM plotting
exporter/ PDF report and CSV generation
Output example/ sample reports, curves and plots
main.py supervisor script
```

---

## Development

I designed and implemented the vision and analysis pipeline: 3D keypoint
reconstruction, sagittal-plane alignment, the Kalman/RTS smoothing stage, the
gait segmentation and biomechanical calculations, the Modbus following
controller, and the reporting output.

Semester project at ENISo, supervised by **Dr. Lamine Houssein** (PhD in
Robotics), with **Yasmine Saad** on the team.

---

## Author

**Yahya Ben Turkia** — Mechatronics engineer, industrial vision and embedded perception

[LinkedIn](https://www.linkedin.com/in/yahya-ben-turkia/) · yahya.benturkiya@gmail.com
