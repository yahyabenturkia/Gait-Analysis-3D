# 🏃‍♂️ 3D Markerless Gait Analysis & Autonomous Following Robot



A high-precision biomechanical analysis system that follows a subject and extracts clinical gait metrics using 3D skeletal reconstruction and depth sensing.



Developed by engineering students at **ENISO** (National School of Engineers of Sousse).



🎓 **Project by:** [Yahya Ben Turkia](https://github.com/yahya-bt), [Yasmine Saad](https://github.com/yasmine-saad)



🧑‍🏫 Supervised by: Dr. Lamine Houssein — PhD in Robotics, Assistant Professor at ENISO



---



## 🎯 Objective



Design and implement a system capable of:

- **Markerless Detection:** Identifying a person using **YOLOv8** and tracking 133 keypoints via **MMPose (HRNet)**.

- **3D Reconstruction:** Converting 2D vision data into 3D coordinates using **Intel RealSense D435i** depth alignment.

- **Biomechanical Analysis:** Transforming data into the **Sagittal Plane** to calculate Hip, Knee, and Ankle angles.

- **Signal Integrity:** Applying **Kalman Filtering** and **RTS Smoothing** for clinical-grade data.

- **Autonomous Following:** Maintaining a **safe distance (3.0m)** via **Modbus TCP** commands.



---



## 🛠️ Technologies Used



- 📷 **Intel RealSense D435i** — RGB + Depth Camera

- 🧍‍♂️ **MMPose & YOLOv8** — Advanced 2D/3D Pose Estimation

- 🧠 **Python 3.10** with:

&nbsp; - `ultralytics` (YOLOv8)

&nbsp; - `mmpose` (Skeletal Tracking)

&nbsp; - `filterpy` (Kalman/RTS Smoothing)

&nbsp; - `pyrealsense2` (Camera API)

&nbsp; - `pyModbusTCP` (Robot Control)

&nbsp; - `fpdf` (Clinical Report Generation)

- ⚙️ **Modbus TCP** — Industrial protocol for robot communication



---



## 🏗️ Project Structure



```text

├── camera/

│   └── camera.py                # RealSense alignment & acquisition

├── vision/

│   ├── pose_estimator.py        # 3D reconstruction & RTS Smoothing

│   └── GaitAnalyzer.py          # Biomechanical math & segmentation

├── robot/

│   └── follow_controller.py     # Modbus-based PID distance control

├── utils/

│   └── utils.py                 # YOLOv8 Person detection logic

├── visualisation/

│   └── visualizer.py            # Gait curve & ROM plotting

├── exporter/

│   └── exporter.py              # PDF Report & CSV generation

├── main.py                      # Main supervisor script

├── environment.yml              # Conda environment setup

└── requirements.txt             # Pip dependencies

```

### 🚀 Installation & Setup

1. Set up Conda Environment

This project requires CUDA 12.1 for high-speed pose estimation:

```

conda env create -f environment.yml

conda activate gait_env_new

```

2. Install Pose Engine (MMPose) :

```

pip install -U openmim

mim install mmengine

mim install "mmcv>=2.1.0"

mim install "mmpose>=1.3.2"

```



### 🎮 How to Use

⚠️ Prerequisite: EduBot Connection

Ensure the EduBot is powered on and connected to the same network as your workstation.



Robot IP: Ensure the IP in robot/follow_controller.py matches the EduBot's Modbus server address.



Camera: Connect the Intel RealSense D435i via USB 3.0.



**Execution Steps:**



1. Launch System:

```

python main.py

```

2. Recording Logic:



'r' (Start): Begins recording ONLY if a person is detected by the vision system.



's' (Stop): Ends recording and immediately triggers offline analysis.



3. View Results: A new folder will be created in curves/ containing your PDF Report, gait graphs, and CSV data.



### 📊 Methodology

The system follows a clinical workflow:



1. Pose Projection: Mapping 2D points to 3D space using the Pinhole Camera model.



2. Sagittal Alignment: Rotating the 3D skeleton to align with the walking direction.



3. Filtering: Using the Rauch-Tung-Striebel (RTS) smoother to remove depth noise.



4. Gait Normalization: Segmenting steps into a 0-100% phase for standard clinical comparison.



### 🎓 Academic Context

**Institution:** National School of Engineers of Sousse (ENISO)



**Major:** Mechatronics Engineering (Méca 3.1)



**Project Type:** Semester Project 2025-2026


