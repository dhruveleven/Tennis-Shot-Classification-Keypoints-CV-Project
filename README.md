# 🎾 Tennis Shot Classification using Pose Estimation  

This project focuses on **automated tennis shot classification** by leveraging **pose estimation** and **machine learning**.  
Using **YOLO11m-pose**, we detect player keypoints from each video frame, extract features, and classify whether the player performs a **forehand**, **backhand**, or is in the **ready position**.  
The system also overlays shot counts directly on the output video.  

---

## 🚀 Project Overview  
- **Input:** Raw tennis match/practice videos  
- **Processing Pipeline:**  
  1. Detect player keypoints per frame using **YOLO11m-pose**  
  2. Extract motion and handedness features from detected keypoints  
  3. Train a **Random Forest classifier** on these features to classify shots  
  4. Run inference on new videos, overlaying predicted actions and cumulative counts  
- **Output:** Video with bounding boxes, pose skeletons, shot classification labels, and counters (forehand / backhand)  

---

## 📂 Dataset  
We use the **Tennis Player Actions Dataset for Human Pose Estimation**:  

- Source: [Mendeley Data](https://data.mendeley.com/datasets/nv3rpsxhhk/1)  
- DOI: [10.17632/nv3rpsxhhk.1](https://doi.org/10.17632/nv3rpsxhhk.1)    
---

## 🛠️ Methodology  

### 1. **Pose Estimation**  
- Model: [YOLO11m-pose](https://github.com/ultralytics/ultralytics)  
- Detects **keypoints** (e.g., wrists, elbows, shoulders, hips) from each frame  

### 2. **Feature Engineering**  
- Compute **joint angles** and **relative positions**  
- Identify **handedness** of the player  
- Temporal features extracted to capture motion dynamics  

### 3. **Classification**  
- Classifier: **Random Forest**  
- Classes:  
  - **Forehand**  
  - **Backhand**  
  - **Ready Position**  

### 4. **Inference & Visualization**  
- Run classifier predictions on input video  
- Overlay:  
  - Pose skeletons  
  - Shot label (Forehand / Backhand / Ready)  
  - **Counters** for forehands and backhands  

---

## 📊 Results  
- Real-time video with annotated poses and shot labels  
- Forehand & backhand counts displayed as live overlays  
- Heatmap-like visualization of action frequency (optional extension)  

---

## 🔧 Tech Stack  
- **Python**  
- **YOLO11m-pose** (Ultralytics)  
- **OpenCV** for video processing  
- **Scikit-learn** (Random Forest)  
- **Matplotlib / Seaborn** for visualization  


**Tennis Player Actions Dataset for Human Pose Estimation**  
Chun-Yi Wang, Kalin Guanlun Lai, Hsu-Chun Huang, Wei-Ting Lin  
Mendeley Data, V1, 2024  
DOI: [10.17632/nv3rpsxhhk.1](https://doi.org/10.17632/nv3rpsxhhk.1)  

---

## Screenshots
<img width="443" height="231" alt="Screenshot 2025-09-17 124457" src="https://github.com/user-attachments/assets/40b8e59f-d7d0-405b-b767-2c36ebd39360" />

<img width="446" height="229" alt="Screenshot 2025-09-17 124646" src="https://github.com/user-attachments/assets/a00c59fe-2fc8-4a6c-9e85-ad721ceb1a3d" />

