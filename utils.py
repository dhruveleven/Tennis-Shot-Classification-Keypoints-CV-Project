
import numpy as np
import cv2
from ultralytics import YOLO
from scipy.spatial import distance
import math

# Load YOLOv11m-pose model
try:
    model = YOLO('yolo11m-pose.pt')
except FileNotFoundError:
    print("Error: 'yolo11m-pose.pt' not found. Downloading model or falling back to YOLOv8m-pose.")
    try:
        model = YOLO('yolo11m-pose.pt')  # Retry download
    except Exception as e:
        print(f"Failed to download yolo11m-pose.pt: {e}. Using yolov8m-pose.pt instead.")
        model = YOLO('yolov8m-pose.pt')

def extract_keypoints_from_frame(frame):
    """Extract 17 keypoints for all detected people in a frame using YOLO model."""
    results = model(frame)
    
    # Get keypoints for all detected people (17 keypoints, each with x, y)
    keypoints_list = results[0].keypoints.xy.cpu().numpy()
    if len(keypoints_list) == 0:
        return []  # Return empty list if no people detected
    
    # Convert to list of keypoint arrays (x, y, no confidence for inference)
    return [keypoints for keypoints in keypoints_list]

def extract_keypoints(image_path):
    """Extract 17 keypoints from an image using YOLO model."""
    img = cv2.imread(image_path)
    keypoints_list = extract_keypoints_from_frame(img)
    if len(keypoints_list) > 0:
        return keypoints_list[0]  # Return first person's keypoints for training
    return np.zeros((17, 2))

def save_keypoints(keypoints, output_path):
    """Save keypoints as .npy file."""
    np.save(output_path, keypoints)

def calculate_angle(p1, p2, p3):
    """Calculate angle between three points."""
    a = np.array(p1)
    b = np.array(p2)
    c = np.array(p3)
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def generate_features(keypoints):
    """Generate features from keypoints (distances and angles)."""
    features = []
    
    # Keypoint indices (OpenPose format)
    nose = 0
    left_shoulder = 5
    right_shoulder = 2
    left_elbow = 6
    right_elbow = 3
    left_wrist = 7
    right_wrist = 4
    left_hip = 11
    right_hip = 8
    
    # Distances
    features.append(distance.euclidean(keypoints[left_shoulder], keypoints[right_shoulder]))
    features.append(distance.euclidean(keypoints[left_elbow], keypoints[left_wrist]))
    features.append(distance.euclidean(keypoints[right_elbow], keypoints[right_wrist]))
    features.append(distance.euclidean(keypoints[left_shoulder], keypoints[left_hip]))
    features.append(distance.euclidean(keypoints[right_shoulder], keypoints[right_hip]))
    
    # Angles
    features.append(calculate_angle(keypoints[left_shoulder], keypoints[left_elbow], keypoints[left_wrist]))
    features.append(calculate_angle(keypoints[right_shoulder], keypoints[right_elbow], keypoints[right_wrist]))
    features.append(calculate_angle(keypoints[left_hip], keypoints[left_shoulder], keypoints[left_elbow]))
    features.append(calculate_angle(keypoints[right_hip], keypoints[right_shoulder], keypoints[right_elbow]))
    
    return np.array(features)

def overlay_prediction(frame, shot_type, forehand_count=0, backhand_count=0, position="bottom"):
    """Overlay the predicted shot type and shot counts on the frame."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (0, 255, 0)  # Green text
    thickness = 2
    
    # Determine y-coordinates based on position
    if position == "top":
        shot_y = 50
        forehand_y = 80
        backhand_y = 110
        prefix = "Far: "
    else:  # bottom
        shot_y = frame.shape[0] - 90
        forehand_y = frame.shape[0] - 60
        backhand_y = frame.shape[0] - 30
        prefix = "Near: "
    
    # Overlay shot type and counts
    cv2.putText(frame, f"{prefix}Shot: {shot_type}", (50, shot_y), font, font_scale, color, thickness, cv2.LINE_AA)
    cv2.putText(frame, f"{prefix}Forehand: {forehand_count}", (50, forehand_y), font, font_scale, color, thickness, cv2.LINE_AA)
    cv2.putText(frame, f"{prefix}Backhand: {backhand_count}", (50, backhand_y), font, font_scale, color, thickness, cv2.LINE_AA)
    return frame

def visualize_keypoints(frame, keypoints):
    """Draw keypoints as circles on the frame."""
    keypoint_color = (0, 0, 255)  # Red for keypoints
    radius = 5
    thickness = -1  # Filled circle
    
    for kp in keypoints:
        x, y = int(kp[0]), int(kp[1])
        if x > 0 and y > 0:  # Only draw valid keypoints
            cv2.circle(frame, (x, y), radius, keypoint_color, thickness)
    
    return frame

def visualize_distances(frame, keypoints):
    """Draw lines for distances and annotate with values."""
    # Keypoint indices
    left_shoulder = 5
    right_shoulder = 2
    left_elbow = 6
    right_elbow = 3
    left_wrist = 7
    right_wrist = 4
    left_hip = 11
    right_hip = 8
    
    # Distance pairs and colors
    distance_pairs = [
        (left_shoulder, right_shoulder),
        (left_elbow, left_wrist),
        (right_elbow, right_wrist),
        (left_shoulder, left_hip),
        (right_shoulder, right_hip)
    ]
    line_color = (255, 0, 0)  # Blue lines
    text_color = (255, 255, 255)  # White text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 2
    
    for idx, (kp1_idx, kp2_idx) in enumerate(distance_pairs):
        kp1 = keypoints[kp1_idx]
        kp2 = keypoints[kp2_idx]
        if kp1[0] > 0 and kp1[1] > 0 and kp2[0] > 0 and kp2[1] > 0:  # Valid keypoints
            x1, y1 = int(kp1[0]), int(kp1[1])
            x2, y2 = int(kp2[0]), int(kp2[1])
            # Draw line
            cv2.line(frame, (x1, y1), (x2, y2), line_color, thickness)
            # Calculate and annotate distance
            dist = distance.euclidean(kp1, kp2)
            mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.putText(frame, f"D{idx+1}: {dist:.1f}", (mid_x, mid_y), font, font_scale, text_color, 1, cv2.LINE_AA)
    
    return frame

def visualize_angles(frame, keypoints):
    """Draw arcs for angles and annotate with values."""
    # Keypoint indices
    left_shoulder = 5
    right_shoulder = 2
    left_elbow = 6
    right_elbow = 3
    left_wrist = 7
    right_wrist = 4
    left_hip = 11
    right_hip = 8
    
    # Angle triplets
    angle_triplets = [
        (left_shoulder, left_elbow, left_wrist),
        (right_shoulder, right_elbow, right_wrist),
        (left_hip, left_shoulder, left_elbow),
        (right_hip, right_shoulder, right_elbow)
    ]
    arc_color = (0, 255, 255)  # Yellow arcs
    text_color = (255, 255, 255)  # White text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 2
    
    for idx, (kp1_idx, kp2_idx, kp3_idx) in enumerate(angle_triplets):
        kp1 = keypoints[kp1_idx]
        kp2 = keypoints[kp2_idx]
        kp3 = keypoints[kp3_idx]
        if kp1[0] > 0 and kp1[1] > 0 and kp2[0] > 0 and kp2[1] > 0 and kp3[0] > 0 and kp3[1] > 0:
            x1, y1 = int(kp1[0]), int(kp1[1])
            x2, y2 = int(kp2[0]), int(kp2[1])
            x3, y3 = int(kp3[0]), int(kp3[1])
            # Calculate angle
            angle = calculate_angle(kp1, kp2, kp3)
            # Draw lines to indicate angle
            cv2.line(frame, (x1, y1), (x2, y2), arc_color, 1)
            cv2.line(frame, (x2, y2), (x3, y3), arc_color, 1)
            # Annotate angle
            cv2.putText(frame, f"A{idx+1}: {angle:.1f}°", (x2, y2 + 20), font, font_scale, text_color, 1, cv2.LINE_AA)
    
    return frame