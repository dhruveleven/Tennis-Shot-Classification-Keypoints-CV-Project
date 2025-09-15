import os
import numpy as np
import cv2
from ultralytics import YOLO
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
from scipy.spatial import distance
import math

# Define paths
DATASET_PATH = "dataset"
IMAGES_PATH = os.path.join(DATASET_PATH, "images")
ANNOTATIONS_PATH = os.path.join(DATASET_PATH, "annotations")
OUTPUT_KEYPOINTS_PATH = os.path.join(DATASET_PATH, "keypoints")
MODEL_PATH = "trained_model.pkl"
OUTPUT_VIDEO_PATH = "output_video_with_predictions.mp4"

# Create output directory for keypoints
os.makedirs(OUTPUT_KEYPOINTS_PATH, exist_ok=True)

# Load YOLOv8m-pose model
model = YOLO('yolov8m-pose.pt')

def extract_keypoints_from_frame(frame):
    """Extract 17 keypoints from a single video frame using YOLOv8m-pose."""
    results = model(frame)
    
    # Get keypoints (17 keypoints, each with x, y, confidence)
    keypoints = results[0].keypoints.xy.cpu().numpy()
    if len(keypoints) > 0:
        return keypoints[0]  # Take the first detected person
    return np.zeros((17, 2))  # Return zeros if no person detected

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

def process_dataset():
    """Process images to extract keypoints and generate features for training."""
    X, y = [], []
    
    for shot_type in ['forehand', 'backhand']:
        image_dir = os.path.join(IMAGES_PATH, shot_type)
        for img_name in os.listdir(image_dir):
            img_path = os.path.join(image_dir, img_name)
            keypoint_path = os.path.join(OUTPUT_KEYPOINTS_PATH, f"{shot_type}_{img_name}.npy")
            
            # Read image and extract keypoints
            img = cv2.imread(img_path)
            keypoints = extract_keypoints_from_frame(img)
            save_keypoints(keypoints, keypoint_path)
            
            # Generate features
            features = generate_features(keypoints)
            X.append(features)
            y.append(1 if shot_type == 'backhand' else 0)
    
    return np.array(X), np.array(y)

def train_model(X, y):
    """Train Random Forest Classifier."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred, target_names=['Backhand', 'Forehand']))
    
    # Save model
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(clf, f)
    
    return clf

def overlay_prediction(frame, shot_type):
    """Overlay the predicted shot type on the frame."""
    # Define text properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (0, 255, 0)  # Green text
    thickness = 2
    position = (50, 50)  # Top-left corner
    
    # Add text to frame
    cv2.putText(frame, f"Shot: {shot_type}", position, font, font_scale, color, thickness, cv2.LINE_AA)
    return frame

def process_video(video_path, clf, output_video_path=OUTPUT_VIDEO_PATH, output_keypoints_dir="video_keypoints"):
    """Process a video, predict shot types, overlay predictions, and save output video."""
    # Create output directory for keypoints
    os.makedirs(output_keypoints_dir, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing video with {frame_count} frames, resolution {frame_width}x{frame_height}, {fps} fps...")
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    predictions = []
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Extract keypoints from frame
        keypoints = extract_keypoints_from_frame(frame)
        
        # Save keypoints
        keypoint_path = os.path.join(output_keypoints_dir, f"frame_{frame_idx:06d}.npy")
        save_keypoints(keypoints, keypoint_path)
        
        # Generate features and predict
        features = generate_features(keypoints)
        prediction = clf.predict([features])[0]
        shot_type = 'Forehand' if prediction == 1 else 'Backhand'
        
        # Overlay prediction on frame
        frame = overlay_prediction(frame, shot_type)
        
        # Write frame to output video
        out.write(frame)
        
        predictions.append((frame_idx, shot_type))
        
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx} frames...")
    
    # Release resources
    cap.release()
    out.release()
    
    # Save predictions to a text file
    output_txt = os.path.join(output_keypoints_dir, "predictions.txt")
    with open(output_txt, 'w') as f:
        for frame_idx, shot_type in predictions:
            f.write(f"Frame {frame_idx}: {shot_type}\n")
    
    print(f"Output video saved to {output_video_path}")
    print(f"Predictions saved to {output_txt}")
    return predictions

def main():
    # Process dataset and train model
    X, y = process_dataset()
    clf = train_model(X, y)
    
    # Process video
    video_path = "input_video.mp4"  # Replace with actual video path
    if os.path.exists(video_path):
        predictions = process_video(video_path, clf)
        print("Video processing complete. Sample predictions:")
        for frame_idx, shot_type in predictions[:5]:  # Show first 5 predictions
            print(f"Frame {frame_idx}: {shot_type}")
    else:
        print("Video file not found. Please provide a valid video path.")

if __name__ == "__main__":
    main()