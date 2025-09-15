import os
import cv2
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from config import IMAGES_PATH, MODEL_PATH
from utils import extract_keypoints, generate_features

def load_data(data_dir):
    """Load images, extract keypoints, generate features, and assign labels."""
    classes = {'righty_forehand': 1, 'righty_backhand': 0, 'ready': 2}
    features = []
    labels = []
    
    for class_name, label in classes.items():
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: Directory {class_dir} not found.")
            continue
        
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Failed to load image {img_path}")
                continue
            
            keypoints = extract_keypoints(img_path)
            if keypoints is None or np.all(keypoints == 0):
                print(f"Warning: No keypoints detected for {img_path}")
                continue
            
            feature_vector = generate_features(keypoints)
            features.append(feature_vector)
            labels.append(label)
    
    return np.array(features), np.array(labels)

def main():
    if not os.path.exists(IMAGES_PATH):
        print(f"Error: Dataset directory {IMAGES_PATH} not found.")
        return
    
    print("Loading data...")
    X, y = load_data(IMAGES_PATH)
    if len(X) == 0 or len(y) == 0:
        print("Error: No data loaded. Check dataset directories and images.")
        return
    
    print(f"Loaded {len(X)} samples with {X.shape[1]} features.")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    print("Training classifier...")
    clf.fit(X_train, y_train)
    
    # Evaluate classifier
    y_pred = clf.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Backhand', 'Forehand', 'Ready']))
    
    # Save model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(clf, f)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()