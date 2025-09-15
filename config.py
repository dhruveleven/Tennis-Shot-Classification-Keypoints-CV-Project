import os

# Define paths
DATASET_PATH = "dataset"
IMAGES_PATH = os.path.join(DATASET_PATH, "images")
ANNOTATIONS_PATH = os.path.join(DATASET_PATH, "annotations")
OUTPUT_KEYPOINTS_PATH = os.path.join(DATASET_PATH, "keypoints")
MODEL_PATH = "trained_model1.pkl"
OUTPUT_VIDEO_PATH = "output_video_with_predictions.mp4"
VIDEO_KEYPOINTS_DIR = "video_keypoints"