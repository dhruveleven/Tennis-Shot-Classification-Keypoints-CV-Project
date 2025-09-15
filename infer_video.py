import os
import cv2
import pickle
import numpy as np
from config import MODEL_PATH, OUTPUT_VIDEO_PATH, VIDEO_KEYPOINTS_DIR
from utils import extract_keypoints_from_frame, save_keypoints, generate_features, overlay_prediction, visualize_keypoints, visualize_distances, visualize_angles

def normalize_keypoints(keypoints, frame_width, frame_height):
    """Normalize keypoints to account for scale and position differences."""
    if keypoints is None or len(keypoints) == 0:
        return keypoints
    normalized_keypoints = keypoints.copy()
    # Normalize x, y coordinates to [0, 1] based on frame dimensions
    normalized_keypoints[:, 0] /= frame_width  # x-coordinate
    normalized_keypoints[:, 1] /= frame_height  # y-coordinate
    return normalized_keypoints

def count_shots_with_states(predictions, min_frames=5, max_follow_through_frames=15, shot_transition_threshold=10):
    """Count forehand and backhand shots using a state machine, suppressing follow-throughs and noise."""
    forehand_count = 0
    backhand_count = 0
    shot_log = []
    state_log = []
    rejected_sequences = []
    raw_forehand_count = 0
    raw_backhand_count = 0
    
    current_state = 'Ready'
    current_shot = None
    start_frame = None
    frame_count = 0
    last_shot_type = None
    
    for frame_idx, shot_type in predictions:
        if frame_count > 0 and shot_type != current_shot:
            if current_state == 'Shot' and frame_count >= min_frames:
                shot_log.append({
                    "type": current_shot,
                    "start_frame": start_frame,
                    "end_frame": frame_idx - 1,
                    "duration_frames": frame_count
                })
                if current_shot == "Forehand":
                    forehand_count += 1
                else:
                    backhand_count += 1
                last_shot_type = current_shot
            elif frame_count < min_frames and current_shot:
                rejected_sequences.append({
                    "type": current_shot,
                    "start_frame": start_frame,
                    "end_frame": frame_idx - 1,
                    "duration_frames": frame_count,
                    "reason": "Too short"
                })
            elif current_state == 'Follow-Through':
                rejected_sequences.append({
                    "type": current_shot,
                    "start_frame": start_frame,
                    "end_frame": frame_idx - 1,
                    "duration_frames": frame_count,
                    "reason": f"Follow-Through after {last_shot_type}"
                })
        
        if current_state == 'Ready':
            if shot_type in ["Forehand", "Backhand"]:
                current_state = 'Shot'
                current_shot = shot_type
                start_frame = frame_idx
                frame_count = 1
            else:
                frame_count = 0
                start_frame = None
        elif current_state == 'Shot':
            if shot_type == current_shot:
                frame_count += 1
            elif shot_type == "Ready":
                current_state = 'Ready'
                current_shot = None
                frame_count = 0
                start_frame = None
            else:
                required_frames = shot_transition_threshold if last_shot_type is None else min_frames
                if frame_count >= required_frames:
                    shot_log.append({
                        "type": current_shot,
                        "start_frame": start_frame,
                        "end_frame": frame_idx - 1,
                        "duration_frames": frame_count
                    })
                    if current_shot == "Forehand":
                        forehand_count += 1
                    else:
                        backhand_count += 1
                    last_shot_type = current_shot
                current_state = 'Follow-Through'
                current_shot = shot_type
                start_frame = frame_idx
                frame_count = 1
        elif current_state == 'Follow-Through':
            if shot_type == "Ready" or frame_count >= max_follow_through_frames:
                current_state = 'Ready'
                current_shot = None
                frame_count = 0
                start_frame = None
            elif shot_type == last_shot_type:
                current_state = 'Shot'
                current_shot = shot_type
                start_frame = frame_idx
                frame_count = 1
            else:
                frame_count += 1
        
        if shot_type == "Forehand" and frame_count == 1:
            raw_forehand_count += 1
        elif shot_type == "Backhand" and frame_count == 1:
            raw_backhand_count += 1
        
        state_log.append((frame_idx, shot_type, current_state, last_shot_type))
    
    if current_state == 'Shot' and current_shot in ["Forehand", "Backhand"] and frame_count >= min_frames:
        shot_log.append({
            "type": current_shot,
            "start_frame": start_frame,
            "end_frame": predictions[-1][0],
            "duration_frames": frame_count
        })
        if current_shot == "Forehand":
            forehand_count += 1
        else:
            backhand_count += 1
    elif current_state == 'Shot' and frame_count < min_frames and current_shot:
        rejected_sequences.append({
            "type": current_shot,
            "start_frame": start_frame,
            "end_frame": predictions[-1][0],
            "duration_frames": frame_count,
            "reason": "Too short"
        })
    elif current_state == 'Follow-Through':
        rejected_sequences.append({
            "type": current_shot,
            "start_frame": start_frame,
            "end_frame": predictions[-1][0],
            "duration_frames": frame_count,
            "reason": f"Follow-Through after {last_shot_type}"
        })
    
    return forehand_count, backhand_count, shot_log, state_log, rejected_sequences, raw_forehand_count, raw_backhand_count

def process_video(video_path, clf, output_video_path=OUTPUT_VIDEO_PATH, output_keypoints_dir=VIDEO_KEYPOINTS_DIR, min_frames=5, max_follow_through_frames=15, shot_transition_threshold=10):
    """Process a video, predict shot types for near and far players, count shots, and overlay counts."""
    os.makedirs(output_keypoints_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing video with {frame_count} frames, resolution {frame_width}x{frame_height}, {fps} fps...")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    near_predictions = []
    far_predictions = []
    near_forehand_count = 0
    near_backhand_count = 0
    far_forehand_count = 0
    far_backhand_count = 0
    near_state = 'Ready'
    far_state = 'Ready'
    near_shot = None
    far_shot = None
    near_shot_start_frame = None
    far_shot_start_frame = None
    near_shot_frame_count = 0
    far_shot_frame_count = 0
    near_last_shot_type = None
    far_last_shot_type = None
    last_shot_player = None
    
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Extract keypoints for all players
        keypoints_list = extract_keypoints_from_frame(frame)
        near_keypoints = None
        far_keypoints = None
        
        # Segment players by court position (y-coordinate)
        for keypoints in keypoints_list:
            if keypoints is None or len(keypoints) == 0:
                continue
            avg_y = np.mean(keypoints[:, 1]) if len(keypoints) > 0 else 0
            if avg_y > frame_height / 2:  # Bottom half (near player)
                near_keypoints = keypoints
            else:  # Top half (far player)
                far_keypoints = keypoints
        
        # Save keypoints
        if near_keypoints is not None:
            near_keypoint_path = os.path.join(output_keypoints_dir, f"near_frame_{frame_idx:06d}.npy")
            save_keypoints(near_keypoints, near_keypoint_path)
        if far_keypoints is not None:
            far_keypoint_path = os.path.join(output_keypoints_dir, f"far_frame_{frame_idx:06d}.npy")
            save_keypoints(far_keypoints, far_keypoint_path)
        
        # Process near player
        near_shot_type = "Ready"
        if near_keypoints is not None:
            near_features = generate_features(near_keypoints)
            near_raw_prediction = clf.predict([near_features])[0]
            near_shot_type = 'Forehand' if near_raw_prediction == 1 else 'Backhand' if near_raw_prediction == 0 else 'Ready'
            
            # State machine for near player
            if near_shot_frame_count > 0 and near_shot_type != near_shot:
                if near_state == 'Shot' and near_shot_frame_count >= min_frames and (last_shot_player != 'near' or frame_idx - near_shot_start_frame >= shot_transition_threshold):
                    if near_shot == "Forehand":
                        near_forehand_count += 1
                    else:
                        near_backhand_count += 1
                    near_last_shot_type = near_shot
                    last_shot_player = 'near'
            
            if near_state == 'Ready':
                if near_shot_type in ["Forehand", "Backhand"] and (last_shot_player != 'far' or frame_idx - far_shot_start_frame >= shot_transition_threshold):
                    near_state = 'Shot'
                    near_shot = near_shot_type
                    near_shot_start_frame = frame_idx
                    near_shot_frame_count = 1
                else:
                    near_shot_frame_count = 0
                    near_shot_start_frame = None
            elif near_state == 'Shot':
                if near_shot_type == near_shot:
                    near_shot_frame_count += 1
                elif near_shot_type == "Ready":
                    near_state = 'Ready'
                    near_shot = None
                    near_shot_frame_count = 0
                    near_shoticina_frame = None
                else:
                    required_frames = shot_transition_threshold if near_last_shot_type is None else min_frames
                    if near_shot_frame_count >= required_frames and (last_shot_player != 'near' or frame_idx - near_shot_start_frame >= shot_transition_threshold):
                        if near_shot == "Forehand":
                            near_forehand_count += 1
                        else:
                            near_backhand_count += 1
                        near_last_shot_type = near_shot
                        last_shot_player = 'near'
                    near_state = 'Follow-Through'
                    near_shot = near_shot_type
                    near_shot_start_frame = frame_idx
                    near_shot_frame_count = 1
            elif near_state == 'Follow-Through':
                if near_shot_type == "Ready" or near_shot_frame_count >= max_follow_through_frames:
                    near_state = 'Ready'
                    near_shot = None
                    near_shot_frame_count = 0
                    near_shot_start_frame = None
                elif near_shot_type == near_last_shot_type:
                    near_state = 'Shot'
                    near_shot = near_shot_type
                    near_shot_start_frame = frame_idx
                    near_shot_frame_count = 1
                else:
                    near_shot_frame_count += 1
            
            near_predictions.append((frame_idx, near_shot_type, near_raw_prediction))
        
        # Process far player
        far_shot_type = "Ready"
        if far_keypoints is not None:
            far_keypoints_normalized = normalize_keypoints(far_keypoints, frame_width, frame_height)
            far_features = generate_features(far_keypoints_normalized)
            far_raw_prediction = clf.predict([far_features])[0]
            far_shot_type = 'Forehand' if far_raw_prediction == 1 else 'Backhand' if far_raw_prediction == 0 else 'Ready'
            
            # State machine for far player
            if far_shot_frame_count > 0 and far_shot_type != far_shot:
                if far_state == 'Shot' and far_shot_frame_count >= min_frames and (last_shot_player != 'far' or frame_idx - far_shot_start_frame >= shot_transition_threshold):
                    if far_shot == "Forehand":
                        far_forehand_count += 1
                    else:
                        far_backhand_count += 1
                    far_last_shot_type = far_shot
                    last_shot_player = 'far'
            
            if far_state == 'Ready':
                if far_shot_type in ["Forehand", "Backhand"] and (last_shot_player != 'near' or frame_idx - near_shot_start_frame >= shot_transition_threshold):
                    far_state = 'Shot'
                    far_shot = far_shot_type
                    far_shot_start_frame = frame_idx
                    far_shot_frame_count = 1
                else:
                    far_shot_frame_count = 0
                    far_shot_start_frame = None
            elif far_state == 'Shot':
                if far_shot_type == far_shot:
                    far_shot_frame_count += 1
                elif far_shot_type == "Ready":
                    far_state = 'Ready'
                    far_shot = None
                    far_shot_frame_count = 0
                    far_shot_start_frame = None
                else:
                    required_frames = shot_transition_threshold if far_last_shot_type is None else min_frames
                    if far_shot_frame_count >= required_frames and (last_shot_player != 'far' or frame_idx - far_shot_start_frame >= shot_transition_threshold):
                        if far_shot == "Forehand":
                            far_forehand_count += 1
                        else:
                            far_backhand_count += 1
                        far_last_shot_type = far_shot
                        last_shot_player = 'far'
                    far_state = 'Follow-Through'
                    far_shot = far_shot_type
                    far_shot_start_frame = frame_idx
                    far_shot_frame_count = 1
            elif far_state == 'Follow-Through':
                if far_shot_type == "Ready" or far_shot_frame_count >= max_follow_through_frames:
                    far_state = 'Ready'
                    far_shot = None
                    far_shot_frame_count = 0
                    far_shot_start_frame = None
                elif far_shot_type == far_last_shot_type:
                    far_state = 'Shot'
                    far_shot = far_shot_type
                    far_shot_start_frame = frame_idx
                    far_shot_frame_count = 1
                else:
                    far_shot_frame_count += 1
            
            far_predictions.append((frame_idx, far_shot_type, far_raw_prediction))
        
        # Visualize keypoints for both players
        if near_keypoints is not None:
            frame = visualize_keypoints(frame, near_keypoints)
            frame = visualize_distances(frame, near_keypoints)
            frame = visualize_angles(frame, near_keypoints)
        if far_keypoints is not None:
            frame = visualize_keypoints(frame, far_keypoints)
            frame = visualize_distances(frame, far_keypoints)
            frame = visualize_angles(frame, far_keypoints)
        
        # Overlay predictions and counts
        frame = overlay_prediction(frame, near_shot_type, near_forehand_count, near_backhand_count, position="bottom")
        frame = overlay_prediction(frame, far_shot_type, far_forehand_count, far_backhand_count, position="top")
        
        out.write(frame)
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx} frames...")
    
    # Handle last shots
    if near_state == 'Shot' and near_shot in ["Forehand", "Backhand"] and near_shot_frame_count >= min_frames and (last_shot_player != 'near' or frame_idx - near_shot_start_frame >= shot_transition_threshold):
        if near_shot == "Forehand":
            near_forehand_count += 1
        else:
            near_backhand_count += 1
    if far_state == 'Shot' and far_shot in ["Forehand", "Backhand"] and far_shot_frame_count >= min_frames and (last_shot_player != 'far' or frame_idx - far_shot_start_frame >= shot_transition_threshold):
        if far_shot == "Forehand":
            far_forehand_count += 1
        else:
            far_backhand_count += 1
    
    cap.release()
    out.release()
    
    # Count shots and generate logs for both players
    near_results = count_shots_with_states(
        [(frame_idx, shot_type) for frame_idx, shot_type, _ in near_predictions], min_frames, max_follow_through_frames, shot_transition_threshold
    ) if near_predictions else (0, 0, [], [], [], 0, 0)
    far_results = count_shots_with_states(
        [(frame_idx, shot_type) for frame_idx, shot_type, _ in far_predictions], min_frames, max_follow_through_frames, shot_transition_threshold
    ) if far_predictions else (0, 0, [], [], [], 0, 0)
    
    near_final_forehand_count, near_final_backhand_count, near_shot_log, near_state_log, near_rejected_sequences, near_raw_forehand_count, near_raw_backhand_count = near_results
    far_final_forehand_count, far_final_backhand_count, far_shot_log, far_state_log, far_rejected_sequences, far_raw_forehand_count, far_raw_backhand_count = far_results
    
    output_txt = os.path.join(output_keypoints_dir, "predictions.txt")
    with open(output_txt, 'w') as f:
        f.write("Near Player Predictions:\n")
        for frame_idx, shot_type, raw_prediction in near_predictions:
            f.write(f"Frame {frame_idx}: Raw Prediction={raw_prediction}, Shot Type={shot_type}\n")
        f.write("\nFar Player Predictions:\n")
        for frame_idx, shot_type, raw_prediction in far_predictions:
            f.write(f"Frame {frame_idx}: Raw Prediction={raw_prediction}, Shot Type={shot_type}\n")
        f.write("\nNear Player State Log:\n")
        for frame_idx, shot_type, state, last_shot in near_state_log:
            last_shot_str = last_shot if last_shot else "None"
            f.write(f"Frame {frame_idx}: {shot_type}, State: {state}, Last Shot: {last_shot_str}\n")
        f.write("\nFar Player State Log:\n")
        for frame_idx, shot_type, state, last_shot in far_state_log:
            last_shot_str = last_shot if last_shot else "None"
            f.write(f"Frame {frame_idx}: {shot_type}, State: {state}, Last Shot: {last_shot_str}\n")
        f.write("\nNear Player Shot Log:\n")
        for shot in near_shot_log:
            f.write(f"{shot['type']} Shot, Start Frame: {shot['start_frame']}, End Frame: {shot['end_frame']}, Duration: {shot['duration_frames']} frames\n")
        f.write("\nFar Player Shot Log:\n")
        for shot in far_shot_log:
            f.write(f"{shot['type']} Shot, Start Frame: {shot['start_frame']}, End Frame: {shot['end_frame']}, Duration: {shot['duration_frames']} frames\n")
        f.write("\nNear Player Rejected Sequences:\n")
        for seq in near_rejected_sequences:
            f.write(f"{seq['type']} Sequence, Start Frame: {seq['start_frame']}, End Frame: {seq['end_frame']}, Duration: {seq['duration_frames']} frames, Reason: {seq['reason']}\n")
        f.write("\nFar Player Rejected Sequences:\n")
        for seq in far_rejected_sequences:
            f.write(f"{seq['type']} Sequence, Start Frame: {seq['start_frame']}, End Frame: {seq['end_frame']}, Duration: {seq['duration_frames']} frames, Reason: {seq['reason']}\n")
        f.write("\nNear Player Raw Counts (before state machine):\n")
        f.write(f"Raw Forehand Shots: {near_raw_forehand_count}\n")
        f.write(f"Raw Backhand Shots: {near_raw_backhand_count}\n")
        f.write("\nFar Player Raw Counts (before state machine):\n")
        f.write(f"Raw Forehand Shots: {far_raw_forehand_count}\n")
        f.write(f"Raw Backhand Shots: {far_raw_backhand_count}\n")
        f.write("\nNear Player Final Counts:\n")
        f.write(f"Total Forehand Shots: {near_final_forehand_count}\n")
        f.write(f"Total Backhand Shots: {near_final_backhand_count}\n")
        f.write("\nFar Player Final Counts:\n")
        f.write(f"Total Forehand Shots: {far_final_forehand_count}\n")
        f.write(f"Total Backhand Shots: {far_final_backhand_count}\n")
    
    print(f"Output video saved to {output_video_path}")
    print(f"Predictions and shot counts saved to {output_txt}")
    print(f"Near Player Raw Counts: Forehand={near_raw_forehand_count}, Backhand={near_raw_backhand_count}")
    print(f"Near Player Final Counts: Forehand={near_final_forehand_count}, Backhand={near_final_backhand_count}")
    print(f"Far Player Raw Counts: Forehand={far_raw_forehand_count}, Backhand={far_raw_backhand_count}")
    print(f"Far Player Final Counts: Forehand={far_final_forehand_count}, Backhand={far_final_backhand_count}")
    print("Video processing complete. Sample predictions and states:")
    for i, ((n_frame_idx, n_shot_type, n_raw_prediction), (f_frame_idx, f_shot_type, f_raw_prediction)) in enumerate(zip(near_predictions[:5], far_predictions[:5])):
        n_state = near_state_log[i][2] if i < len(near_state_log) else "N/A"
        f_state = far_state_log[i][2] if i < len(far_state_log) else "N/A"
        n_last_shot = near_state_log[i][3] if i < len(near_state_log) else None
        f_last_shot = far_state_log[i][3] if i < len(far_state_log) else None
        n_last_shot_str = n_last_shot if n_last_shot else "None"
        f_last_shot_str = f_last_shot if f_last_shot else "None"
        print(f"Frame {n_frame_idx}: Near: Raw={n_raw_prediction}, Shot={n_shot_type}, State={n_state}, Last={n_last_shot_str} | Far: Raw={f_raw_prediction}, Shot={f_shot_type}, State={f_state}, Last={f_last_shot_str}")
    print(f"Near Player Shot Log:")
    for shot in near_shot_log:
        print(f"{shot['type']} Shot, Start Frame: {shot['start_frame']}, End Frame: {shot['end_frame']}, Duration: {shot['duration_frames']} frames")
    print(f"Far Player Shot Log:")
    for shot in far_shot_log:
        print(f"{shot['type']} Shot, Start Frame: {shot['start_frame']}, End Frame: {shot['end_frame']}, Duration: {shot['duration_frames']} frames")
    print(f"Near Player Rejected Sequences:")
    for seq in near_rejected_sequences[:5]:
        print(f"{seq['type']} Sequence, Start Frame: {seq['start_frame']}, End Frame: {seq['end_frame']}, Duration: {seq['duration_frames']} frames, Reason: {seq['reason']}")
    print(f"Far Player Rejected Sequences:")
    for seq in far_rejected_sequences[:5]:
        print(f"{seq['type']} Sequence, Start Frame: {seq['start_frame']}, End Frame: {seq['end_frame']}, Duration: {seq['duration_frames']} frames, Reason: {seq['reason']}")
    
    return (near_predictions, near_final_forehand_count, near_final_backhand_count, near_shot_log, near_state_log, near_rejected_sequences,
            far_predictions, far_final_forehand_count, far_final_backhand_count, far_shot_log, far_state_log, far_rejected_sequences)

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Trained model not found at {MODEL_PATH}. Please run train.py first.")
        return
    
    with open(MODEL_PATH, 'rb') as f:
        clf = pickle.load(f)
    
    video_path = "input videos/input_video2.mp4"
    if os.path.exists(video_path):
        results = process_video(video_path, clf)
    else:
        print("Video file not found. Please provide a valid video path.")

if __name__ == "__main__":
    main()