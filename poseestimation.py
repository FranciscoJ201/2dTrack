import numpy as np 
import json 
from ultralytics import YOLO 
import os
import re

def sanitize_filename(name):
    """Remove or replace characters that are unsafe in filenames / zsh."""
    name = name.replace(' ', '_')
    name = re.sub(r'[^\w\-.]', '', name)   # keep letters, digits, _, -, .
    return name

def poseestimate(source):
    model = YOLO('yolov8n-pose.pt') 

    sor = source
    results = model.track(
        source=sor, 
        tracker='botsort.yaml', 
        show=True, 
        conf=0.3, 
        save=False 
    )
    base_name = os.path.basename(sor)
    video_name, _ = os.path.splitext(base_name)
    video_name = sanitize_filename(video_name)   # <-- sanitize here

    all_detection_data = []

    for i, result in enumerate(results):
        
        if result.keypoints.data.numel() == 0 or result.boxes.data.numel() == 0:
            continue

        track_ids = result.boxes.id
        keypoints_tensor = result.keypoints.data
        box_data = result.boxes.data.cpu().numpy() 

        if track_ids is None:
            track_ids = [-1] * len(keypoints_tensor)
        else:
            track_ids = track_ids.cpu().numpy().astype(int).tolist()

        for j, keypoint_data in enumerate(keypoints_tensor):
            
            track_id = track_ids[j] if j < len(track_ids) else -1
            box_xywh = result.boxes.xywh[j].cpu().numpy().round(1).tolist()
            
            confidence = 0.0 
            if box_data.shape[1] > 4:
                confidence = float(box_data[j, 4])
            else:
                confidence = 1.0 

            keypoints_array = keypoint_data.cpu().numpy().tolist()

            detection_record = {
                "frame_index": i,
                "track_id_native": track_id,
                "bbox_xywh": box_xywh,
                "conf": confidence,
                "keypoints_xyz": keypoints_array 
            }
            all_detection_data.append(detection_record)

    output_file = f'{video_name}_pose_detection.json'
    with open(output_file, 'w') as f:
        json.dump(all_detection_data, f, indent=4) 
    print(f"\nData extraction complete. Saved {len(all_detection_data)} detections to {output_file}")
    return output_file