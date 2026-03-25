import matplotlib
matplotlib.use('TkAgg') # Force TkAgg backend to prevent macOS trace trap errors

import numpy as np 
import json 
from ultralytics import YOLO 
import os
import tkinter as tk
from tkinter import filedialog
import cv2 # Added to extract native video FPS
from poseestimation import poseestimate, sanitize_filename
from plot import PoseViewer2D

def get_video_fps(file_path):
    """
    Extracts the frames per second (FPS) from a video file to ensure synced playback.
    
    Arguments:
    - file_path (str): The absolute or relative path to the input video or image file.
    
    Returns:
    - float: The FPS of the video. Returns 15.0 as a fallback if the file is an image or unreadable.
    """
    cap = cv2.VideoCapture(file_path)
    fps = 15.0 # Default fallback
    if cap.isOpened():
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps > 0:
            fps = video_fps
        cap.release()
        print(fps)
    return fps

def select_file_and_estimate():
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    file_path = filedialog.askopenfilename(
        title="Select Input for Pose Estimation",
        filetypes=[("Image/Video files", "*.jpg *.jpeg *.png *.mp4 *.avi"), ("All files", "*.*")]
    )

    if file_path:
        print(f"File selected: {file_path}")
        results = poseestimate(file_path)
        return file_path, results
    else:
        print("No file selected.")
        return None, None

if __name__ == "__main__":
    file_path, json_path = select_file_and_estimate()
    if json_path:
        # Derive the clean video name (same sanitization as poseestimation.py)
        base_name  = os.path.basename(file_path)
        video_name = sanitize_filename(os.path.splitext(base_name)[0])
        
        # Extract the exact FPS from the original video
        original_fps = get_video_fps(file_path)
        print(f"Detected original FPS: {original_fps}")

        # Pass the extracted original_fps to the player
        player = PoseViewer2D(json_path, fps=original_fps, video_name=video_name)
        
        # --- VIDEO SAVING TOGGLE ---
        SAVE_PLOT_VIDEO = False
        output_vid_path = f"{video_name}_plot_output.mp4"
        
        player.run(save_video=SAVE_PLOT_VIDEO, output_path=output_vid_path)