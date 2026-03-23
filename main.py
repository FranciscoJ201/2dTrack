import numpy as np 
import json 
from ultralytics import YOLO 
import os
import tkinter as tk
from tkinter import filedialog
from poseestimation import poseestimate, sanitize_filename
from plot import PoseViewer2D

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

        player = PoseViewer2D(json_path, video_name=video_name)
        player.run()