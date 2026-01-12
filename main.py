import numpy as np 
import json 
from ultralytics import YOLO 
import os
import tkinter as tk
from tkinter import filedialog
from poseestimation import poseestimate
from plot import PoseViewer2D

def select_file_and_estimate():
    # 1. Set up the hidden Tkinter root window
    root = tk.Tk()
    root.withdraw()  # Prevents a blank window from staying open
    root.attributes('-topmost', True)  # Brings the file dialog to the front

    # 2. Open the file explorer
    # You can adjust filetypes to include .mp4, .png, etc.
    file_path = filedialog.askopenfilename(
        title="Select Input for Pose Estimation",
        filetypes=[("Image/Video files", "*.jpg *.jpeg *.png *.mp4 *.avi"), ("All files", "*.*")]
    )

    # 3. Check if a file was actually selected
    if file_path:
        print(f"File selected: {file_path}")
        
        # 4. Pass the file_path into your poseestimate function
        # (Assuming poseestimate takes the path as its first argument)
        results = poseestimate(file_path)
        
        return results
    else:
        print("No file selected.")
        return None

if __name__ == "__main__":
    result = select_file_and_estimate()
    player = PoseViewer2D(result)
    player.run()