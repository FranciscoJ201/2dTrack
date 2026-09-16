import cv2
import numpy as np
import time
import os
from insightface.app import FaceAnalysis

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
VIDEO_PATH = "/Users/franciscojimenez/Downloads/WhatsApp Image 2026-06-15 at 10.17.47 PM.jpeg"
OUTPUT_PATH = "/Users/franciscojimenez/Desktop/2dTrack/tracked_output.png" # <-- Destination for saved file
DETECTION_SIZE = 640                              

def main():
    print("Loading lightweight ONNX face alignment models...")
    app = FaceAnalysis(allowed_modules=['detection', 'landmark_2d_106'])
    app.prepare(ctx_id=-1, det_size=(DETECTION_SIZE, DETECTION_SIZE))
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video file at: {VIDEO_PATH}")
        return

    # Gather original video specs to ensure output attributes match perfectly
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_source = cap.get(cv2.CAP_PROP_FPS)
    
    # Fallback to standard 30 FPS if metadata reading fails
    if fps_source <= 0:
        fps_source = 30.0

    print(f"Source video dimensions: {frame_width}x{frame_height} @ {fps_source} FPS")
    
    # Initialize the video saving framework
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps_source, (frame_width, frame_height))
    
    p_time = 0
    print(f"Tracking active. Saving video to: {OUTPUT_PATH}")
    print("Press 'q' in the window to stop tracking early.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Video playback completed or frame unreadable.")
            break

        faces = app.get(frame)

        for face in faces:
            landmarks = face.landmark_2d_106
            if landmarks is not None:
                for pt in landmarks:
                    x, y = int(pt[0]), int(pt[1])
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

                jaw_points = landmarks[0:33].astype(np.int32)
                cv2.polylines(frame, [jaw_points], isClosed=False, color=(255, 255, 0), thickness=1)

        # Calculate live script execution performance for the HUD overlay
        c_time = time.time()
        fps = 1 / (c_time - p_time) if (c_time - p_time) > 0 else 0
        p_time = c_time
        cv2.putText(frame, f'FPS: {int(fps)}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Write the annotated frame straight to the file stream
        out.write(frame)

        cv2.imshow("Grant Demo - 106 Pt Dense Face Tracker", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Tracking stopped early by user.")
            break

    # Clean up and close all writing handlers safely
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Video file saved successfully!")

if __name__ == "__main__":
    main()