import cv2
import numpy as np
import os
from insightface.app import FaceAnalysis

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
IMAGE_PATH = "/Users/franciscojimenez/Downloads/WhatsApp Image 2026-06-15 at 10.17.47 PM.jpeg"
OUTPUT_PATH = "/Users/franciscojimenez/Desktop/2dTrack/tracked_output.png" # <-- Destination for saved file
DETECTION_SIZE = 640                              

def main():
    print("Loading lightweight ONNX face alignment models...")
    # Arguments: allowed_modules takes a list of string features to extract
    app = FaceAnalysis(allowed_modules=['detection', 'landmark_2d_106'])
    # Arguments: ctx_id=-1 forces CPU execution; det_size sets the face detection image scaling
    app.prepare(ctx_id=-1, det_size=(DETECTION_SIZE, DETECTION_SIZE))
    
    # Load the static image
    # Arguments: filename path to target image
    frame = cv2.imread(IMAGE_PATH)
    if frame is None:
        print(f"Error: Could not open image file at: {IMAGE_PATH}")
        return

    print(f"Source image dimensions: {frame.shape[1]}x{frame.shape[0]}")
    print(f"Tracking active. Processing image...")

    # Extract face analytics from the single frame
    # Arguments: img is the source BGR numpy array
    faces = app.get(frame)

    for face in faces:
        landmarks = face.landmark_2d_106
        if landmarks is not None:
            # Render individual landmark points
            for pt in landmarks:
                x, y = int(pt[0]), int(pt[1])
                # Arguments: img, center coordinates, radius, color, thickness (-1 fills the circle)
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            # Isolate and draw the jawline continuity path
            jaw_points = landmarks[0:33].astype(np.int32)
            # Arguments: img, array of shapes, closed polygon flag, color, thickness
            cv2.polylines(frame, [jaw_points], isClosed=False, color=(255, 255, 0), thickness=1)

    # Ensure output directory exists before writing
    out_dir = os.path.dirname(OUTPUT_PATH)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Save the annotated frame straight to the file stream
    # Arguments: filename destination path, img data array
    cv2.imwrite(OUTPUT_PATH, frame)

    # Display the result in a GUI window
    # Arguments: window title string, image array
    cv2.imshow("Grant Demo - 106 Pt Dense Face Tracker (Image Mode)", frame)
    print(f"Image file saved successfully to: {OUTPUT_PATH}")
    print("Press any key in the window to exit.")
    
    # Arguments: delay in milliseconds (0 blocks indefinitely until a keypress)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()