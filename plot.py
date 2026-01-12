import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, TextBox
from collections import defaultdict

# =========================================================================
# 1. CONSTANTS & UTILS
# =========================================================================

def distTwoPoints2D(x1, y1, x2, y2):
    """Calculates the euclidean distance between two 2D keypoints."""
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)

SKELETON_EDGES = [
    (15, 13), (13, 11), (16, 14), (14, 12),  # Legs
    (11, 12), (5, 11), (6, 12),              # Hips/Torso connection
    (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), # Shoulders/Arms
    (1, 2), (0, 1), (0, 2), (1, 3), (2, 4)   # Face/Head
]

LIMB_SEGMENTS_TO_SAVE = [
    (14, 16, "Right_Knee_to_Ankle"), (14, 12, "Right_Knee_to_Hip"),
    (10, 8, "Right_Wrist_to_Elbow"), (8, 6, "Right_Elbow_to_Shoulder"),
    (13, 15, "Left_Knee_to_Ankle"), (13, 11, "Left_Knee_to_Hip"),
    (9, 7, "Left_Wrist_to_Elbow"), (7, 5, "Left_Elbow_to_Shoulder"),
]

ID_COLORS = plt.cm.get_cmap('hsv', 10)
CUSTOM_LINE_COLOR = 'magenta'
REF_HEAD_IDX = 0
REF_FOOT_IDX = 15

# =========================================================================
# 2. INTERACTIVE 2D POSE PLAYER
# =========================================================================

class PoseViewer2D:
    def __init__(self, json_path, fps=15):
        # --- Data Loading ---
        self.frames_map, self.x_lim, self.y_lim = self.load_pose_data(json_path)
        self.sorted_frames = sorted(self.frames_map.keys())
        self.num_frames = len(self.sorted_frames)
        
        if self.num_frames == 0:
            raise RuntimeError("No pose data found in the provided JSON.")

        # --- State ---
        self.i = 0 
        self.is_playing = False
        self.fps = fps
        self.interval = int(1000 / self.fps)
        self.custom_line_points = (0, 15)
        self.sf_vertical = 1.0
        self.ref_height_inches = 70.0
        self.target_person_id = 0 

        # --- Figure Setup ---
        self.fig, self.ax = plt.subplots(figsize=(11, 7))
        plt.subplots_adjust(bottom=0.30)
        
        self.ax.set_xlim(0, self.x_lim)
        self.ax.set_ylim(self.y_lim, 0) # Origin at top-left for pixels
        self.ax.set_aspect('equal')
        self.ax.grid(True, linestyle=':', alpha=0.6)
        
        self.scatters = []
        self.lines = []
        self.labels = []
        self.custom_line, = self.ax.plot([], [], color=CUSTOM_LINE_COLOR, lw=3, ls='--', zorder=5)
        self.custom_line_text = self.ax.text(0, 0, "", color=CUSTOM_LINE_COLOR, fontweight='bold', zorder=6)

        self._add_widgets()
        
        # --- Timer for Playback ---
        self.timer = self.fig.canvas.new_timer(interval=self.interval)
        self.timer.add_callback(self._on_timer)
        
        self._draw_frame(0)

    def load_pose_data(self, file_path):
        """Processes JSON where detections have track_id_native and keypoints_xyz."""
        try:
            with open(file_path, 'r') as f:
                all_detections = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load JSON: {e}")
        
        frames_map = defaultdict(list)
        max_x, max_y = 0, 0
        
        for d in all_detections:
            frame_idx = d['frame_index']
            kp = np.array(d.get('keypoints_xyz', []), dtype=float)
            if kp.size == 0: continue
            
            # CHANGE 'track_id' TO 'track_id_native' HERE
            tid = d.get('track_id_native', 0)
            
            frames_map[frame_idx].append({
                'keypoints': kp, 
                'track_id': tid
            })
            
            max_x = max(max_x, np.max(kp[:, 0]))
            max_y = max(max_y, np.max(kp[:, 1]))
            
        return frames_map, int(max_x * 1.1), int(max_y * 1.1)
    
    def _add_widgets(self):
        # Position definitions [left, bottom, width, height]
        ax_prev = plt.axes([0.15, 0.20, 0.08, 0.05])
        ax_play = plt.axes([0.24, 0.20, 0.08, 0.05])
        ax_next = plt.axes([0.33, 0.20, 0.08, 0.05])
        ax_save = plt.axes([0.15, 0.12, 0.12, 0.05])
        
        ax_kp_input = plt.axes([0.70, 0.20, 0.10, 0.05])
        ax_height_input = plt.axes([0.70, 0.12, 0.10, 0.05])
        ax_target_id = plt.axes([0.50, 0.20, 0.10, 0.05])
        ax_slider = plt.axes([0.15, 0.05, 0.7, 0.03])

        self.slider = Slider(ax_slider, 'Frame', 0, self.num_frames - 1, valinit=0, valstep=1)
        self.btn_prev = Button(ax_prev, 'Prev')
        self.btn_play = Button(ax_play, 'Play')
        self.btn_next = Button(ax_next, 'Next')
        self.btn_save = Button(ax_save, 'Save JSON')
        
        self.tb_kp = TextBox(ax_kp_input, 'Points (A,B) ', initial="0,15")
        self.tb_height = TextBox(ax_height_input, 'Height (In) ', initial="70.0")
        self.tb_target = TextBox(ax_target_id, 'Target ID ', initial="0")

        # Wire up events
        self.btn_play.on_clicked(self.toggle_play)
        self.btn_prev.on_clicked(lambda e: self.step(-1))
        self.btn_next.on_clicked(lambda e: self.step(1))
        self.btn_save.on_clicked(lambda e: self._on_save())
        self.slider.on_changed(self._on_slider_change)
        self.tb_kp.on_submit(self._on_kp_submit)
        self.tb_height.on_submit(self._on_height_submit)
        self.tb_target.on_submit(self._on_target_submit)

    def _draw_frame(self, frame_idx_in_list):
        actual_frame = self.sorted_frames[frame_idx_in_list]
        people = self.frames_map.get(actual_frame, [])
        
        # Remove old artists
        for a in self.scatters + self.lines + self.labels:
            a.remove()
        self.scatters, self.lines, self.labels = [], [], []
        
        target_kp = None
        
        for p in people:
            kp = p['keypoints']
            tid = p['track_id']
            color = ID_COLORS(tid % 10)
            
            if tid == self.target_person_id:
                target_kp = kp

            # Draw Keypoints
            scat = self.ax.scatter(kp[:, 0], kp[:, 1], c=[color], s=40, zorder=3)
            self.scatters.append(scat)
            
            # Draw Skeleton
            for a_idx, b_idx in SKELETON_EDGES:
                if a_idx < len(kp) and b_idx < len(kp):
                    line, = self.ax.plot([kp[a_idx, 0], kp[b_idx, 0]], 
                                        [kp[a_idx, 1], kp[b_idx, 1]], 
                                        c=color, lw=1.5, alpha=0.6, zorder=2)
                    self.lines.append(line)
            
            # ID Label
            label = self.ax.text(kp[0,0], kp[0,1]-10, f"ID:{tid}", color=color, fontweight='bold')
            self.labels.append(label)

        # Custom Distance Measurement
        a, b = self.custom_line_points
        if target_kp is not None and a < len(target_kp) and b < len(target_kp):
            x_vals = [target_kp[a, 0], target_kp[b, 0]]
            y_vals = [target_kp[a, 1], target_kp[b, 1]]
            self.custom_line.set_data(x_vals, y_vals)
            
            px_dist = distTwoPoints2D(x_vals[0], y_vals[0], x_vals[1], y_vals[1])
            real_dist = px_dist * self.sf_vertical
            self.custom_line_text.set_position(((x_vals[0]+x_vals[1])/2, (y_vals[0]+y_vals[1])/2))
            self.custom_line_text.set_text(f"{real_dist:.1f} in")
        else:
            self.custom_line.set_data([], [])
            self.custom_line_text.set_text("")

        self.ax.set_title(f"2D Multi-Pose Viewer | Frame: {actual_frame} | Target Tracking ID: {self.target_person_id}")
        self.fig.canvas.draw_idle()

    # --- Interaction Logic ---
    def _on_timer(self):
        if self.is_playing:
            self.i = (self.i + 1) % self.num_frames
            self.slider.set_val(self.i)

    def toggle_play(self, event=None):
        self.is_playing = not self.is_playing
        self.btn_play.label.set_text('Pause' if self.is_playing else 'Play')
        if self.is_playing:
            self.timer.start()
        else:
            self.timer.stop()
        self.fig.canvas.draw_idle()

    def step(self, delta):
        if self.is_playing:
            self.toggle_play()
        self.i = (self.i + delta) % self.num_frames
        self.slider.set_val(self.i)

    def _on_slider_change(self, val):
        self.i = int(val)
        self._draw_frame(self.i)

    def _on_kp_submit(self, text):
        try:
            self.custom_line_points = tuple(map(int, text.split(',')))
            self._draw_frame(self.i)
        except: print("Invalid keypoint input. Format: 0,15")

    def _on_target_submit(self, text):
        try:
            self.target_person_id = int(text)
            self._draw_frame(self.i)
        except: print("Invalid Target ID.")

    def _on_height_submit(self, text):
        try:
            h = float(text)
            actual_frame = self.sorted_frames[self.i]
            found = False
            for p in self.frames_map[actual_frame]:
                if p['track_id'] == self.target_person_id:
                    kp = p['keypoints']
                    pixel_height = abs(kp[REF_HEAD_IDX, 1] - kp[REF_FOOT_IDX, 1])
                    if pixel_height > 0:
                        self.sf_vertical = h / pixel_height
                        self.ref_height_inches = h
                        found = True
                        print(f"New Scale Factor: {self.sf_vertical:.4f} in/px")
            if not found: print(f"Target ID {self.target_person_id} not in current frame.")
        except Exception as e: print(f"Scaling error: {e}")
        self._draw_frame(self.i)

    def _on_save(self):
        actual_frame = self.sorted_frames[self.i]
        target_kp = next((p['keypoints'] for p in self.frames_map[actual_frame] if p['track_id'] == self.target_person_id), None)
        
        if target_kp is None:
            print("Target not found in this frame. Cannot save.")
            return
        
        output = {
            "frame": actual_frame, 
            "target_id": self.target_person_id,
            "scale_factor": self.sf_vertical,
            "measurements_inches": {}
        }
        for a, b, name in LIMB_SEGMENTS_TO_SAVE:
            d = distTwoPoints2D(target_kp[a,0], target_kp[a,1], target_kp[b,0], target_kp[b,1]) * self.sf_vertical
            output["measurements_inches"][name] = round(float(d), 2)
        
        filename = f'measurements_frame_{actual_frame}_id_{self.target_person_id}.json'
        with open(filename, 'w') as f:
            json.dump(output, f, indent=4)
        print(f"Saved to {filename}")

    def run(self):
        print(f"Viewer loaded with {self.num_frames} frames.")
        plt.show()

