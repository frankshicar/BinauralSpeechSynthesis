import os
import glob
import numpy as np
import scipy.io.wavfile as wavfile
from scipy.spatial.transform import Rotation as R
import math

# Configuration
DATASET_DIR = "dataset" # Relative to run location
SUBJECTS = {
    1: -60,
    2: -60,
    3: -90,
    4: 0,
    5: 30,
    6: 60
}
RADIUS = 1.0
UPDATE_RATE_HZ = 120
AUDIO_SR_HZ = 48000 # Standard for this dataset usually, but we check file
FRAME_SIZE_AUDIO = 400 # 48000 / 400 = 120Hz

def get_wav_path(subject_dir):
    # Try mono.wav
    p = os.path.join(subject_dir, "mono.wav")
    if os.path.exists(p):
        return p
    # Try any wav
    wavs = glob.glob(os.path.join(subject_dir, "*.wav"))
    if wavs:
        return wavs[0]
    return None

def generate_data():
    for subj_id, angle_deg in SUBJECTS.items():
        subj_dir = os.path.join(DATASET_DIR, f"subject{subj_id}")
        if not os.path.exists(subj_dir):
            print(f"Warning: Directory {subj_dir} not found. Skipping.")
            continue
            
        wav_path = get_wav_path(subj_dir)
        if not wav_path:
            print(f"Warning: No wav file found in {subj_dir}. Skipping.")
            continue
            
        print(f"Processing Subject {subj_id} (Angle {angle_deg}) using {wav_path}...")
        
        # Read WAV duration
        sr, data = wavfile.read(wav_path)
        # Handle stereo/mono
        if data.ndim > 1:
            n_samples = data.shape[0]
        else:
            n_samples = len(data)
            
        # Calculate number of tracking frames
        # In src/dataset.py: view is expected to be T/400.
        # But actually dataset.py says:
        # view = tx_view...
        # dataset.py check:
        # offset//400 : (offset+chunk)//400
        # So it expects 1 tracking sample per 400 audio samples.
        num_frames = int(math.ceil(n_samples / FRAME_SIZE_AUDIO))
        
        # Calculate Position
        angle_rad = math.radians(angle_deg)
        # Coordinate system assumed from S4 in dataset_original: 0 deg = +X, 90 deg = +Y
        x = RADIUS * math.cos(angle_rad)
        y = RADIUS * math.sin(angle_rad)
        z = 0.0
        pos = np.array([x, y, z])
        
        # Calculate Rotation (Facing Origin)
        # Target Forward = Normalize(Origin - Pos)
        if np.linalg.norm(pos) < 1e-6:
            forward = np.array([1.0, 0.0, 0.0]) # At origin, arbitrary
        else:
            forward = -pos / np.linalg.norm(pos)
            
        up = np.array([0.0, 0.0, 1.0])
        # Local Y = Forward, Local Z = Up, Local X = Right
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        
        # Recompute up to ensure ortho
        up = np.cross(right, forward)
        
        # Rotation Matrix: [Right, Forward, Up]
        rot_mat = np.column_stack([right, forward, up])
        quat = R.from_matrix(rot_mat).as_quat() 
        # Scipy returns [x, y, z, w]
        
        # Write to file
        out_path = os.path.join(subj_dir, "tx_positions.txt")
        with open(out_path, "w") as f:
            line = f"{x:.7f} {y:.7f} {z:.7f} {quat[0]:.7f} {quat[1]:.7f} {quat[2]:.7f} {quat[3]:.7f}\n"
            for _ in range(num_frames):
                f.write(line)
                
        print(f"Written {num_frames} lines to {out_path}")

if __name__ == "__main__":
    generate_data()
