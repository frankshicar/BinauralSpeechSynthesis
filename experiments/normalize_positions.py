import numpy as np
import os
import glob

def normalize_positions(dataset_dir):
    subjects = glob.glob(os.path.join(dataset_dir, "subject*"))
    subjects += glob.glob(os.path.join(dataset_dir, "validation_sequence"))
    
    for subject_dir in subjects:
        if not os.path.isdir(subject_dir):
            continue
            
        print(f"Processing {subject_dir}...")
        
        # 1. Normalize Tx Positions
        tx_path = os.path.join(subject_dir, "tx_positions.txt")
        if os.path.exists(tx_path):
            data = np.loadtxt(tx_path)
            # data shape: (N, 7) -> [x, y, z, qx, qy, qz, qw]
            
            # Extract positions (first 3 columns)
            positions = data[:, :3]
            
            # Calculate current norms
            norms = np.linalg.norm(positions, axis=1, keepdims=True)
            
            # Normalize to 1.0 (100cm = 1.0m)
            # Avoid division by zero
            norms[norms == 0] = 1e-8
            normalized_positions = positions / norms * 1.0
            
            # Update data
            data[:, :3] = normalized_positions
            
            # --- NEW: Force Tx Orientation to Face Forward ---
            # Set quaternion to identity (0, 0, 0, 1) or specific forward direction
            # Usually (0, 0, 0, 1) means "no rotation". 
            # Assuming 'forward' is aligned with default coordinate system.
            data[:, 3] = 0.0 # qx
            data[:, 4] = 0.0 # qy
            data[:, 5] = 0.0 # qz
            data[:, 6] = 1.0 # qw
            print(f"  -> Forced Tx orientation to Face Forward (0,0,0,1)")
            # -------------------------------------------------
            
            # Save back
            np.savetxt(tx_path, data, fmt='%.7f')
            print(f"  -> Normalized Tx positions to 1.0m")
            
        # 2. Create/Overlap Rx Positions (Rx is at origin)
        # Even though the model currently assumes Rx is at origin, 
        # creating this file makes the dataset explicit.
        if os.path.exists(tx_path): # Only create if Tx exists
            n_frames = data.shape[0]
            # Rx is fixed at origin: (0, 0, 0) position, (0, 0, 0, 1) quaternion (identity)
            rx_data = np.zeros((n_frames, 7))
            rx_data[:, 6] = 1.0 # Set qw = 1
            
            rx_path = os.path.join(subject_dir, "rx_positions.txt")
            np.savetxt(rx_path, rx_data, fmt='%.7f')
            print(f"  -> Created Rx positions at origin (0,0,0)")

if __name__ == "__main__":
    dataset_dir = "./dataset/testset"
    normalize_positions(dataset_dir)
