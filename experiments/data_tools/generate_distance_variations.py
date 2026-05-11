import numpy as np
import os
import soundfile as sf
import shutil

def generate_variations(base_subject_dir, output_root, distances=[0.2, 0.5, 1.0, 2.0, 3.0]):
    """
    Creates copies of the subject dataset with overridden Tx distances.
    """
    if not os.path.exists(output_root):
        os.makedirs(output_root)
        
    print(f"Generating distance variations for {base_subject_dir} in {output_root}...")
    
    # Read original data
    tx_path = os.path.join(base_subject_dir, "tx_positions.txt")
    mono_path = os.path.join(base_subject_dir, "mono.wav")
    binaural_path = os.path.join(base_subject_dir, "binaural.wav")
    
    # Read Tx
    tx_data = np.loadtxt(tx_path)
    # Ensure ID quaternion if not already (optional, but good for control)
    tx_data[:, 3:7] = [0, 0, 0, 1] 
    
    # Read Audio (to copy)
    # We just need to copy files, but let's be explicit
    
    for dist in distances:
        dist_name = f"subject1_dist_{int(dist*100)}cm"
        target_dir = os.path.join(output_root, dist_name)
        
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
            
        # 1. Update Tx Positions
        # Normalize direction and scale to new distance
        positions = tx_data[:, :3]
        norms = np.linalg.norm(positions, axis=1, keepdims=True)
        norms[norms == 0] = 1e-8
        
        new_positions = (positions / norms) * dist
        
        new_tx_data = tx_data.copy()
        new_tx_data[:, :3] = new_positions
        
        # Save new Tx
        np.savetxt(os.path.join(target_dir, "tx_positions.txt"), new_tx_data, fmt='%.7f')
        
        # 2. Copy Rx (at origin 0,0,0)
        # Create zero-filled rx if not exists
        n_frames = tx_data.shape[0]
        rx_data = np.zeros((n_frames, 7))
        rx_data[:, 6] = 1.0
        np.savetxt(os.path.join(target_dir, "rx_positions.txt"), rx_data, fmt='%.7f')
        
        # 3. Copy Audio Files
        shutil.copy(mono_path, os.path.join(target_dir, "mono.wav"))
        shutil.copy(binaural_path, os.path.join(target_dir, "binaural.wav"))
        
        print(f"  -> Generated {dist_name} (Distance: {dist}m)")

if __name__ == "__main__":
    # Use subject1 as the base
    base_subject = "./dataset/testset/subject1"
    # Create a new 'testset_variations' folder to run evaluation on
    output_root = "./dataset/testset_variations"
    
    generate_variations(base_subject, output_root)
