import os
import argparse
import numpy as np
import torch as th
import soundfile as sf
import sys
from src.models import BinauralNetwork
from src.losses import L2Loss

# Simplified version of evaluate.py to scan angles
def load_model(weights_path):
    if not os.path.exists(weights_path):
        print("Error: Model weights not found in the outputs directory.")
        exit()
    network = th.load(weights_path)
    if isinstance(network, dict):
        return network
    network.eval()
    if th.cuda.is_available():
        network.cuda()
    return network.state_dict()

def chunked_forwarding(network, segments, mono, view):
    binaural_segments = []
    for i in range(len(segments)):
        start_frame = segments[i]['start']
        end_frame = segments[i]['end']
        start_view = start_frame // 400
        end_view = end_frame // 400
        mono_chunk = mono[:, :, start_frame:end_frame]
        view_chunk = view[:, :, start_view:end_view]
        if th.cuda.is_available():
            mono_chunk = mono_chunk.cuda()
            view_chunk = view_chunk.cuda()
        binaural_chunk = network(mono_chunk, view_chunk)["output"]
        binaural_segments.append(binaural_chunk.detach().cpu())
    return th.cat(binaural_segments, dim=-1)

def scan_angles(dataset_dir, model_file, subjects):
    # Initialize Network
    net = BinauralNetwork(view_dim=7, wavenet_blocks=3)
    net.load_state_dict(load_model(model_file))
    if th.cuda.is_available():
        net.cuda()
    net.eval()
    
    criterion = L2Loss()
    
    print("| Subject | Best Angle | Min L2 Error |")
    print("| :--- | :--- | :--- |")
    
    for subject in subjects:
        subj_dir = os.path.join(dataset_dir, subject)
        mono_path = os.path.join(subj_dir, "mono.wav")
        ref_path = os.path.join(subj_dir, "binaural.wav")
        
        if not os.path.exists(mono_path):
            continue
            
        # Load Audio Once
        try:
            mono_data, sr = sf.read(mono_path, dtype='float32')
            if mono_data.ndim == 1: mono_data = mono_data.reshape(1, -1)
            else: mono_data = mono_data.T
            mono = th.from_numpy(mono_data)
            
            ref_data, sr = sf.read(ref_path, dtype='float32')
            if ref_data.ndim == 1: ref_data = ref_data.reshape(1, -1)
            else: ref_data = ref_data.T
            reference = th.from_numpy(ref_data)
        except Exception as e:
            print(f"Error loading audio for {subject}: {e}")
            continue

        best_angle = 0
        min_error = float('inf')
        
        # Scan Angles -180 to 180 step 10
        # (Using coarse step for speed, can refine if needed)
        angles = range(-180, 180, 10)
        
        for angle in angles:
            # Generate View on the fly
            az_rad = np.radians(angle)
            # Face Forward, 1m distance
            x, y, z = np.cos(az_rad), np.sin(az_rad), 0.0
            # Quaternion (Identity)
            qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0
            
            # Create View Tensor
            n_frames = mono.shape[-1] // 400 + 1 # Approx
            view_np = np.zeros((7, n_frames), dtype=np.float32)
            view_np[0, :] = x
            view_np[1, :] = y
            view_np[2, :] = z
            view_np[3, :] = qx
            view_np[4, :] = qy
            view_np[5, :] = qz
            view_np[6, :] = qw
            
            view = th.from_numpy(view_np)
            
            # Align
            n_frames = min(mono.shape[-1] // 400, view.shape[-1])
            mono_clip = mono[:, :n_frames * 400]
            view_clip = view[:, :n_frames]
            ref_clip = reference[:, :n_frames * 400]
            
            # Forward
            with th.no_grad():
                # Process only first 3 seconds to save time/memory if long
                # Actually entire file is safer for metrics
                chunk_size = 48000
                segments = [{'start': i, 'end': min(i+chunk_size, mono_clip.shape[-1])} 
                            for i in range(0, mono_clip.shape[-1], chunk_size)]
                
                binaural = chunked_forwarding(net, segments, mono_clip.unsqueeze(0), view_clip.unsqueeze(0))
                
                # Compute Error (CPU)
                binaural = binaural.reshape(2, -1).cpu()
                ref_clip = ref_clip.cpu() 
                # Need batch dim for loss?
                # L2Loss expects (B, C, T) usually?
                # Let's check src/losses.py. Loss.__call__ takes (data, target).
                # PhaseLoss._loss takes BxCxT.
                binaural_b = binaural.unsqueeze(0)
                ref_b = ref_clip.unsqueeze(0)
                
                loss = criterion(binaural_b, ref_b).item()
                
                if loss < min_error:
                    min_error = loss
                    best_angle = angle
        
        print(f"| {subject} | {best_angle}° | {min_error*1000:.3f} |")

if __name__ == "__main__":
    subjects = [f"subject{i+1}" for i in range(6)]
    scan_angles("./dataset/testset", "outputs/binaural_network.net", subjects)
