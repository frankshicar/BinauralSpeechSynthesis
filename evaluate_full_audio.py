import sys
sys.path.insert(0, '/home/sbplab/frank/BinauralSpeechSynthesis')

import os
import torch
import numpy as np
import soundfile as sf
from tqdm import tqdm
import glob

from src.models_improved_residual import ImprovedResidualPhaseNet


def load_audio_and_view(subject_dir):
    """Load full audio and view from subject directory"""
    # Load mono
    mono_path = os.path.join(subject_dir, 'mono.wav')
    mono, sr = sf.read(mono_path)
    
    # Load binaural (ground truth)
    binaural_files = sorted(glob.glob(os.path.join(subject_dir, '[0-9]*.wav')))
    if binaural_files:
        binaural, _ = sf.read(binaural_files[0])  # Use first angle
    else:
        binaural = None
    
    # Load view
    view_path = os.path.join(subject_dir, 'tx_positions.txt')
    view = np.loadtxt(view_path)
    
    return mono, binaural, view, sr


def process_full_audio(model, mono, view, device='cuda', chunk_size=48000):
    """Process full audio in chunks to avoid OOM"""
    model.eval()
    
    # Prepare view
    # view shape from file: (N, 7) where N is number of frames
    # Need: (7, T) where T = len(mono) / 400
    
    if view.ndim == 2 and view.shape[1] == 7:
        # Transpose to (7, N)
        view = view.T
    
    num_view_frames = len(mono) // 400
    
    # Interpolate or repeat view to match audio length
    if view.shape[1] < num_view_frames:
        # Repeat last frame
        padding = np.tile(view[:, -1:], (1, num_view_frames - view.shape[1]))
        view = np.concatenate([view, padding], axis=1)
    elif view.shape[1] > num_view_frames:
        # Truncate
        view = view[:, :num_view_frames]
    
    # Convert to tensor
    mono_tensor = torch.from_numpy(mono).float().unsqueeze(0).unsqueeze(0).to(device)
    view_tensor = torch.from_numpy(view).float().unsqueeze(0).to(device)
    
    # Process in chunks
    output_L = []
    output_R = []
    
    with torch.no_grad():
        for start in range(0, len(mono), chunk_size):
            end = min(start + chunk_size, len(mono))
            
            # Get chunk
            mono_chunk = mono_tensor[:, :, start:end]
            
            # Get corresponding view frames
            view_start = start // 400
            view_end = end // 400
            if view_end > view_tensor.shape[2]:
                view_end = view_tensor.shape[2]
            view_chunk = view_tensor[:, :, view_start:view_end]
            
            # Ensure view_chunk has at least 1 frame
            if view_chunk.shape[2] == 0:
                view_chunk = view_tensor[:, :, -1:]
            
            # Forward
            y_L, y_R, _ = model(mono_chunk, view_chunk)
            
            # Collect
            output_L.append(y_L.squeeze().cpu().numpy())
            output_R.append(y_R.squeeze().cpu().numpy())
    
    # Concatenate
    output_L = np.concatenate(output_L)
    output_R = np.concatenate(output_R)
    
    return output_L, output_R


def evaluate_full_audio(model_path, dataset_dir, output_dir, device='cuda'):
    """Evaluate on full audio files"""
    
    # Load model
    model = ImprovedResidualPhaseNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded model from: {model_path}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving audio to: {output_dir}")
    
    # Find all subjects
    subject_dirs = sorted([d for d in glob.glob(os.path.join(dataset_dir, 'subject*')) 
                          if os.path.isdir(d)])
    print(f"Found {len(subject_dirs)} subjects")
    
    # Process each subject
    for subject_dir in tqdm(subject_dirs, desc="Processing subjects"):
        subject_name = os.path.basename(subject_dir)
        
        # Load audio
        mono, binaural, view, sr = load_audio_and_view(subject_dir)
        
        # Process
        pred_L, pred_R = process_full_audio(model, mono, view, device)
        
        # Stack stereo
        pred_stereo = np.stack([pred_L, pred_R], axis=1)
        
        # Save
        output_path = os.path.join(output_dir, f'{subject_name}_pred.wav')
        sf.write(output_path, pred_stereo, sr)
        
        # Also save mono and gt for comparison
        mono_path = os.path.join(output_dir, f'{subject_name}_mono.wav')
        sf.write(mono_path, mono, sr)
        
        if binaural is not None:
            gt_path = os.path.join(output_dir, f'{subject_name}_gt.wav')
            sf.write(gt_path, binaural, sr)
        
        print(f"  Saved: {subject_name}")
    
    print(f"\n✅ Done! Audio files saved to: {output_dir}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '--model_file', type=str,
                       default='checkpoints/improved_residual_best.net',
                       help='Path to model checkpoint')
    parser.add_argument('--dataset', '--dataset_directory', type=str,
                       default='dataset/testset',
                       help='Path to test dataset')
    parser.add_argument('--output', '--artifacts_directory', type=str,
                       default='results_full_audio',
                       help='Output directory for audio files')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    evaluate_full_audio(args.model, args.dataset, args.output, args.device)
