import sys
sys.path.insert(0, '/home/sbplab/frank/BinauralSpeechSynthesis')

import os
import torch
import numpy as np
import soundfile as sf
from tqdm import tqdm

from src.models_improved_residual import ImprovedResidualPhaseNet
from src.dataset import BinauralDataset


def evaluate_model(model_path, dataset_dir, device='cuda', output_dir=None):
    """Evaluate ImprovedResidualPhaseNet on test set"""
    
    # Load model
    model = ImprovedResidualPhaseNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded model from: {model_path}")
    
    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving audio to: {output_dir}")
    
    # Load dataset
    dataset = BinauralDataset(
        dataset_directory=dataset_dir,
        chunk_size_ms=200,
        overlap=0.5
    )
    print(f"Test samples: {len(dataset)}")
    
    # Metrics
    total_l2 = 0
    total_amplitude = 0
    total_phase = 0
    num_samples = 0
    
    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="Evaluating"):
            mono, binaural, view = dataset[i]
            
            # Convert to tensor if needed
            if isinstance(mono, np.ndarray):
                mono = torch.from_numpy(mono).float()
            if isinstance(binaural, np.ndarray):
                binaural = torch.from_numpy(binaural).float()
            if isinstance(view, np.ndarray):
                view = torch.from_numpy(view).float()
            
            # Add batch dimension
            mono = mono.unsqueeze(0).to(device)
            binaural = binaural.unsqueeze(0).to(device)
            view = view.unsqueeze(0).to(device)
            
            # Forward
            y_L, y_R, outputs = model(mono, view)
            
            # Save audio if requested (only first 100 samples to save space)
            if output_dir and i < 100:
                # Convert to numpy
                pred_L = y_L.squeeze().cpu().numpy()
                pred_R = y_R.squeeze().cpu().numpy()
                gt_L = binaural[0, 0].cpu().numpy()
                gt_R = binaural[0, 1].cpu().numpy()
                mono_audio = mono.squeeze().cpu().numpy()
                
                # Stack stereo
                pred_stereo = np.stack([pred_L, pred_R], axis=0)
                gt_stereo = np.stack([gt_L, gt_R], axis=0)
                
                # Save
                sf.write(os.path.join(output_dir, f'sample_{i:04d}_pred.wav'), 
                        pred_stereo.T, 48000)
                sf.write(os.path.join(output_dir, f'sample_{i:04d}_gt.wav'), 
                        gt_stereo.T, 48000)
                sf.write(os.path.join(output_dir, f'sample_{i:04d}_mono.wav'), 
                        mono_audio, 48000)
            
            # Ground truth
            y_L_gt = binaural[:, 0:1]
            y_R_gt = binaural[:, 1:2]
            
            # L2 loss (waveform)
            l2_loss = (
                torch.mean((y_L - y_L_gt) ** 2) +
                torch.mean((y_R - y_R_gt) ** 2)
            ) / 2
            
            # STFT for amplitude and phase
            window = torch.hann_window(1024).to(device)
            
            pred_L_stft = torch.stft(y_L.squeeze(1), n_fft=1024, hop_length=64, 
                                     win_length=1024, window=window, return_complex=True)
            pred_R_stft = torch.stft(y_R.squeeze(1), n_fft=1024, hop_length=64,
                                     win_length=1024, window=window, return_complex=True)
            gt_L_stft = torch.stft(y_L_gt.squeeze(1), n_fft=1024, hop_length=64,
                                   win_length=1024, window=window, return_complex=True)
            gt_R_stft = torch.stft(y_R_gt.squeeze(1), n_fft=1024, hop_length=64,
                                   win_length=1024, window=window, return_complex=True)
            
            # Amplitude loss
            pred_mag = (pred_L_stft.abs() + pred_R_stft.abs()) / 2
            gt_mag = (gt_L_stft.abs() + gt_R_stft.abs()) / 2
            amplitude_loss = torch.mean((pred_mag - gt_mag) ** 2)
            
            # Phase loss
            pred_phase_L = torch.angle(pred_L_stft)
            pred_phase_R = torch.angle(pred_R_stft)
            gt_phase_L = torch.angle(gt_L_stft)
            gt_phase_R = torch.angle(gt_R_stft)
            
            phase_loss = (
                torch.mean((torch.sin(pred_phase_L) - torch.sin(gt_phase_L)) ** 2) +
                torch.mean((torch.cos(pred_phase_L) - torch.cos(gt_phase_L)) ** 2) +
                torch.mean((torch.sin(pred_phase_R) - torch.sin(gt_phase_R)) ** 2) +
                torch.mean((torch.cos(pred_phase_R) - torch.cos(gt_phase_R)) ** 2)
            ) / 4
            
            # Accumulate
            total_l2 += l2_loss.item()
            total_amplitude += amplitude_loss.item()
            total_phase += phase_loss.item()
            num_samples += 1
    
    # Average
    avg_l2 = total_l2 / num_samples
    avg_amplitude = total_amplitude / num_samples
    avg_phase = total_phase / num_samples
    
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    print(f"L2 誤差 (x10^3): {avg_l2 * 1000:.3f}")
    print(f"振幅誤差 (Amplitude): {avg_amplitude:.3f}")
    print(f"相位誤差 (Phase): {avg_phase:.3f}")
    print("="*60)
    
    return {
        'l2': avg_l2,
        'amplitude': avg_amplitude,
        'phase': avg_phase
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '--model_file', type=str, 
                       default='checkpoints/improved_residual_best.net',
                       help='Path to model checkpoint')
    parser.add_argument('--dataset', '--dataset_directory', type=str,
                       default='dataset/testset',
                       help='Path to test dataset')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--artifacts_directory', type=str, default=None,
                       help='Directory to save output audio files (optional)')
    parser.add_argument('--blocks', type=int, default=3,
                       help='Ignored (for compatibility with old script)')
    
    args = parser.parse_args()
    
    results = evaluate_model(args.model, args.dataset, args.device, args.artifacts_directory)
