"""
用 Meta 原版 L2Loss / AmplitudeLoss / PhaseLoss 評估 GeoWarpFiLMNet
"""
import sys
sys.path.insert(0, '/home/sbplab/frank/BinauralSpeechSynthesis')

import os
import argparse
import torch
import numpy as np
import soundfile as sf
import glob

from src.models_geowarp_film import GeoWarpFiLMNet
from src.losses import L2Loss, AmplitudeLoss, PhaseLoss
from src.doa import gcc_phat_estimate

parser = argparse.ArgumentParser()
parser.add_argument('--model_file', type=str, default='geowarp_film/best.net')
parser.add_argument('--dataset_directory', type=str, default='dataset/testset')
parser.add_argument('--artifacts_directory', type=str, default=None)
args = parser.parse_args()

# Subject angle mapping (for test_static and test_13angles)
SUBJECT_ANGLES = {
    'subject1': -90.0,
    'subject2': -60.0,
    'subject3': -30.0,
    'subject4': 0.0,
    'subject5': 30.0,
    'subject6': 60.0,
    'subject7': 90.0,
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model = GeoWarpFiLMNet().to(device)
ckpt = torch.load(args.model_file, map_location=device)
# Handle both full checkpoint and state_dict only
if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
    model.load_state_dict(ckpt['model_state_dict'])
else:
    model.load_state_dict(ckpt)
model.eval()
print(f"Loaded: {args.model_file}")

if args.artifacts_directory:
    os.makedirs(args.artifacts_directory, exist_ok=True)

# Loss functions (Meta originals)
l2_fn        = L2Loss()
amplitude_fn = AmplitudeLoss(sample_rate=48000)
phase_fn     = PhaseLoss(sample_rate=48000, ignore_below=0.2)

subject_dirs = sorted([d for d in glob.glob(os.path.join(args.dataset_directory, 'subject*'))
                       if os.path.isdir(d)])
print(f"Found {len(subject_dirs)} subjects\n")

all_l2, all_amp, all_phase, all_weights = [], [], [], []
all_angular_errors = []

for subject_dir in subject_dirs:
    name = os.path.basename(subject_dir)

    # Load mono
    mono_np, sr = sf.read(os.path.join(subject_dir, 'mono.wav'), dtype='float32')
    if mono_np.ndim == 1:
        mono_np = mono_np[None, :]   # (1, T)
    else:
        mono_np = mono_np.T

    # Load binaural ground truth
    ref_path = os.path.join(subject_dir, 'binaural.wav')
    if not os.path.exists(ref_path):
        print(f"  {name}: binaural.wav not found, skipping")
        continue
    ref_np, _ = sf.read(ref_path, dtype='float32')
    if ref_np.ndim == 1:
        ref_np = ref_np[None, :]
    else:
        ref_np = ref_np.T   # (2, T)

    # Load view
    view_np = np.loadtxt(os.path.join(subject_dir, 'tx_positions.txt')).T.astype(np.float32)  # (7, K)

    # Align lengths to view
    target_samples = view_np.shape[1] * 400
    mono_np = mono_np[:, :target_samples]
    if mono_np.shape[1] < target_samples:
        mono_np = np.pad(mono_np, ((0,0),(0, target_samples - mono_np.shape[1])))

    mono_t  = torch.from_numpy(mono_np).unsqueeze(0).to(device)   # (1, 1, T)
    view_t  = torch.from_numpy(view_np).unsqueeze(0).to(device)   # (1, 7, K)

    # Run model in chunks (1s = 48000 samples) to avoid OOM
    chunk_samples = 48000
    pred_L_chunks, pred_R_chunks = [], []

    with torch.no_grad():
        T = mono_t.shape[-1]
        for start in range(0, T, chunk_samples):
            end = min(start + chunk_samples, T)
            m_chunk = mono_t[:, :, start:end]

            v_start = start // 400
            v_end   = min(end // 400, view_t.shape[2])
            v_chunk = view_t[:, :, v_start:v_end]
            if v_chunk.shape[2] == 0:
                v_chunk = view_t[:, :, -1:]

            y_L, y_R, _, _, _, _ = model(m_chunk, v_chunk)  # ignore STFT outputs
            pred_L_chunks.append(y_L.squeeze().cpu())
            pred_R_chunks.append(y_R.squeeze().cpu())

    pred_L = torch.cat(pred_L_chunks)   # (T,)
    pred_R = torch.cat(pred_R_chunks)

    # Align to reference length
    min_len = min(pred_L.shape[0], ref_np.shape[1])
    pred_binaural = torch.stack([pred_L[:min_len], pred_R[:min_len]]).unsqueeze(0)  # (1, 2, T)
    ref_t = torch.from_numpy(ref_np[:, :min_len]).unsqueeze(0)                      # (1, 2, T)
    if torch.cuda.is_available():
        pred_binaural = pred_binaural.cuda()
        ref_t = ref_t.cuda()

    # Compute Meta-style metrics
    l2  = l2_fn(pred_binaural, ref_t).item()
    amp = amplitude_fn(pred_binaural, ref_t).item()
    ph  = phase_fn(pred_binaural, ref_t).item()

    # Compute angular error if ground truth angle exists
    gt_angle = SUBJECT_ANGLES.get(name, None)
    angular_error = None
    pred_angle = None
    
    if gt_angle is not None:
        try:
            audio_np = pred_binaural.squeeze(0).cpu().numpy()  # (2, T)
            left_channel = audio_np[0, :]
            right_channel = audio_np[1, :]
            
            # Trim trailing silence (same as evaluate.py)
            valid_indices = np.where(np.abs(left_channel) > 1e-4)[0]
            if len(valid_indices) > 0:
                valid_len = int(valid_indices[-1] * 0.95)
                left_channel = left_channel[:valid_len]
                right_channel = right_channel[:valid_len]
            
            trimmed_binaural = np.stack([left_channel, right_channel])
            pred_angle = gcc_phat_estimate(trimmed_binaural, sample_rate=48000)
            
            # Invert angle sign to match coordinate system
            if pred_angle is not None:
                pred_angle = -pred_angle
                error = abs(pred_angle - gt_angle)
                error = min(error, 360.0 - error)
                angular_error = error
        except Exception as e:
            print(f"    [Warning] Angular error computation failed: {e}")

    if angular_error is not None:
        print(f"  {name}: L2={l2*1000:.3f}e-3  Amp={amp:.3f}  Phase={ph:.3f}  Angle={angular_error:.1f}° (Pred={pred_angle:.1f}°, GT={gt_angle:.1f}°)")
        all_angular_errors.append(angular_error)
    else:
        print(f"  {name}: L2={l2*1000:.3f}e-3  Amp={amp:.3f}  Phase={ph:.3f}")

    all_l2.append(l2)
    all_amp.append(amp)
    all_phase.append(ph)
    all_weights.append(min_len)

    # Optionally save predicted audio
    if args.artifacts_directory:
        out = pred_binaural.squeeze(0).cpu().numpy().T   # (T, 2)
        sf.write(os.path.join(args.artifacts_directory, f'{name}.wav'), out, sr)

# Weighted average (by number of samples, same as Meta evaluate.py)
w = np.array(all_weights, dtype=np.float64)
w /= w.sum()

l2_avg  = float(np.dot(all_l2,    w))
amp_avg = float(np.dot(all_amp,   w))
ph_avg  = float(np.dot(all_phase, w))

print("\n" + "="*50)
print("Fair Comparison (Meta-style metrics)")
print("="*50)
print(f"l2 (x10^3):     {l2_avg * 1000:.3f}")
print(f"amplitude:      {amp_avg:.3f}")
print(f"phase:          {ph_avg:.3f}")
if all_angular_errors:
    ang_avg = float(np.mean(all_angular_errors))
    print(f"angular error:  {ang_avg:.1f}°")
print("="*50)
print("\nMeta pretrained baselines:")
print("  small (1 block): l2=0.197  amp=0.043  phase=0.862")
print("  large (3 block): l2=0.144  amp=0.036  phase=0.804")
print("\nDPATFNet (ICASSP 2025):")
print("  l2=0.148  amp=0.037  phase=0.717")
