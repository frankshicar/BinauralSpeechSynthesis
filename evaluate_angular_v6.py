"""
Angular error evaluation: mono -> model -> binaural -> GCC-PHAT -> angle
No binaural GT needed.
"""
import sys
sys.path.insert(0, '/home/sbplab/frank/BinauralSpeechSynthesis')

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf
import glob

from src.doa import gcc_phat_estimate

parser = argparse.ArgumentParser()
parser.add_argument('--model_file', type=str, required=True)
parser.add_argument('--dataset_directory', type=str, required=True)
parser.add_argument('--model_type', type=str, default='geowarp_v4',
                    choices=['geowarp_v4', 'meta'],
                    help='geowarp_v4 (default) or meta (BinauralNetwork)')
parser.add_argument('--blocks', type=int, default=3, help='blocks for meta model')
args = parser.parse_args()

SUBJECT_ANGLES = {
    'subject1': -90.0, 'subject2': -60.0, 'subject3': -30.0, 'subject4': 0.0,
    'subject5': 30.0,  'subject6': 60.0,  'subject7': 90.0,
    'subject8': -45.0, 'subject9': 45.0,  'subject10': -15.0, 'subject11': 15.0,
    'subject12': 75.0, 'subject13': -75.0,
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.model_type == 'meta':
    from src.models import BinauralNetwork
    model = BinauralNetwork(view_dim=7, wavenet_blocks=args.blocks).to(device)
    ckpt = torch.load(args.model_file, map_location=device)
    state = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state, strict=False)
else:
    from src.models_geowarp_film_v6_4 import GeoWarpFiLMNet
    model = GeoWarpFiLMNet().to(device)
    ckpt = torch.load(args.model_file, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
model.eval()

subject_dirs = sorted([d for d in glob.glob(os.path.join(args.dataset_directory, 'subject*'))
                       if os.path.isdir(d)])

errors = []
print(f"{'Subject':<12} {'GT':>6}  {'Pred':>7}  {'Error':>7}")
print("-" * 40)

for subject_dir in subject_dirs:
    name = os.path.basename(subject_dir)
    gt_angle = SUBJECT_ANGLES.get(name)
    if gt_angle is None:
        continue

    mono_np, sr = sf.read(os.path.join(subject_dir, 'mono.wav'), dtype='float32')
    if mono_np.ndim > 1:
        mono_np = mono_np[:, 0]

    # Resample to 48kHz if needed
    if sr != 48000:
        mono_t = torch.from_numpy(mono_np).view(1, 1, -1)
        new_len = int(len(mono_np) * 48000 / sr)
        mono_np = F.interpolate(mono_t, size=new_len, mode='linear', align_corners=False).squeeze().numpy()

    original_len = len(mono_np)

    tx = np.loadtxt(os.path.join(subject_dir, 'tx_positions.txt')).astype(np.float32)
    view_t = torch.from_numpy(tx.T).unsqueeze(0).to(device)  # (1, 7, K)

    # Pad mono to match view length
    target_len = tx.shape[0] * 400
    mono_padded = np.zeros(target_len, dtype=np.float32)
    mono_padded[:min(original_len, target_len)] = mono_np[:min(original_len, target_len)]
    mono_t = torch.from_numpy(mono_padded).view(1, 1, -1).to(device)

    with torch.no_grad():
        if args.model_type == 'meta':
            # Meta BinauralNetwork: chunked forwarding
            chunk = 480000
            rec_field = model.receptive_field() + 1000
            rec_field -= rec_field % 400
            chunks_out = []
            for i in range(0, mono_t.shape[-1], chunk):
                m_c = mono_t[:, :, max(0, i-rec_field):i+chunk].to(device)
                v_c = view_t[:, :, max(0, i-rec_field)//400:(i+chunk)//400]
                out = model(m_c, v_c)['output'].squeeze(0)
                if i > 0:
                    out = out[:, -(m_c.shape[-1]-rec_field):]
                chunks_out.append(out.cpu())
            binaural = torch.cat(chunks_out, dim=-1).clamp(-1, 1)
            L = binaural[0].numpy()[:original_len]
            R = binaural[1].numpy()[:original_len]
        else:
            y_L, y_R, _, _, _, _ = model(mono_t, view_t)
            L = y_L.squeeze().cpu().numpy()[:original_len]
            R = y_R.squeeze().cpu().numpy()[:original_len]

    # Trim trailing silence (matching evaluate.py)
    valid = np.where(np.abs(L) > 1e-4)[0]
    if len(valid) > 0:
        vlen = int((valid[-1] + 1) * 0.95)
        L, R = L[:vlen], R[:vlen]

    audio = np.stack([L, R])
    pred_angle = gcc_phat_estimate(audio, sample_rate=48000)
    if pred_angle is None:
        print(f"{name:<12} {gt_angle:>+6.1f}°  {'N/A':>7}  {'N/A':>7}")
        continue

    pred_angle = -pred_angle
    err = min(abs(pred_angle - gt_angle), 360.0 - abs(pred_angle - gt_angle))
    errors.append(err)
    print(f"{name:<12} {gt_angle:>+6.1f}°  {pred_angle:>+7.1f}°  {err:>6.1f}°")

print("-" * 40)
print(f"Mean angular error: {np.mean(errors):.1f}°")
