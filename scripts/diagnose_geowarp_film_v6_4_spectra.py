#!/usr/bin/env python
"""Generate spectrum diagnostics for GeoWarpFiLMNet v6.4 blocks."""
import argparse
import json
import os
import sys

sys.path.insert(0, "/home/sbplab/frank/BinauralSpeechSynthesis")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch

from src.models_geowarp_film_v6_4 import GeoWarpFiLMNet


def load_subject(dataset_directory, subject, seconds):
    subject_dir = os.path.join(dataset_directory, subject)
    mono_np, sr = sf.read(os.path.join(subject_dir, "mono.wav"), dtype="float32")
    binaural_np, ref_sr = sf.read(os.path.join(subject_dir, "binaural.wav"), dtype="float32")
    if sr != 48000 or ref_sr != 48000:
        raise ValueError(f"Expected 48 kHz audio, got mono={sr}, binaural={ref_sr}")

    if mono_np.ndim == 1:
        mono_np = mono_np[None, :]
    else:
        mono_np = mono_np.T
    if binaural_np.ndim == 1:
        binaural_np = binaural_np[None, :]
    else:
        binaural_np = binaural_np.T

    view_np = np.loadtxt(os.path.join(subject_dir, "tx_positions.txt")).T.astype(np.float32)

    target_samples = min(int(seconds * 48000), mono_np.shape[1], binaural_np.shape[1])
    target_samples = max(400, target_samples - (target_samples % 400))
    view_frames = target_samples // 400

    mono = torch.from_numpy(mono_np[:, :target_samples]).unsqueeze(0)
    binaural = torch.from_numpy(binaural_np[:, :target_samples]).unsqueeze(0)
    view = torch.from_numpy(view_np[:, :view_frames]).unsqueeze(0)
    return mono, binaural, view


def load_model(path, device):
    model = GeoWarpFiLMNet().to(device)
    ckpt = torch.load(path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    model.eval()
    return model


def stft_mag(audio, n_fft=1024, hop_length=256):
    if audio.dim() == 3:
        audio = audio[0]
    window = torch.hann_window(n_fft, device=audio.device)
    mags = []
    for channel in range(audio.shape[0]):
        spec = torch.stft(
            audio[channel],
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=window,
            return_complex=True,
        )
        mags.append(spec.abs().detach().cpu().numpy())
    return np.stack(mags, axis=0)


def ipd(spec_l, spec_r, bins=32):
    z = spec_l[:bins] * np.conj(spec_r[:bins])
    return np.angle(z)


def channel_stft(audio, n_fft=1024, hop_length=256):
    if audio.dim() == 3:
        audio = audio[0]
    window = torch.hann_window(n_fft, device=audio.device)
    out = []
    for channel in range(audio.shape[0]):
        out.append(
            torch.stft(
                audio[channel],
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=n_fft,
                window=window,
                return_complex=True,
            )
            .detach()
            .cpu()
            .numpy()
        )
    return out


def mean_spectrum_db(mag):
    spec = mag.mean(axis=-1)
    spec = 20 * np.log10(np.maximum(spec, 1e-8))
    return spec


def save_spectrum_plot(path, title, series):
    plt.figure(figsize=(12, 6))
    for label, values in series:
        plt.plot(values, label=label, linewidth=1.2)
    plt.title(title)
    plt.xlabel("Frequency bin")
    plt.ylabel("Mean magnitude (dB)")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_heatmap(path, title, image, ylabel):
    plt.figure(figsize=(12, 5))
    plt.imshow(image, aspect="auto", origin="lower", interpolation="nearest")
    plt.title(title)
    plt.xlabel("Frame")
    plt.ylabel(ylabel)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_neural_activation_spectrum(path, title, activation):
    act = activation[0]
    spectrum = torch.fft.rfft(act, dim=-1).abs().mean(dim=0)
    values = 20 * np.log10(np.maximum(spectrum.detach().cpu().numpy(), 1e-8))
    save_spectrum_plot(path, title, [("activation", values)])


def spectral_l1(a, b):
    return float(np.mean(np.abs(mean_spectrum_db(a) - mean_spectrum_db(b))))


def summarize_audio(label, audio, gt_audio):
    pred_stft = channel_stft(audio)
    gt_stft = channel_stft(gt_audio)
    pred_mag = np.stack([np.abs(pred_stft[0]), np.abs(pred_stft[1])], axis=0)
    gt_mag = np.stack([np.abs(gt_stft[0]), np.abs(gt_stft[1])], axis=0)

    pred_ipd = ipd(pred_stft[0], pred_stft[1])
    gt_ipd = ipd(gt_stft[0], gt_stft[1])
    diff = np.angle(np.exp(1j * (pred_ipd - gt_ipd)))

    pred_ild = 20 * np.log10((pred_mag[0, :32] + 1e-8) / (pred_mag[1, :32] + 1e-8))
    gt_ild = 20 * np.log10((gt_mag[0, :32] + 1e-8) / (gt_mag[1, :32] + 1e-8))

    return {
        "label": label,
        "spectral_l1_db": spectral_l1(pred_mag, gt_mag),
        "low_freq_ipd_mae_rad": float(np.mean(np.abs(diff))),
        "low_freq_ild_mae_db": float(np.mean(np.abs(pred_ild - gt_ild))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_file", default="geowarp_film_v6_4/best.net")
    parser.add_argument("--dataset_directory", default="dataset/testset")
    parser.add_argument("--subject", default="subject4")
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--output_dir", default="geowarp_film_v6_4/spectra_diagnostics")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model_file, device)

    mono, gt, view = load_subject(args.dataset_directory, args.subject, args.seconds)
    mono, gt, view = mono.to(device), gt.to(device), view.to(device)

    with torch.no_grad():
        outputs, debug = model(mono, view, return_debug=True)

    pred = debug["final_audio"]
    geo = debug["geometric_warp_audio"]
    neural = debug["neural_warp"]["neural_warp_audio"]

    gt_mag = stft_mag(gt)
    geo_mag = stft_mag(geo)
    neural_mag = stft_mag(neural)
    pred_mag = stft_mag(pred)

    save_spectrum_plot(
        os.path.join(args.output_dir, "00_audio_stage_mean_spectrum.png"),
        f"{args.subject}: audio-stage mean spectra",
        [
            ("GT L", mean_spectrum_db(gt_mag)[0]),
            ("GT R", mean_spectrum_db(gt_mag)[1]),
            ("Geo L", mean_spectrum_db(geo_mag)[0]),
            ("Geo R", mean_spectrum_db(geo_mag)[1]),
            ("NeuralWarp L", mean_spectrum_db(neural_mag)[0]),
            ("NeuralWarp R", mean_spectrum_db(neural_mag)[1]),
            ("Final L", mean_spectrum_db(pred_mag)[0]),
            ("Final R", mean_spectrum_db(pred_mag)[1]),
        ],
    )

    save_heatmap(
        os.path.join(args.output_dir, "01_delta_warpfield.png"),
        "Neural warp delta field",
        debug["neural_warp"]["delta_warpfield"][0].detach().cpu().numpy(),
        "Channel",
    )

    for idx, activation in enumerate(debug["neural_warp"]["neural_warp_activations"], start=1):
        save_neural_activation_spectrum(
            os.path.join(args.output_dir, f"neural_warp_layer_{idx:02d}_temporal_spectrum.png"),
            f"Neural warp layer {idx}: temporal activation spectrum",
            activation,
        )

    summaries = [
        summarize_audio("geometric_warp", geo, gt),
        summarize_audio("neural_warp", neural, gt),
        summarize_audio("final_output", pred, gt),
    ]

    for idx, block in enumerate(debug["film_blocks"], start=1):
        block_in = block["input"][0].abs().mean(axis=0).detach().cpu().numpy()
        conv_out = block["conv_out"][0].abs().mean(axis=0).detach().cpu().numpy()
        film_out = block["film_out"][0].abs().mean(axis=0).detach().cpu().numpy()
        out = block["output"][0].abs().mean(axis=0).detach().cpu().numpy()

        save_spectrum_plot(
            os.path.join(args.output_dir, f"film_block_{idx:02d}_mean_spectrum.png"),
            f"FiLM block {idx}: activation spectra",
            [
                ("input", 20 * np.log10(np.maximum(block_in.mean(axis=-1), 1e-8))),
                ("conv_out", 20 * np.log10(np.maximum(conv_out.mean(axis=-1), 1e-8))),
                ("film_out", 20 * np.log10(np.maximum(film_out.mean(axis=-1), 1e-8))),
                ("output", 20 * np.log10(np.maximum(out.mean(axis=-1), 1e-8))),
            ],
        )

        gamma = block["film"]["gamma_delta"][0].detach().cpu().numpy()
        beta = block["film"]["beta"][0].detach().cpu().numpy()
        save_heatmap(
            os.path.join(args.output_dir, f"film_block_{idx:02d}_gamma_delta.png"),
            f"FiLM block {idx}: gamma_delta",
            gamma,
            "Frequency bin",
        )
        save_heatmap(
            os.path.join(args.output_dir, f"film_block_{idx:02d}_beta.png"),
            f"FiLM block {idx}: beta",
            beta,
            "Frequency bin",
        )

    band_ids = debug["film_blocks"][0]["film"]["band_ids"].numpy().tolist()
    unique, counts = np.unique(np.array(band_ids), return_counts=True)
    summary = {
        "model_file": args.model_file,
        "dataset_directory": args.dataset_directory,
        "subject": args.subject,
        "seconds": args.seconds,
        "audio_stage_metrics": summaries,
        "film_band_counts": {str(int(k)): int(v) for k, v in zip(unique, counts)},
        "expected_learning_signal": [
            "neural_warp should reduce low_freq_ipd_mae_rad versus geometric_warp",
            "final_output should reduce spectral_l1_db and low_freq_ild_mae_db versus neural_warp",
            "FiLM gamma/beta should concentrate changes in low-frequency bins for ITD/IPD corrections and broader spectral bands for ILD/timbre",
        ],
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Wrote spectra diagnostics to {args.output_dir}")


if __name__ == "__main__":
    main()
