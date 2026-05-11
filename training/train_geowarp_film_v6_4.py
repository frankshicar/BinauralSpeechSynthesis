"""
GeoWarpFiLMNet v6.4 training script.

v6.4 changes vs v6.3:
1. Fixed log-frequency FiLM bands.
2. Actual 6-layer NeuralWarpCorrector.
3. Identity-friendly FiLM modulation.
4. Low-frequency IPD + differentiable ITD loss for spatial cue supervision.
5. Partial warm start from v6 baseline when shapes match.
"""
import os
import sys
import json

sys.path.insert(0, "/home/sbplab/frank/BinauralSpeechSynthesis")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from src.dataset import BinauralDataset
from src.losses import L2Loss, PhaseLoss
from src.models_geowarp_film_v6_4 import GeoWarpFiLMNet


def atomic_save(state_dict, path):
    tmp_path = path + ".tmp"
    torch.save(state_dict, tmp_path)
    os.replace(tmp_path, path)


config = {
    "train_dir": "dataset/trainset",
    "val_dir": "dataset/testset",
    "output_dir": "geowarp_film_v6_4",
    "checkpoint": "geowarp_film_v6_4/best.net",
    "log_file": "geowarp_film_v6_4/train.log",
    "pretrain": "geowarp_film_v6/best.net",
    "stage1_epochs": 20,
    "stage2_epochs": 80,
    "lr": 5e-4,
    "batch_size": 16,
    "patience": 20,
    "low_freq_bins": 32,
    "monitor_every": 1,
    "monitor_dir": "geowarp_film_v6_4/training_spectra",
    "monitor_max_examples": 1,
    "film_gamma_limit": 0.25,
    "film_beta_limit": 0.25,
    "w_l2": None,
    "w_phase": 1.0,
    "w_ipd": 1.0,
    "w_itd": 0.25,
}


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def mean_spectrum_db_from_feature(feature):
    arr = to_numpy(feature.abs())
    while arr.ndim > 2:
        arr = arr.mean(axis=0)
    if arr.ndim == 2:
        arr = arr.mean(axis=-1)
    return 20 * np.log10(np.maximum(arr, 1e-8))


def audio_stft(audio, model):
    if audio.dim() == 3:
        audio = audio[0]
    specs = []
    for channel in range(audio.shape[0]):
        specs.append(
            torch.stft(
                audio[channel],
                n_fft=model.n_fft,
                hop_length=model.hop_length,
                win_length=model.win_length,
                window=model.window,
                return_complex=True,
            )
        )
    return specs


def mean_audio_spectrum_db(audio, model):
    specs = audio_stft(audio, model)
    return [
        20 * np.log10(np.maximum(to_numpy(spec.abs().mean(dim=-1)), 1e-8))
        for spec in specs
    ]


def get_band_slices(num_bins):
    low_end = max(1, int(round(num_bins * 0.2)))
    mid_end = max(low_end + 1, int(round(num_bins * 0.6)))
    mid_end = min(mid_end, num_bins)
    return {
        "low": slice(0, low_end),
        "mid": slice(low_end, mid_end),
        "high": slice(mid_end, num_bins),
    }


def summarize_band_values(values):
    band_slices = get_band_slices(values.shape[0])
    return {
        name: float(values[band_slice].mean().item()) if values[band_slice].numel() > 0 else 0.0
        for name, band_slice in band_slices.items()
    }


def summarize_band_curve(values):
    band_slices = get_band_slices(values.shape[0])
    out = {}
    for name, band_slice in band_slices.items():
        region = values[band_slice]
        if region.size == 0:
            out[name] = 0.0
        else:
            out[name] = float(region.mean())
    return out


def low_freq_ipd_mae(pred, target, model, bins):
    pred_l, pred_r = audio_stft(pred, model)
    tgt_l, tgt_r = audio_stft(target, model)
    pred_ipd = torch.angle(pred_l[:bins] * pred_r[:bins].conj())
    tgt_ipd = torch.angle(tgt_l[:bins] * tgt_r[:bins].conj())
    diff = torch.atan2(torch.sin(pred_ipd - tgt_ipd), torch.cos(pred_ipd - tgt_ipd))
    return float(diff.abs().mean().item())


def low_freq_ild_mae(pred, target, model, bins):
    pred_l, pred_r = audio_stft(pred, model)
    tgt_l, tgt_r = audio_stft(target, model)
    pred_ild = 20 * torch.log10((pred_l[:bins].abs() + 1e-8) / (pred_r[:bins].abs() + 1e-8))
    tgt_ild = 20 * torch.log10((tgt_l[:bins].abs() + 1e-8) / (tgt_r[:bins].abs() + 1e-8))
    return float((pred_ild - tgt_ild).abs().mean().item())


def spectral_l1_db(pred, target, model):
    pred_spec = mean_audio_spectrum_db(pred, model)
    tgt_spec = mean_audio_spectrum_db(target, model)
    diffs = [np.mean(np.abs(p - t)) for p, t in zip(pred_spec, tgt_spec)]
    return float(np.mean(diffs))


def spectral_error_curve_db(pred, target, model):
    pred_spec = mean_audio_spectrum_db(pred, model)
    tgt_spec = mean_audio_spectrum_db(target, model)
    return np.mean([np.abs(p - t) for p, t in zip(pred_spec, tgt_spec)], axis=0)


def summarize_stage(label, audio, gt, model):
    return {
        "label": label,
        "spectral_l1_db": spectral_l1_db(audio, gt, model),
        "low_freq_ipd_mae_rad": low_freq_ipd_mae(audio, gt, model, config["low_freq_bins"]),
        "low_freq_ild_mae_db": low_freq_ild_mae(audio, gt, model, config["low_freq_bins"]),
    }


def save_line_plot(path, title, series, xlabel, ylabel):
    plt.figure(figsize=(12, 6))
    for label, values in series:
        plt.plot(values, label=label, linewidth=1.2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def save_heatmap(path, title, image, ylabel):
    plt.figure(figsize=(12, 5))
    plt.imshow(image, aspect="auto", origin="lower", interpolation="nearest")
    plt.title(title)
    plt.xlabel("Frame")
    plt.ylabel(ylabel)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def save_neural_activation_spectrum(path, title, activation):
    # Neural warp activations are temporal 1-D features. Plot rFFT over time.
    act = activation[0]
    spectrum = torch.fft.rfft(act, dim=-1).abs().mean(dim=0)
    values = 20 * np.log10(np.maximum(to_numpy(spectrum), 1e-8))
    save_line_plot(path, title, [("activation", values)], "Temporal frequency bin", "Mean activation (dB)")


@torch.no_grad()
def save_training_spectra_monitor(model, loader, device, stage, epoch, log_print):
    if config["monitor_every"] <= 0 or epoch % config["monitor_every"] != 0:
        return

    model.eval()
    mono, binaural, view = next(iter(loader))
    mono = mono[: config["monitor_max_examples"]].to(device)
    binaural = binaural[: config["monitor_max_examples"]].to(device)
    view = view[: config["monitor_max_examples"]].to(device)

    outputs, debug = model(mono, view, return_debug=True)
    del outputs

    out_dir = os.path.join(config["monitor_dir"], f"stage{stage}_epoch{epoch:03d}")
    ensure_dir(out_dir)

    gt = binaural[0]
    geo = debug["geometric_warp_audio"][0]
    neural = debug["neural_warp"]["neural_warp_audio"][0]
    final = debug["final_audio"][0]

    gt_l, gt_r = mean_audio_spectrum_db(gt, model)
    geo_l, geo_r = mean_audio_spectrum_db(geo, model)
    neural_l, neural_r = mean_audio_spectrum_db(neural, model)
    final_l, final_r = mean_audio_spectrum_db(final, model)

    save_line_plot(
        os.path.join(out_dir, "00_audio_stage_mean_spectrum.png"),
        f"Stage {stage} epoch {epoch}: audio-stage mean spectra",
        [
            ("GT L", gt_l),
            ("GT R", gt_r),
            ("Geo L", geo_l),
            ("Geo R", geo_r),
            ("NeuralWarp L", neural_l),
            ("NeuralWarp R", neural_r),
            ("Final L", final_l),
            ("Final R", final_r),
        ],
        "Frequency bin",
        "Mean magnitude (dB)",
    )

    save_heatmap(
        os.path.join(out_dir, "01_delta_warpfield.png"),
        "Neural warp delta field",
        to_numpy(debug["neural_warp"]["delta_warpfield"][0]),
        "Channel",
    )

    neural_err_curve = spectral_error_curve_db(neural, gt, model)
    final_err_curve = spectral_error_curve_db(final, gt, model)
    save_line_plot(
        os.path.join(out_dir, "02_spectral_error_before_after_film.png"),
        f"Stage {stage} epoch {epoch}: spectral error before/after FiLM",
        [
            ("neural_warp_vs_gt", neural_err_curve),
            ("final_vs_gt", final_err_curve),
        ],
        "Frequency bin",
        "Absolute spectral error (dB)",
    )

    for idx, activation in enumerate(debug["neural_warp"]["neural_warp_activations"], start=1):
        save_neural_activation_spectrum(
            os.path.join(out_dir, f"neural_warp_layer_{idx:02d}_temporal_spectrum.png"),
            f"Neural warp layer {idx}: temporal activation spectrum",
            activation,
        )

    film_summary = []
    for idx, block in enumerate(debug["film_blocks"], start=1):
        block_in = mean_spectrum_db_from_feature(block["input"][0])
        conv_out = mean_spectrum_db_from_feature(block["conv_out"][0])
        film_out = mean_spectrum_db_from_feature(block["film_out"][0])
        out = mean_spectrum_db_from_feature(block["output"][0])

        save_line_plot(
            os.path.join(out_dir, f"film_block_{idx:02d}_mean_spectrum.png"),
            f"FiLM block {idx}: activation spectra",
            [
                ("input", block_in),
                ("conv_out", conv_out),
                ("film_out", film_out),
                ("output", out),
            ],
            "Frequency bin",
            "Mean activation (dB)",
        )

        gamma_logits = debug["film_blocks"][idx - 1]["film"]["gamma_logits"][0]
        beta_logits = debug["film_blocks"][idx - 1]["film"]["beta_logits"][0]
        gamma = debug["film_blocks"][idx - 1]["film"]["gamma_delta"][0]
        beta = debug["film_blocks"][idx - 1]["film"]["beta"][0]
        gamma_abs = float(gamma.abs().mean().item())
        beta_abs = float(beta.abs().mean().item())
        gamma_band_curve = to_numpy(gamma.abs().mean(dim=-1))
        beta_band_curve = to_numpy(beta.abs().mean(dim=-1))
        save_line_plot(
            os.path.join(out_dir, f"film_block_{idx:02d}_band_activity.png"),
            f"FiLM block {idx}: per-band activity",
            [
                ("|gamma_delta|", gamma_band_curve),
                ("|beta|", beta_band_curve),
            ],
            "Frequency bin",
            "Mean modulation magnitude",
        )
        save_heatmap(
            os.path.join(out_dir, f"film_block_{idx:02d}_gamma_delta.png"),
            f"FiLM block {idx}: gamma_delta",
            to_numpy(gamma),
            "Frequency bin",
        )
        save_heatmap(
            os.path.join(out_dir, f"film_block_{idx:02d}_beta.png"),
            f"FiLM block {idx}: beta",
            to_numpy(beta),
            "Frequency bin",
        )
        gamma_band_summary = summarize_band_values(gamma.abs())
        beta_band_summary = summarize_band_values(beta.abs())
        gamma_logit_summary = summarize_band_values(gamma_logits.abs())
        beta_logit_summary = summarize_band_values(beta_logits.abs())
        film_summary.append(
            {
                "block": idx,
                "gamma_abs_mean": gamma_abs,
                "beta_abs_mean": beta_abs,
                "film_delta_db_mean": float(np.mean(np.abs(film_out - conv_out))),
                "identity_like": gamma_abs < 0.01 and beta_abs < 0.01,
                "gamma_abs_by_band": gamma_band_summary,
                "beta_abs_by_band": beta_band_summary,
                "gamma_logit_abs_by_band": gamma_logit_summary,
                "beta_logit_abs_by_band": beta_logit_summary,
            }
        )

    stage_metrics = [
        summarize_stage("geometric_warp", geo, gt, model),
        summarize_stage("neural_warp", neural, gt, model),
        summarize_stage("final_output", final, gt, model),
    ]
    summary = {
        "stage": stage,
        "epoch": epoch,
        "output_dir": out_dir,
        "audio_stage_metrics": stage_metrics,
        "film_blocks": film_summary,
        "spectral_error_by_band_db": {
            "neural_warp": summarize_band_curve(neural_err_curve),
            "final_output": summarize_band_curve(final_err_curve),
        },
        "learning_checks": {
            "neural_warp_ipd_improved": (
                stage_metrics[1]["low_freq_ipd_mae_rad"]
                < stage_metrics[0]["low_freq_ipd_mae_rad"]
            ),
            "final_spectrum_improved": (
                stage_metrics[2]["spectral_l1_db"]
                < stage_metrics[1]["spectral_l1_db"]
            ),
            "final_ild_improved": (
                stage_metrics[2]["low_freq_ild_mae_db"]
                < stage_metrics[1]["low_freq_ild_mae_db"]
            ),
        },
    }

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(config["monitor_dir"], "summary.jsonl"), "a") as f:
        f.write(json.dumps(summary) + "\n")

    report_lines = [
        f"# v6.4 Training Spectra Monitor: stage {stage}, epoch {epoch}",
        "",
        "## Learning checks",
        f"- Neural warp IPD: {stage_metrics[0]['low_freq_ipd_mae_rad']:.4f} -> "
        f"{stage_metrics[1]['low_freq_ipd_mae_rad']:.4f} "
        f"({'improved' if summary['learning_checks']['neural_warp_ipd_improved'] else 'not improved'})",
        f"- Final spectrum L1: {stage_metrics[1]['spectral_l1_db']:.2f} -> "
        f"{stage_metrics[2]['spectral_l1_db']:.2f} dB "
        f"({'improved' if summary['learning_checks']['final_spectrum_improved'] else 'not improved'})",
        f"- Final low-frequency ILD: {stage_metrics[1]['low_freq_ild_mae_db']:.2f} -> "
        f"{stage_metrics[2]['low_freq_ild_mae_db']:.2f} dB "
        f"({'improved' if summary['learning_checks']['final_ild_improved'] else 'not improved'})",
        "",
        "## FiLM block activity",
    ]
    for block in film_summary:
        state = "identity-like" if block["identity_like"] else "active"
        report_lines.append(
            f"- Block {block['block']:02d}: gamma_abs={block['gamma_abs_mean']:.5f}, "
            f"beta_abs={block['beta_abs_mean']:.5f}, "
            f"film_delta={block['film_delta_db_mean']:.3f} dB, {state}, "
            f"gamma(low/mid/high)="
            f"{block['gamma_abs_by_band']['low']:.4f}/"
            f"{block['gamma_abs_by_band']['mid']:.4f}/"
            f"{block['gamma_abs_by_band']['high']:.4f}, "
            f"beta(low/mid/high)="
            f"{block['beta_abs_by_band']['low']:.4f}/"
            f"{block['beta_abs_by_band']['mid']:.4f}/"
            f"{block['beta_abs_by_band']['high']:.4f}"
        )
    report_lines.extend(
        [
            "",
            "## Frequency-region error",
            f"- Neural warp spectral error low/mid/high: "
            f"{summary['spectral_error_by_band_db']['neural_warp']['low']:.2f}/"
            f"{summary['spectral_error_by_band_db']['neural_warp']['mid']:.2f}/"
            f"{summary['spectral_error_by_band_db']['neural_warp']['high']:.2f} dB",
            f"- Final output spectral error low/mid/high: "
            f"{summary['spectral_error_by_band_db']['final_output']['low']:.2f}/"
            f"{summary['spectral_error_by_band_db']['final_output']['mid']:.2f}/"
            f"{summary['spectral_error_by_band_db']['final_output']['high']:.2f} dB",
            "",
            "## Interpretation rule",
            "- We want neural_warp IPD to improve versus geometric_warp.",
            "- We want final_output spectrum and low-frequency ILD to improve versus neural_warp.",
            "- Early FiLM blocks can stay close to identity at the start, but should become active after training progresses.",
        ]
    )
    with open(os.path.join(out_dir, "analysis.md"), "w") as f:
        f.write("\n".join(report_lines) + "\n")

    log_print(
        "  Spectra monitor -> "
        f"{out_dir} | "
        f"IPD geo/neural/final="
        f"{stage_metrics[0]['low_freq_ipd_mae_rad']:.4f}/"
        f"{stage_metrics[1]['low_freq_ipd_mae_rad']:.4f}/"
        f"{stage_metrics[2]['low_freq_ipd_mae_rad']:.4f} | "
        f"specL1 neural/final="
        f"{stage_metrics[1]['spectral_l1_db']:.2f}/"
        f"{stage_metrics[2]['spectral_l1_db']:.2f} dB"
    )
    active_blocks = sum(1 for block in film_summary if not block["identity_like"])
    log_print(
        f"  Monitor checks -> neural_ipd={summary['learning_checks']['neural_warp_ipd_improved']} "
        f"final_spec={summary['learning_checks']['final_spectrum_improved']} "
        f"final_ild={summary['learning_checks']['final_ild_improved']} "
        f"active_film_blocks={active_blocks}/{len(film_summary)}"
    )
    log_print(
        "  Region error dB -> "
        f"neural(low/mid/high)="
        f"{summary['spectral_error_by_band_db']['neural_warp']['low']:.2f}/"
        f"{summary['spectral_error_by_band_db']['neural_warp']['mid']:.2f}/"
        f"{summary['spectral_error_by_band_db']['neural_warp']['high']:.2f} | "
        f"final(low/mid/high)="
        f"{summary['spectral_error_by_band_db']['final_output']['low']:.2f}/"
        f"{summary['spectral_error_by_band_db']['final_output']['mid']:.2f}/"
        f"{summary['spectral_error_by_band_db']['final_output']['high']:.2f}"
    )
    model.train()


def load_partial_pretrain(model, checkpoint_path, device):
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        print("No pretrain checkpoint loaded.")
        return

    ckpt = torch.load(checkpoint_path, map_location=device)
    source = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    target = model.state_dict()

    compatible = {}
    skipped = []
    for key, value in source.items():
        if ".film.film_net." in key:
            skipped.append(key)
            continue
        if key in target and target[key].shape == value.shape:
            compatible[key] = value
        else:
            skipped.append(key)

    target.update(compatible)
    model.load_state_dict(target)
    print(
        f"Loaded partial pretrain from {checkpoint_path}: "
        f"{len(compatible)} tensors, skipped {len(skipped)}"
    )
    if skipped:
        print("  Skipped first 8:", ", ".join(skipped[:8]))


def ipd_loss_fn(Y_L, Y_R, Y_L_gt, Y_R_gt, low_freq_bins=32):
    if torch.isnan(Y_L).any() or torch.isnan(Y_R).any():
        return torch.tensor(0.0, device=Y_L.device, requires_grad=True)

    pred_ipd = torch.angle(Y_L[:, :low_freq_bins, :] * Y_R[:, :low_freq_bins, :].conj())
    gt_ipd = torch.angle(Y_L_gt[:, :low_freq_bins, :] * Y_R_gt[:, :low_freq_bins, :].conj())
    diff = torch.atan2(torch.sin(pred_ipd - gt_ipd), torch.cos(pred_ipd - gt_ipd))

    energy = (Y_L_gt[:, :low_freq_bins, :].abs() + Y_R_gt[:, :low_freq_bins, :].abs()) / 2
    mask = energy > 0.1 * energy.mean()
    if mask.sum() == 0:
        return torch.tensor(0.0, device=Y_L.device, requires_grad=True)

    loss = torch.sin(diff / 2).pow(2)
    loss = loss[mask].mean()
    return loss if not torch.isnan(loss) else torch.tensor(0.0, device=Y_L.device, requires_grad=True)


def itd_loss_fn(pred, target, sample_rate=48000, max_freq=1500):
    """Differentiable low-frequency GCC-PHAT phase loss."""
    pred_L = torch.fft.rfft(pred[:, 0, :])
    pred_R = torch.fft.rfft(pred[:, 1, :])
    tgt_L = torch.fft.rfft(target[:, 0, :])
    tgt_R = torch.fft.rfft(target[:, 1, :])

    pred_cross = pred_L * pred_R.conj()
    tgt_cross = tgt_L * tgt_R.conj()
    pred_cross = pred_cross / (pred_cross.abs() + 1e-8)
    tgt_cross = tgt_cross / (tgt_cross.abs() + 1e-8)

    cutoff = max(1, int(max_freq * pred.shape[-1] / sample_rate))
    pred_phase = torch.angle(pred_cross[:, :cutoff])
    tgt_phase = torch.angle(tgt_cross[:, :cutoff])
    diff = torch.atan2(torch.sin(pred_phase - tgt_phase), torch.cos(pred_phase - tgt_phase))
    return diff.pow(2).mean()


def stft_targets(model, y_L_gt, y_R_gt):
    Y_L_gt = torch.stft(
        y_L_gt.squeeze(1),
        n_fft=model.n_fft,
        hop_length=model.hop_length,
        win_length=model.win_length,
        window=model.window,
        return_complex=True,
    )
    Y_R_gt = torch.stft(
        y_R_gt.squeeze(1),
        n_fft=model.n_fft,
        hop_length=model.hop_length,
        win_length=model.win_length,
        window=model.window,
        return_complex=True,
    )
    return Y_L_gt, Y_R_gt


def run_epoch(model, loader, optimizer, device, stage, weights):
    model.train()
    l2_fn = L2Loss()
    phase_fn = PhaseLoss(sample_rate=48000, ignore_below=0.2)

    sums = {"total": 0, "l2": 0, "phase": 0, "ipd": 0, "itd": 0, "mag_anchor": 0}
    n = 0

    for mono, binaural, view in loader:
        mono, binaural, view = mono.to(device), binaural.to(device), view.to(device)
        y_L_gt, y_R_gt = binaural[:, 0:1], binaural[:, 1:2]

        y_L, y_R, Y_L, Y_R, Y_L_init, Y_R_init = model(mono, view)
        pred = torch.cat([y_L, y_R], dim=1)
        gt = torch.cat([y_L_gt, y_R_gt], dim=1)
        Y_L_gt, Y_R_gt = stft_targets(model, y_L_gt, y_R_gt)

        l2 = l2_fn(pred, gt)
        phase = phase_fn(pred, gt)
        ipd = ipd_loss_fn(Y_L, Y_R, Y_L_gt, Y_R_gt, config["low_freq_bins"])
        itd = itd_loss_fn(pred, gt)

        if stage == 1:
            mag_anchor = (
                torch.nn.functional.l1_loss(Y_L.abs(), Y_L_init.abs().detach())
                + torch.nn.functional.l1_loss(Y_R.abs(), Y_R_init.abs().detach())
            ) / 2
            loss = 0.1 * mag_anchor + 0.1 * phase + weights["itd"] * itd
            sums["mag_anchor"] += mag_anchor.item()
        else:
            loss = (
                weights["l2"] * l2
                + weights["phase"] * phase
                + weights["ipd"] * ipd
                + weights["itd"] * itd
            )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        sums["total"] += loss.item()
        sums["l2"] += l2.item()
        sums["phase"] += phase.item()
        sums["ipd"] += ipd.item()
        sums["itd"] += itd.item()
        n += 1

    return {key: value / max(n, 1) for key, value in sums.items()}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    l2_fn = L2Loss()
    phase_fn = PhaseLoss(sample_rate=48000, ignore_below=0.2)

    sums = {"l2": 0, "phase": 0, "ipd": 0, "itd": 0}
    n = 0

    for mono, binaural, view in loader:
        mono, binaural, view = mono.to(device), binaural.to(device), view.to(device)
        y_L_gt, y_R_gt = binaural[:, 0:1], binaural[:, 1:2]
        y_L, y_R, Y_L, Y_R, _, _ = model(mono, view)
        pred = torch.cat([y_L, y_R], dim=1)
        gt = torch.cat([y_L_gt, y_R_gt], dim=1)

        if torch.isnan(pred).any() or torch.isnan(gt).any():
            continue

        Y_L_gt, Y_R_gt = stft_targets(model, y_L_gt, y_R_gt)
        sums["l2"] += l2_fn(pred, gt).item()
        sums["phase"] += phase_fn(pred, gt).item()
        sums["ipd"] += ipd_loss_fn(Y_L, Y_R, Y_L_gt, Y_R_gt, config["low_freq_bins"]).item()
        sums["itd"] += itd_loss_fn(pred, gt).item()
        n += 1

    return {key: value / max(n, 1) for key, value in sums.items()}


def calibrate_weights(model, loader, device):
    model.eval()
    l2_fn = L2Loss()
    phase_fn = PhaseLoss(sample_rate=48000, ignore_below=0.2)

    mono, binaural, view = next(iter(loader))
    mono, binaural, view = mono.to(device), binaural.to(device), view.to(device)
    y_L_gt, y_R_gt = binaural[:, 0:1], binaural[:, 1:2]

    with torch.no_grad():
        y_L, y_R, Y_L, Y_R, _, _ = model(mono, view)
        pred = torch.cat([y_L, y_R], dim=1)
        gt = torch.cat([y_L_gt, y_R_gt], dim=1)
        Y_L_gt, Y_R_gt = stft_targets(model, y_L_gt, y_R_gt)

        l2 = l2_fn(pred, gt).item()
        phase = phase_fn(pred, gt).item()
        ipd = ipd_loss_fn(Y_L, Y_R, Y_L_gt, Y_R_gt, config["low_freq_bins"]).item()
        itd = itd_loss_fn(pred, gt).item()

    ref = max(phase, 1e-4)
    weights = {
        "l2": config["w_l2"] if config["w_l2"] is not None else ref / (l2 + 1e-8),
        "phase": config["w_phase"],
        "ipd": config["w_ipd"],
        "itd": config["w_itd"],
    }
    print(
        f"  Raw losses -> L2: {l2:.6f}  Phase: {phase:.4f} "
        f"IPD: {ipd:.4f}  ITD: {itd:.4f}"
    )
    print(
        f"  Weights -> L2: {weights['l2']:.2f}  Phase: {weights['phase']:.2f} "
        f"IPD: {weights['ipd']:.2f}  ITD: {weights['itd']:.2f}"
    )
    model.train()
    return weights


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(config["output_dir"], exist_ok=True)

    model = GeoWarpFiLMNet(
        gamma_limit=config["film_gamma_limit"],
        beta_limit=config["film_beta_limit"],
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Neural warp layers: {len(model.neural_warp.convs)}")
    load_partial_pretrain(model, config["pretrain"], device)

    train_ds = BinauralDataset(config["train_dir"], chunk_size_ms=200, overlap=0.5)
    val_ds = BinauralDataset(config["val_dir"], chunk_size_ms=200, overlap=0.5)
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, num_workers=0)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    print("\nCalibrating loss weights...")
    weights = calibrate_weights(model, train_loader, device)

    with open(config["log_file"], "w", buffering=1) as log:
        def log_print(msg):
            print(msg)
            log.write(msg + "\n")

        best_metric = float("inf")
        patience = 0

        log_print("\n=== Initial spectra monitor ===")
        save_training_spectra_monitor(model, val_loader, device, stage=0, epoch=0, log_print=log_print)

        optimizer = optim.Adam(model.parameters(), lr=config["lr"])
        log_print("\n=== Stage 1: mag_anchor + phase + ITD ===")
        for epoch in range(1, config["stage1_epochs"] + 1):
            tr = run_epoch(model, train_loader, optimizer, device, stage=1, weights=weights)
            val = evaluate(model, val_loader, device)
            log_print(
                f"[S1] Ep {epoch:3d} | train={tr['total']:.4f} "
                f"(mag={tr['mag_anchor']:.4f} ph={tr['phase']:.3f} itd={tr['itd']:.4f}) | "
                f"val_l2={val['l2']*1000:.3f}e-3 val_phase={val['phase']:.3f} "
                f"val_ipd={val['ipd']:.4f} val_itd={val['itd']:.4f}"
            )
            save_training_spectra_monitor(model, val_loader, device, stage=1, epoch=epoch, log_print=log_print)

            metric = val["phase"] + 0.25 * val["itd"]
            if metric < best_metric:
                best_metric = metric
                patience = 0
                atomic_save(
                    {
                        "epoch": epoch,
                        "stage": 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_metric": best_metric,
                        "loss_weights": weights,
                        "config": config,
                    },
                    config["checkpoint"],
                )
                log_print(f"  Best metric: {best_metric:.4f}")
            else:
                patience += 1
                if patience >= config["patience"]:
                    log_print("  Early stop (stage 1)")
                    break

        log_print("\nReloading best Stage 1 checkpoint...")
        ckpt = torch.load(config["checkpoint"], map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

        log_print("Recalibrating loss weights after Stage 1...")
        weights = calibrate_weights(model, train_loader, device)

        best_metric = float("inf")
        patience = 0
        optimizer = optim.Adam(model.parameters(), lr=config["lr"] / 3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        log_print("\n=== Stage 2: L2 + Phase + IPD + ITD ===")
        for epoch in range(1, config["stage2_epochs"] + 1):
            tr = run_epoch(model, train_loader, optimizer, device, stage=2, weights=weights)
            val = evaluate(model, val_loader, device)
            metric = val["phase"] + 0.25 * val["itd"]
            scheduler.step(metric)

            log_print(
                f"[S2] Ep {epoch:3d} | train={tr['total']:.4f} "
                f"(l2={tr['l2']*1000:.3f}e-3 ph={tr['phase']:.3f} "
                f"ipd={tr['ipd']:.4f} itd={tr['itd']:.4f}) | "
                f"val_l2={val['l2']*1000:.3f}e-3 val_phase={val['phase']:.3f} "
                f"val_ipd={val['ipd']:.4f} val_itd={val['itd']:.4f} | "
                f"metric={metric:.4f} lr={optimizer.param_groups[0]['lr']:.1e}"
            )
            save_training_spectra_monitor(model, val_loader, device, stage=2, epoch=epoch, log_print=log_print)

            if metric < best_metric:
                best_metric = metric
                patience = 0
                atomic_save(
                    {
                        "epoch": epoch,
                        "stage": 2,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "best_metric": best_metric,
                        "loss_weights": weights,
                        "config": config,
                    },
                    config["checkpoint"],
                )
                log_print(f"  Best metric: {best_metric:.4f}")
            else:
                patience += 1
                if patience >= config["patience"]:
                    log_print("  Early stop (stage 2)")
                    break

        log_print(f"\nFinal best metric: {best_metric:.4f}")


if __name__ == "__main__":
    main()
