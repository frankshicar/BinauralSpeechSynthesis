"""
GeoWarpFiLMNet v6.4.

Changes vs v6.3:
1. Fix log-frequency FiLM band allocation so all STFT bins are covered.
2. Actually use a 6-layer neural warp corrector by default.
3. Make FiLM modulation identity-friendly: x * (1 + gamma) + beta.
4. Allow phase gradients to reach the neural warp path.
5. Add an optional debug path for spectrum/activation diagnostics.
"""
import math
import sys

sys.path.insert(0, "/home/sbplab/frank/BinauralSpeechSynthesis")

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models import GeometricWarper
from src.warping import MonotoneTimeWarper


class NeuralWarpCorrector(nn.Module):
    """Learn a causal residual correction on top of the geometric warpfield."""

    def __init__(self, view_dim=7, channels=64, layers=6):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(view_dim if layer == 0 else channels, channels, kernel_size=2)
                for layer in range(layers)
            ]
        )
        self.linear = nn.Conv1d(channels, 2, kernel_size=1)
        self.neural_warper = MonotoneTimeWarper()
        self._init_extra_layers_as_identity(start_layer=4)

    def _init_extra_layers_as_identity(self, start_layer):
        for conv in self.convs[start_layer:]:
            if conv.in_channels != conv.out_channels:
                continue
            nn.init.zeros_(conv.weight)
            nn.init.zeros_(conv.bias)
            with torch.no_grad():
                channels = conv.out_channels
                idx = torch.arange(channels)
                conv.weight[idx, idx, -1] = 1.0

    def forward(self, mono, view, geometric_warpfield, return_debug=False):
        x = view
        activations = []
        for conv in self.convs:
            x = F.pad(x, [1, 0])
            x = F.relu(conv(x))
            if return_debug:
                activations.append(x.detach())

        delta = self.linear(x)
        delta = F.interpolate(delta, size=mono.shape[-1], mode="linear", align_corners=False)

        warpfield = geometric_warpfield + delta
        warpfield = -F.relu(-warpfield)

        mono_stereo = torch.cat([mono, mono], dim=1)
        warped = self.neural_warper(mono_stereo, warpfield)

        if not return_debug:
            return warped

        return warped, {
            "neural_warp_activations": activations,
            "delta_warpfield": delta.detach(),
            "corrected_warpfield": warpfield.detach(),
            "neural_warp_audio": warped.detach(),
        }


class TemporalPositionEncoder(nn.Module):
    """Per-frame Fourier position/velocity encoder."""

    def __init__(self, input_dim=7, L=8, output_dim=256):
        super().__init__()
        self.L = L
        freqs = (2.0 ** torch.arange(L)) * torch.pi
        self.register_buffer("freqs", freqs)

        in_ch = input_dim * 4 * L + input_dim
        self.mlp = nn.Sequential(
            nn.Conv1d(in_ch, output_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(output_dim, output_dim, kernel_size=3, padding=1),
        )

    def _fourier(self, x):
        scaled = x.unsqueeze(-1) * self.freqs.view(1, 1, 1, -1)
        enc = torch.cat([scaled.sin(), scaled.cos()], dim=-1)
        return enc.permute(0, 1, 3, 2).flatten(1, 2)

    def forward(self, view, T_stft):
        vel = torch.zeros_like(view)
        vel[:, :, 1:] = view[:, :, 1:] - view[:, :, :-1]

        pos_enc = self._fourier(view)
        vel_enc = self._fourier(vel)
        feat = torch.cat([pos_enc, vel_enc, view], dim=1)
        feat = self.mlp(feat)
        return F.interpolate(feat, size=T_stft, mode="linear", align_corners=False)


class FiLMLayer(nn.Module):
    """Per-frame, log-frequency FiLM modulation with bounded scale."""

    def __init__(
        self,
        channels,
        pos_dim=256,
        num_bands=64,
        band_scale="log",
        gamma_limit=0.25,
        beta_limit=0.25,
    ):
        super().__init__()
        self.num_bands = num_bands
        self.band_scale = band_scale
        self.gamma_limit = gamma_limit
        self.beta_limit = beta_limit
        self.film_net = nn.Conv1d(pos_dim, num_bands * 2, kernel_size=1)
        self.band_ids = None
        self.band_ids_size = None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.film_net.weight)
        nn.init.zeros_(self.film_net.bias)

    def _log_band_edges(self, F):
        raw = torch.round(torch.logspace(0, math.log10(F), self.num_bands + 1)).long()
        edges = [0]
        prev = 0
        for i in range(1, self.num_bands + 1):
            remaining = self.num_bands - i
            max_allowed = F - remaining
            proposed = int(raw[i].item())
            edge = max(prev + 1, min(proposed, max_allowed))
            edges.append(edge)
            prev = edge
        edges[-1] = F
        return edges

    def _linear_band_edges(self, F):
        return [round(F * i / self.num_bands) for i in range(self.num_bands + 1)]

    def _init_band_ids(self, F, device):
        if F < self.num_bands:
            raise ValueError(f"num_bands={self.num_bands} must be <= frequency bins={F}")

        if self.band_scale == "linear":
            band_edges = self._linear_band_edges(F)
        elif self.band_scale == "log":
            band_edges = self._log_band_edges(F)
        else:
            raise ValueError(f"Unsupported band_scale: {self.band_scale}")

        band_ids = torch.empty(F, dtype=torch.long, device=device)
        for band in range(self.num_bands):
            start = band_edges[band]
            end = band_edges[band + 1]
            if end <= start:
                raise ValueError(f"Empty FiLM band {band}: {start} -> {end}")
            band_ids[start:end] = band

        self.band_ids = band_ids
        self.band_ids_size = F

    def forward(self, x, pos_feat, return_debug=False):
        B, C, F_bins, T = x.shape
        if (
            self.band_ids is None
            or self.band_ids.device != x.device
            or self.band_ids_size != F_bins
        ):
            self._init_band_ids(F_bins, x.device)

        params = self.film_net(pos_feat)
        params = params.view(B, self.num_bands, 2, T)

        gamma_logits = params[:, self.band_ids, 0, :]
        beta_logits = params[:, self.band_ids, 1, :]
        gamma_delta = self.gamma_limit * torch.tanh(gamma_logits)
        beta = self.beta_limit * torch.tanh(beta_logits)
        out = x * (1.0 + gamma_delta.unsqueeze(1)) + beta.unsqueeze(1)

        if not return_debug:
            return out

        return out, {
            "gamma_logits": gamma_logits.detach(),
            "beta_logits": beta_logits.detach(),
            "gamma_delta": gamma_delta.detach(),
            "beta": beta.detach(),
            "band_ids": self.band_ids.detach().cpu(),
            "gamma_limit": self.gamma_limit,
            "beta_limit": self.beta_limit,
        }


class FiLMResBlock(nn.Module):
    def __init__(
        self,
        channels,
        pos_dim=256,
        num_bands=64,
        dilation=1,
        band_scale="log",
        gamma_limit=0.25,
        beta_limit=0.25,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=(3, 3),
                padding=(dilation, 1),
                dilation=(dilation, 1),
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=(1, 1)),
            nn.BatchNorm2d(channels),
        )
        self.film = FiLMLayer(
            channels,
            pos_dim,
            num_bands,
            band_scale=band_scale,
            gamma_limit=gamma_limit,
            beta_limit=beta_limit,
        )
        self.relu = nn.ReLU()

    def forward(self, x, pos_feat, return_debug=False):
        residual = x
        conv_out = self.conv(x)
        if return_debug:
            film_out, film_debug = self.film(conv_out, pos_feat, return_debug=True)
        else:
            film_out = self.film(conv_out, pos_feat)
            film_debug = None

        out = self.relu(film_out + residual)
        if not return_debug:
            return out

        return out, {
            "input": residual.detach(),
            "conv_out": conv_out.detach(),
            "film_out": film_out.detach(),
            "output": out.detach(),
            "film": film_debug,
        }


class GeoWarpFiLMNet(nn.Module):
    """Geometric warp + neural warp correction + per-frame log-band FiLM stack."""

    def __init__(
        self,
        n_fft=1024,
        hop_length=256,
        channels=128,
        num_blocks=8,
        fourier_L=8,
        num_bands=64,
        pos_dim=256,
        warp_channels=64,
        warp_layers=6,
        band_scale="log",
        detach_geo_phase=False,
        gamma_limit=0.25,
        beta_limit=0.25,
    ):
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = n_fft
        self.detach_geo_phase = detach_geo_phase

        self.geo_warper = GeometricWarper(sampling_rate=48000)
        self.neural_warp = NeuralWarpCorrector(7, warp_channels, warp_layers)
        self.pos_encoder = TemporalPositionEncoder(7, fourier_L, pos_dim)

        self.encoder = nn.Sequential(
            nn.Conv2d(2, channels // 2, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(),
            nn.Conv2d(channels // 2, channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )

        dilations = [1, 2, 4, 8, 1, 2, 4, 8]
        self.res_blocks = nn.ModuleList(
            [
                FiLMResBlock(
                    channels,
                    pos_dim,
                    num_bands,
                    dilations[i % len(dilations)],
                    band_scale=band_scale,
                    gamma_limit=gamma_limit,
                    beta_limit=beta_limit,
                )
                for i in range(num_blocks)
            ]
        )

        self.output_head = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(channels // 2, 2, kernel_size=(1, 1)),
            nn.Softplus(),
        )
        self.phase_head = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(channels // 2, 2, kernel_size=(1, 1)),
            nn.Tanh(),
        )

        self.register_buffer("window", torch.hann_window(n_fft))

    def _stft(self, x):
        return torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
        )

    def forward(self, mono, view, return_debug=False):
        B, _, T = mono.shape

        geo_wf = self.geo_warper._warpfield(view, T)
        if return_debug:
            mono_stereo = torch.cat([mono, mono], dim=1)
            y_geo = self.neural_warp.neural_warper(mono_stereo, geo_wf)
            y_init, neural_debug = self.neural_warp(
                mono, view, geo_wf, return_debug=True
            )
        else:
            y_geo = None
            neural_debug = None
            y_init = self.neural_warp(mono, view, geo_wf)

        Y_L_init = self._stft(y_init[:, 0])
        Y_R_init = self._stft(y_init[:, 1])

        x = torch.stack([Y_L_init.abs(), Y_R_init.abs()], dim=1)
        T_stft = x.shape[-1]

        pos_feat = self.pos_encoder(view, T_stft)

        x = self.encoder(x)
        block_debug = []
        for block in self.res_blocks:
            if return_debug:
                x, dbg = block(x, pos_feat, return_debug=True)
                block_debug.append(dbg)
            else:
                x = block(x, pos_feat)

        mag_out = self.output_head(x)
        mag_L_out = mag_out[:, 0].clamp(min=1e-6)
        mag_R_out = mag_out[:, 1].clamp(min=1e-6)

        phase_res = self.phase_head(x) * torch.pi
        phase_L_geo = torch.angle(Y_L_init)
        phase_R_geo = torch.angle(Y_R_init)
        if self.detach_geo_phase:
            phase_L_geo = phase_L_geo.detach()
            phase_R_geo = phase_R_geo.detach()

        phase_L = torch.atan2(
            torch.sin(phase_L_geo + phase_res[:, 0]),
            torch.cos(phase_L_geo + phase_res[:, 0]),
        )
        phase_R = torch.atan2(
            torch.sin(phase_R_geo + phase_res[:, 1]),
            torch.cos(phase_R_geo + phase_res[:, 1]),
        )

        Y_L = mag_L_out * torch.exp(1j * phase_L)
        Y_R = mag_R_out * torch.exp(1j * phase_R)

        y_L = torch.istft(
            Y_L,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            length=T,
        )
        y_R = torch.istft(
            Y_R,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            length=T,
        )

        outputs = (y_L.unsqueeze(1), y_R.unsqueeze(1), Y_L, Y_R, Y_L_init, Y_R_init)
        if not return_debug:
            return outputs

        debug = {
            "geometric_warpfield": geo_wf.detach(),
            "geometric_warp_audio": y_geo.detach(),
            "neural_warp": neural_debug,
            "pos_feat": pos_feat.detach(),
            "film_blocks": block_debug,
            "mag_out": mag_out.detach(),
            "phase_res": phase_res.detach(),
            "final_audio": torch.cat([outputs[0], outputs[1]], dim=1).detach(),
        }
        return outputs, debug


if __name__ == "__main__":
    model = GeoWarpFiLMNet()
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,}")
    print(f"Neural warp layers: {len(model.neural_warp.convs)}")

    mono = torch.randn(2, 1, 9600)
    view = torch.randn(2, 7, 24)
    view[:, 3:, :] = F.normalize(view[:, 3:, :], dim=1)

    y_L, y_R, Y_L, Y_R, Y_L_init, Y_R_init = model(mono, view)
    print(f"Output L: {y_L.shape}, R: {y_R.shape}")
    print(f"STFT L: {Y_L.shape}")
