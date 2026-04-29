"""
Integration tests for BinauralSpeechSynthesis.

Run with:  python -m pytest tests/test_integration.py -v
"""

import os
import sys
import math
import tempfile

import pytest
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import BinauralNetwork
from src.models_hybrid_physical import HybridPhysicalLearned

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SR = 48000
CHUNK = 9600   # 200 ms @ 48 kHz
K = CHUNK // 400  # view frames (120 Hz)
B = 2          # batch size kept small for speed


def make_model():
    return HybridPhysicalLearned(sample_rate=SR, n_fft=1024, hop_size=64).to(DEVICE)


def make_binaural_model():
    return BinauralNetwork(use_cuda=DEVICE.type == "cuda")


def random_view(batch=B, k=K):
    return torch.randn(batch, 7, k, device=DEVICE)


def random_mono(batch=B, t=CHUNK):
    return torch.randn(batch, 1, t, device=DEVICE)


# ---------------------------------------------------------------------------
# 1. Silent input test
# ---------------------------------------------------------------------------

class TestSilentInput:
    """All-zero mono audio must not crash or produce NaN/Inf."""

    def test_forward_no_nan(self):
        model = make_model()
        mono = torch.zeros(B, 1, CHUNK, device=DEVICE)
        view = random_view()

        binaural, outputs = model(mono, view)

        assert not torch.isnan(binaural).any(), "NaN in output for silent input"
        assert not torch.isinf(binaural).any(), "Inf in output for silent input"

    def test_training_step_no_nan(self):
        model = make_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        mono = torch.zeros(B, 1, CHUNK, device=DEVICE)
        view = random_view()
        target = torch.zeros(B, 2, CHUNK, device=DEVICE)

        pred, _ = model(mono, view)
        loss = torch.nn.functional.mse_loss(pred, target)

        assert not torch.isnan(loss), "NaN loss on silent input"

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Parameters must remain finite after the update
        for name, p in model.named_parameters():
            assert not torch.isnan(p).any(), f"NaN in parameter {name} after silent-input step"


# ---------------------------------------------------------------------------
# 2. Extreme position test  (transmitter at receiver origin)
# ---------------------------------------------------------------------------

class TestExtremePosition:
    """view with all-zero position/orientation must not crash."""

    def test_zero_view_forward(self):
        model = make_model()
        mono = random_mono()
        view = torch.zeros(B, 7, K, device=DEVICE)

        binaural, outputs = model(mono, view)

        assert binaural.shape == (B, 2, CHUNK)
        assert not torch.isnan(binaural).any(), "NaN with zero view"

    def test_zero_view_binaural_network(self):
        """BinauralNetwork uses scipy quaternion rotation — zero quat must be handled."""
        model = make_binaural_model()
        mono = random_mono()
        view = torch.zeros(B, 7, K, device=DEVICE)

        result = model(mono, view)
        out = result["output"]

        assert not torch.isnan(out).any(), "NaN in BinauralNetwork output with zero view"


# ---------------------------------------------------------------------------
# 3. Gradient flow test
# ---------------------------------------------------------------------------

class TestGradientFlow:
    """Every parameter should receive a gradient after one backward pass.

    Known zero-grad candidates:
    - Parameters gated by a dead ReLU path (e.g. if view_feat is all-zero)
    - The `freqs` buffer (registered_buffer, not a parameter — no grad expected)
    - WaveoutBlock.second if the sin activation saturates
    """

    def test_all_params_have_grad(self):
        model = make_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        mono = random_mono()
        view = random_view()
        target = torch.randn(B, 2, CHUNK, device=DEVICE)

        pred, _ = model(mono, view)
        loss = torch.nn.functional.mse_loss(pred, target)
        optimizer.zero_grad()
        loss.backward()

        zero_grad_params = [
            name for name, p in model.named_parameters()
            if p.requires_grad and (p.grad is None or p.grad.abs().max() == 0)
        ]

        if zero_grad_params:
            pytest.fail(
                f"Parameters with zero/None gradient:\n" +
                "\n".join(f"  {n}" for n in zero_grad_params)
            )

    def test_binaural_network_grad_flow(self):
        model = make_binaural_model()
        mono = random_mono()
        view = random_view()
        target = torch.randn(B, 2, CHUNK, device=DEVICE)

        result = model(mono, view)
        loss = torch.nn.functional.mse_loss(result["output"], target)
        loss.backward()

        zero_grad_params = [
            name for name, p in model.named_parameters()
            if p.requires_grad and (p.grad is None or p.grad.abs().max() == 0)
        ]

        # Report but don't hard-fail — document which params are dead
        if zero_grad_params:
            print(f"\n[WARN] BinauralNetwork zero-grad params: {zero_grad_params}")


# ---------------------------------------------------------------------------
# 4. Memory leak test
# ---------------------------------------------------------------------------

class TestMemoryLeak:
    """GPU memory must not grow unboundedly over 100 training steps."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_no_memory_growth(self):
        model = make_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        def one_step():
            mono = random_mono()
            view = random_view()
            target = torch.randn(B, 2, CHUNK, device=DEVICE)
            pred, _ = model(mono, view)
            loss = torch.nn.functional.mse_loss(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Warm-up: let CUDA allocator stabilise
        for _ in range(5):
            one_step()
        torch.cuda.synchronize()

        mem_before = torch.cuda.memory_allocated(DEVICE)

        for _ in range(100):
            one_step()
        torch.cuda.synchronize()

        mem_after = torch.cuda.memory_allocated(DEVICE)

        # Allow up to 10 MB of growth (allocator fragmentation headroom)
        growth_mb = (mem_after - mem_before) / 1024 ** 2
        assert growth_mb < 10, (
            f"GPU memory grew by {growth_mb:.1f} MB over 100 steps — possible leak"
        )


# ---------------------------------------------------------------------------
# 5. Checkpoint save / load test
# ---------------------------------------------------------------------------

class TestCheckpointSaveLoad:
    """Stage 1 checkpoint must be fully restored for Stage 2."""

    def test_state_dict_roundtrip(self):
        model = make_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        # Simulate one training step so weights are non-default
        mono, view = random_mono(), random_view()
        pred, _ = model(mono, view)
        loss = torch.nn.functional.mse_loss(pred, torch.randn_like(pred))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "stage1.pth")
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "stage": 1,
            }, ckpt_path)

            # Load into a fresh model (Stage 2 init)
            model2 = make_model()
            optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-4)
            ckpt = torch.load(ckpt_path, map_location=DEVICE)
            model2.load_state_dict(ckpt["model_state_dict"])
            optimizer2.load_state_dict(ckpt["optimizer_state_dict"])

            # All parameter tensors must match exactly
            for (n1, p1), (n2, p2) in zip(
                model.named_parameters(), model2.named_parameters()
            ):
                assert torch.equal(p1, p2), f"Parameter mismatch after reload: {n1}"

            # Optimizer state must be preserved (check first param group lr)
            assert (
                optimizer2.state_dict()["param_groups"][0]["lr"]
                == optimizer.state_dict()["param_groups"][0]["lr"]
            ), "Optimizer lr not preserved"

    def test_inference_identical_after_reload(self):
        """Reloaded model must produce bit-identical output."""
        model = make_model().eval()

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "model.pth")
            torch.save({"model_state_dict": model.state_dict()}, ckpt_path)

            model2 = make_model().eval()
            model2.load_state_dict(
                torch.load(ckpt_path, map_location=DEVICE)["model_state_dict"]
            )

        mono, view = random_mono(batch=1), random_view(batch=1)
        with torch.no_grad():
            out1, _ = model(mono, view)
            out2, _ = model2(mono, view)

        assert torch.equal(out1, out2), "Outputs differ after checkpoint reload"


# ---------------------------------------------------------------------------
# 6. Multi-GPU test
# ---------------------------------------------------------------------------

class TestMultiGPU:
    """Model must work under DataParallel when multiple GPUs are available."""

    @pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.device_count() < 2,
        reason="Requires ≥2 GPUs",
    )
    def test_data_parallel_forward(self):
        model = make_model()
        dp_model = torch.nn.DataParallel(model)

        mono = random_mono(batch=4)
        view = random_view(batch=4)

        with torch.no_grad():
            binaural, _ = dp_model(mono, view)

        assert binaural.shape == (4, 2, CHUNK)
        assert not torch.isnan(binaural).any()

    @pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.device_count() < 2,
        reason="Requires ≥2 GPUs",
    )
    def test_data_parallel_backward(self):
        model = make_model()
        dp_model = torch.nn.DataParallel(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        mono = random_mono(batch=4)
        view = random_view(batch=4)
        target = torch.randn(4, 2, CHUNK, device=DEVICE)

        binaural, _ = dp_model(mono, view)
        loss = torch.nn.functional.mse_loss(binaural, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for name, p in model.named_parameters():
            assert p.grad is not None, f"No gradient for {name} under DataParallel"

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA required for single-GPU DataParallel smoke test",
    )
    def test_single_gpu_data_parallel_smoke(self):
        """Single-GPU DataParallel (as used by Trainer) must not crash."""
        model = make_model()
        dp_model = torch.nn.DataParallel(model, device_ids=[0])

        mono = random_mono(batch=B)
        view = random_view(batch=B)

        with torch.no_grad():
            binaural, _ = dp_model(mono, view)

        assert not torch.isnan(binaural).any()
