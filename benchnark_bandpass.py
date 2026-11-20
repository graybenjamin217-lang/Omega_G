"""
Clinical head-to-head benchmark: Omega_G_odd vs. standard threshold/bandpass gates.
Measures MSE, ternary classification accuracy, and wall-clock runtime honestly.
"""

from __future__ import annotations

import time
import numpy as np
import omega_g as og


def eval_omega_odd(x: np.ndarray, iters: int) -> np.ndarray:
    y = x.copy()
    for _ in range(iters):
        y = og.f_odd_scalar(y)
    return y


def run_benchmark(n: int = 200_000, seed: int = 2026) -> None:
    rng = np.random.default_rng(seed)
    truth = rng.choice([-1.0, 0.0, 1.0], size=n, p=[0.25, 0.50, 0.25])

    scenarios = {
        "Gaussian noise only (sigma=0.12)": truth + rng.normal(0.0, 0.12, size=n),
        "Gaussian + 8% impulse spikes (amp 5-15)": None,
    }
    spiked = truth + rng.normal(0.0, 0.12, size=n)
    spike_mask = rng.random(n) < 0.08
    spiked[spike_mask] += rng.choice([-1.0, 1.0], size=int(np.sum(spike_mask))) * rng.uniform(
        5.0, 15.0, size=int(np.sum(spike_mask))
    )
    scenarios["Gaussian + 8% impulse spikes (amp 5-15)"] = spiked

    methods = [
        ("Omega_G_odd (1 pass, C^inf)", lambda x: eval_omega_odd(x, 1)),
        ("Omega_G_odd (6 passes, C^inf)", lambda x: eval_omega_odd(x, 6)),
        ("Hard threshold (|x|>0.5)", lambda x: np.where(np.abs(x) > 0.5, np.sign(x), 0.0)),
        ("Hard bandpass (0.5<|x|<x_max)", lambda x: np.where((np.abs(x) > 0.5) & (np.abs(x) < og.X_MAX), np.sign(x), 0.0)),
        ("Soft saturation tanh(2.5x)", lambda x: np.tanh(2.5 * x)),
    ]

    for title, sig in scenarios.items():
        print(f"\n=== {title} (N={n:,}) ===")
        print(f"{'Method':<32} | {'MSE':>10} | {'Acc (%)':>8} | {'Time (ms)':>10}")
        print("-" * 70)
        for name, fn in methods:
            t0 = time.perf_counter()
            out = fn(sig)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            mse = float(np.mean((out - truth) ** 2))
            acc = 100.0 * float(np.mean(np.round(out) == truth))
            print(f"{name:<32} | {mse:10.6f} | {acc:8.3f} | {dt_ms:10.2f}")


if __name__ == "__main__":
    run_benchmark()
