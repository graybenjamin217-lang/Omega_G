# $\hat{\Omega}_G$ — Variable-Exponent Feedback Operator Suite

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24445638.svg)](https://doi.org/10.5281/zenodo.24445638)

Reference implementation, numerical theorem verification suite, and outlier-rejection benchmark accompanying:

> **The Variable-Exponent Feedback Operator $\hat{\Omega}_G$ and Its Odd Extension on Unital $C^*$-Algebras**  
> Benjamin Edward Gray (2026)  
> Version 1 DOI: [10.5281/zenodo.24445638](https://doi.org/10.5281/zenodo.24445638) · Concept DOI: [10.5281/zenodo.24445637](https://doi.org/10.5281/zenodo.24445637)

## Repository Structure

```text
Omega_G/
├── omega_g.py              # Core scalar, Hermitian C*-algebra, and non-normal implementations
├── test_omega_g.py         # Strict pytest suite verifying all theorems + all 5 edge cases
├── benchmark_bandpass.py   # Honest head-to-head bandpass outlier-rejection benchmark
├── requirements.txt        # Python dependencies
├── CITATION.cff            # Machine-readable citation metadata
└── LICENSE                 # Software license
```

============================= test session starts ==============================
platform linux -- Python 3.14.2, pytest-9.2.0, pluggy-1.7.0
rootdir: /workspaces/Omega_G
plugins: platformdirs-4.14.0, anyio-4.15.1
collected 26 items

test_omega_g.py ..........................                               [100%]

============================== 26 passed in 0.21s ==============================
```

## Benchmark Results

### Gaussian noise only (sigma=0.12), N=200,000

| Method | MSE | MAE | Accuracy | Time |
|---|---:|---:|---:|---:|
| Omega_G_odd (1 pass, C^inf) | 0.000566 | 0.008947 | 99.997% | 18.91 ms |
| Omega_G_odd (6 passes, C^inf) | 0.000025 | 0.000026 | 99.997% | 84.50 ms |
| Hard threshold (|x|>0.5) | 0.000025 | 0.000025 | 99.997% | 2.49 ms |
| Hard bandpass (0.5<|x|<x_max) | 0.000025 | 0.000025 | 99.997% | 2.29 ms |
| Soft saturation tanh(2.5x) | 0.038743 | 0.121526 | 96.637% | 4.16 ms |

### Gaussian noise plus 8% impulse spikes (amplitude 5-15), N=200,000

| Method | MSE | MAE | Accuracy | Time |
|---|---:|---:|---:|---:|
| Omega_G_odd (1 pass, C^inf) | 0.043764 | 0.054466 | 96.032% | 16.44 ms |
| Omega_G_odd (6 passes, C^inf) | 0.039675 | 0.039675 | 96.032% | 78.70 ms |
| Hard threshold (|x|>0.5) | 0.119740 | 0.079980 | 93.990% | 2.42 ms |
| Hard bandpass (0.5<|x|<x_max) | 0.039675 | 0.039675 | 96.032% | 2.53 ms |
| Soft saturation tanh(2.5x) | 0.155383 | 0.191804 | 90.877% | 4.12 ms |
