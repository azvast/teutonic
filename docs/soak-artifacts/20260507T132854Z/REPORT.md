# Teutonic-LXXX Soak-10 Report

Run timestamp (UTC): 20260507T132854Z
Iterations: 10
Total wall: 28.3 min
eval_server PID at start: 457759
eval_server PIDs across soak: [457759] (STABLE)

## Per-iteration summary

| iter | seed | perturb | eval | total | mu_hat | lcb | accepted | peak VRAM (MiB) | cache GB pre/post |
|---:|---:|---:|---:|---:|---:|---:|:-:|---:|:-:|
| 0 | 42 | 0s | 170s | 170s | 0.0 | 0.0 | ✗ | 89308 | 305/305 |
| 1 | 43 | 0s | 170s | 170s | 0.0 | 0.0 | ✗ | 89314 | 305/305 |
| 2 | 44 | 0s | 173s | 173s | 0.0 | 0.0 | ✗ | 89320 | 305/305 |
| 3 | 45 | 0s | 169s | 169s | 0.0 | 0.0 | ✗ | 89328 | 305/305 |
| 4 | 46 | 0s | 168s | 168s | 0.0 | 0.0 | ✗ | 89336 | 305/305 |
| 5 | 47 | 0s | 170s | 170s | 0.0 | 0.0 | ✗ | 89344 | 305/305 |
| 6 | 48 | 0s | 170s | 170s | 0.0 | 0.0 | ✗ | 89350 | 305/305 |
| 7 | 49 | 0s | 168s | 168s | 0.0 | 0.0 | ✗ | 89358 | 305/305 |
| 8 | 50 | 0s | 168s | 168s | 0.0 | 0.0 | ✗ | 89364 | 305/305 |
| 9 | 51 | 0s | 170s | 171s | 0.0 | 0.0 | ✗ | 89372 | 305/305 |

## Leak detection

- Peak VRAM range: 89308 .. 89372 MiB (spread 64 MiB)
- Mean peak first-5 vs last-5: drift +36 MiB (OK)