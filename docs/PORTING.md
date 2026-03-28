# Porting notes

← Back to **[documentation index](README.md)** · Root **[`README.md`](../README.md)**

This codebase is developed and tuned on **AMD ROCm (HIP)**. The **algorithmic ideas** (BF3W fusion, plane scanning, software prefetch) are not vendor-specific; **low-level tuning** (warp width, LDS layout, instruction-level choices) must be revisited on other GPUs.

## NVIDIA CUDA

- Map **HIP device kernels** to CUDA compilation (`nvcc` or clang CUDA mode).
- Replace **warp shuffle** idioms with CUDA equivalents (`__shfl_*`); verify width (32 vs 64) against your architecture.
- **Shared memory** and **async copy** (`cp.async`, TMA on Hopper-class hardware) are natural targets to re-express the same buffering/prefetch intent as the HIP fused kernel.
- Expect **register pressure** and **occupancy** to change; the paper discusses controlling unrolling for this reason.

## Changing WENO order

The fused scanner assumes a stencil **radius** consistent with the chosen scheme:

- **5th-order** WENO typically uses a narrower star stencil (e.g. 3D19P); the same scanning pattern can apply with **smaller halos** and often **more reusable surface data** on-chip.
- **7th-order** (3D25P) matches the default windowing in the implementation (see `scanY` / `scanZ` and halo width in `src/kernels/OCFD_split.cpp` and headers such as `LAP` in `config_parameters.h`).
- **9th-order** needs a **larger radius**; increase sub-window sizes (e.g. \(G_y, G_z\)) so sub-windows still cover halos. **Sections 3.2–3.3** of the paper (scheduling + prefetch) must be rebalanced because **on-chip storage** becomes tighter.

After any order change, re-run **numerical regression** against a trusted baseline (global quantities, probes, or field diffs).

## Profiling

For publication-grade evidence of latency hiding, capture **vendor profilers** (e.g. ROCprofiler on AMD, Nsight Compute on NVIDIA) and archive summary metrics or anonymized traces under a `profiling/` directory if your policy allows sharing them.
