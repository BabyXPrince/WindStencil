# WindStencil code layout

Documentation index: **[`README.md`](README.md)** (this folder).

Sources are grouped by role so readers can connect modules to the paper (**BF3W**, **plane scanning**, **prefetching**) while navigating a **full compressible NS** stack (not a kernel-only drop-in). The implementation uses **HIP/ROCm**; surrounding solver modules follow **OpenCFD-SC** conventions. All `#include "xxx.h"` lines stay short; the compiler sees them via **`-I include/windstencil`**.

<p align="center">
  <img src="images/repo-layout.svg" alt="High-level repository directories" width="640"/>
</p>

## Directory map

| Path | Role | Paper (conceptual) |
|------|------|---------------------|
| `include/windstencil/` | All headers | Types, device SoA, MPI / parameters |
| `src/kernels/` | Device kernels and HIP common code | **§3.1** fused path (`OCFD_split.cpp`), **§3.3** prefetch / shared helpers (`commen_kernel.cpp`), WENO stencils (`OCFD_Schemes.cpp`), `cuda_commen.c` / `cuda_utility.c` |
| `src/solver/` | Time marching, NS / Jacobian, flux & scheme drivers | Solver orchestration, Jacobian, `OCFD_Stream`, etc. |
| `src/boundary/` | Boundary conditions and boundary schemes | Boundaries and special geometries |
| `src/io/` | I/O and MPI-I/O helpers | Reads / writes, `io_warp` |
| `src/mpi/` | Host- and device-side MPI wrappers | Halo exchange, device sync |
| `src/runtime/` | Parameters, utils, init, filtering, analysis, test hooks | Runtime support |
| `src/app/` | `opencfd.c` (CUDA source) / `opencfd_hip.c` (HIP, used by build) | `main`, init, `NS_solver_real()` |

## Concept → code (quick)

| Idea in the paper | Where to read the code |
|-------------------|-------------------------|
| Block-fused 3D WENO (BF3W) | `src/kernels/OCFD_split.cpp` — single fused inviscid path vs. decomposed kernels |
| yz-plane scanning (`scanY` / `scanZ`) | Same file — macros and index arithmetic for sub-windows |
| Software prefetch / buffering | Same file — comments and double-buffer style patterns in the main loop |
| WENO stencil arithmetic | `src/kernels/OCFD_Schemes.cpp`, headers under `include/windstencil/` |
| Time step & coupling | `src/solver/OCFD_NS_Solver.cpp` |

## Key files

- **Fused WindStencil path:** `src/kernels/OCFD_split.cpp` — plane-scan macros `scanY` / `scanZ`, on-chip buffers, prefetch comments.
- **High-order stencils:** `src/kernels/OCFD_Schemes.cpp`; declarations in `include/windstencil/OCFD_Schemes.h`.
- **Time-step driver:** `src/solver/OCFD_NS_Solver.cpp`.
- **Entry:** `src/app/opencfd_hip.c`, regenerated from `src/app/opencfd.c` with **`make hipify-main`** (`hipify-perl`).

## Figures

- Directory sketch (above): [`images/repo-layout.svg`](images/repo-layout.svg)
- How to add your own diagrams or paper exports: [`images/README.md`](images/README.md)

## Build

On a machine with **ROCm (`hipcc`)** and **MPI**:

```bash
make -j
# produces opencfd-windstencil (or TARGET=...)
```

Override **`ROCM_PATH`**, **`MPI_PATH`**, and **`GPU_ARCH`** as needed. Details match the root **[`README.md`](../README.md)**.

If you still maintain a CUDA upstream, run `hipify-perl` on those sources and place headers under `include/windstencil` and sources under the `src/` subtrees. Keep `src/app/opencfd.c` as the host entry template and run **`make hipify-main`** to refresh `src/app/opencfd_hip.c`.

## Related

- **[`PORTING.md`](PORTING.md)** — NVIDIA, other WENO orders, profiling artifacts.
