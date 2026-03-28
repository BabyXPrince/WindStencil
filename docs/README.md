# WindStencil documentation

| Document | Description |
|----------|-------------|
| [**ARCHITECTURE.md**](ARCHITECTURE.md) | Directory map, key source files, build reminder, links to paper §3 concepts. |
| [**PORTING.md**](PORTING.md) | Other GPU vendors, changing WENO order, profiling artifacts. |
| [**images/**](images/) | SVG overviews for README / docs; place exported paper figures here. |

## Quick links

- Main project readme: [../README.md](../README.md)
- Fused kernel entry point: `../src/kernels/OCFD_split.cpp`
- Build: `make -j` from repository root (see readme **Build** section)

## Badges (main README)

Shields for OS, ROCm/HIP, MPI, and venue live on the **repository homepage** [`README.md`](../README.md); this folder stays text-first unless you add figures under `images/`.
