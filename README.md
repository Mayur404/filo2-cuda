# FILO2 with parallel CPU and CUDA preprocessing

This repository extends [FILO2](https://github.com/acco93/filo2), a capacitated vehicle routing problem (CVRP) solver, with two ways to accelerate nearest-neighbor preprocessing:

- **OpenMP CPU build:** runs independent queries against the original `KDTree` on multiple CPU threads. This is the option to use when the solution must match the original FILO2 run.
- **CUDA build:** uses a GPU uniform-grid nearest-neighbor implementation, plus CPU work in other preprocessing stages. It is faster in the measurements below, but can produce a different valid solution when distance ties change neighbor ordering.

The route minimization and CoreOpt search rules are retained. The solver is a heuristic, so a different neighbor order can change the route merges and the final objective even with the same seed. Exact output equality is an observed result for the tested CPU runs below, not a claim for every compiler, platform, or instance.

## Measured results

The following runs were made on one Windows 11 machine with an AMD Ryzen 7 7735HS (8 cores, 16 logical threads) and an NVIDIA GeForce RTX 4070 Laptop GPU. All three executables used the same local instance files, `--neighbors-num 1500`, `--coreopt-iterations 5000`, and `--seed 0`. The OpenMP build used `OMP_NUM_THREADS=16`. Times are the solver's printed whole-second measurements from one run per build and instance; they are not averages or external wall-clock timings.

The original executable came from the adjacent `filo2` folder. Both accelerated executables came from this repository. The instance dimensions below are the `DIMENSION` fields in the `.vrp` files, including the depot.

| Instance | Dimension | Build | Preprocessing | Total | Total speedup vs original | Best objective | Routes |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Abruzzo | 250,000 | Original CPU | 57 s | 76 s | 1.00× | 315,504,793 | 2,525 |
| Abruzzo | 250,000 | OpenMP CPU | 5 s | 22 s | 3.45× | 315,504,793 | 2,525 |
| Abruzzo | 250,000 | CUDA | 3 s | 19 s | 4.00× | 314,800,751 | 2,519 |
| Campania | 500,000 | Original CPU | 124 s | 148 s | 1.00× | 398,952,768 | 5,064 |
| Campania | 500,000 | OpenMP CPU | 13 s | 35 s | 4.23× | 398,952,768 | 5,064 |
| Campania | 500,000 | CUDA | 7 s | 30 s | 4.93× | 399,196,947 | 5,062 |
| Lazio | 1,000,000 | Original CPU | 258 s | 300 s | 1.00× | 3,164,137,956 | 40,154 |
| Lazio | 1,000,000 | OpenMP CPU | 27 s | 52 s | 5.77× | 3,164,137,956 | 40,154 |
| Lazio | 1,000,000 | CUDA | 16 s | 41 s | 7.32× | 3,165,547,931 | 40,162 |

For Lazio, the printed stage timings show where the 300-to-52-second CPU
reduction comes from:

| Stage | Original CPU | OpenMP CPU | CUDA |
| --- | ---: | ---: | ---: |
| Instance preprocessing (including neighbors) | 258 s | 27 s | 16 s |
| Clarke–Wright initial solution | 11 s | 8 s | 8 s |
| Move-generator setup | 7 s | 3 s | 3 s |
| Greedy route bound | 10,745 ms | 55 ms | 55 ms |
| RouteMin | 4 s | 4 s | 4 s |
| Entire run | 300 s | 52 s | 41 s |

These are whole-second stage prints (except the bound); they need not add up
exactly to the total. Most of the CPU reduction is in independent neighbor
queries. The route-bound reduction comes from the existing segment-tree
implementation, and some other preprocessing also changed. **The total
speedup is for this repository as a whole, not an isolated OpenMP-only
experiment.** CoreOpt search itself was not parallelized.

The original and OpenMP `.vrp.sol` files are **byte-for-byte identical** for Abruzzo, Campania, and Lazio. Their SHA-256 hashes match for each instance. Matching objective values alone would be weaker evidence because distinct routes can have the same cost. CUDA's solution file differs for all three: its objective is lower on Abruzzo and higher on Campania and Lazio. All objectives above are those printed by the corresponding solver run; this table does not claim that one build always finds a better solution.

| Instance | SHA-256 of original and OpenMP solution file |
| --- | --- |
| Abruzzo | `A0AEAB82C245144CB62424C38C8308CE009C0A81E3EEDFD8BF3B873F426A2AA5` |
| Campania | `E62D7553FF505DE676E1F90A7A29BCC5B21D1E031D0A7D7FF47F2239FD8D42C9` |
| Lazio | `58774F75930C79DAB89A397DD95257CE36BE43B2AEFAADA79C8CA4950C9E2785` |

### Earlier Lazio scaling measurements

The previous committed README recorded the following measurements on the same hardware, before the current savings-construction and CUDA-copy changes. They were measured at the time of that commit and are preserved here as historical results. Do not combine these timings with the current table to calculate a single speedup. That README reported one solution objective per row; it did not report separate CPU and CUDA objectives.

| Neighbors (`k`) | Original preprocessing | CUDA preprocessing | Original total | CUDA total | Reported objective (routes) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | 14 s | 1 s | 45 s | 21 s | 3,228,495,464 (40,771) |
| 100 | 22 s | 1 s | 55 s | 24 s | 3,192,659,758 (40,434) |
| 250 | 47 s | 2 s | 83 s | 30 s | 3,179,521,210 (40,298) |
| 500 | 88 s | 4 s | 126 s | 32 s | 3,174,678,736 (40,254) |
| 1,500 | 264 s | 18 s | 311 s | 47 s | 3,165,547,931 (40,162) |

## What changed from original FILO2

The original solver uses one CPU `KDTree` query per vertex. In the CPU build here, OpenMP assigns those independent queries to threads; each query still calls the original `KDTree::GetNearestNeighbors`. This keeps the tree's ordering of equal-distance neighbors. `OMP_NUM_THREADS` controls the thread count. With OpenMP unavailable, the same loop runs sequentially.

Other preprocessing changes shared by the CPU and CUDA builds:

1. Clarke–Wright savings are counted per customer and written in parallel into disjoint ranges. The original `std::sort` comparison and serial sort remain, so equal-valued savings retain the same processing order observed in the parity runs. Route merging itself remains sequential.
2. Move-generator setup reserves storage in advance, caches each vertex's neighbor cutoff cost, and computes those independent cutoff costs in parallel. Arc insertion still follows the original order.
3. The greedy route-count bound uses the segment-tree implementation in `opt/bpp.hpp` instead of the original linear search. This change predates the current OpenMP work. The tested CPU results above match the original solution files despite this different implementation of the bound.

The CUDA build additionally replaces the CPU `KDTree` query loop with `cuda/CudaGridKnn.cu`. It searches a spatial grid on the GPU, processes queries in batches, copies neighbor IDs into pinned host memory when available, and overlaps a batch transfer with CPU scattering of the preceding batch. It falls back to a synchronous copy if a second pinned buffer cannot be allocated. OpenMP also distributes independent host-side output-row work. If CUDA is unavailable at runtime, the solver falls back to the CPU `KDTree` path.

The CUDA grid resolves exact distance ties by vertex ID. The original `KDTree` can order tied candidates by its traversal and heap behavior. Even if two neighbor lists contain the same vertices, a different order can change Clarke–Wright and later heuristic decisions. The side-by-side unit test covers its test cases; it is not a proof of identical neighbor lists on every instance. **Use the OpenMP CPU build when exact original output is required.**

## Build

Requirements: CMake 3.22 or newer, a C++17 compiler, and optional OpenMP support for CPU parallelism. The CUDA build additionally needs an NVIDIA GPU and a compatible CUDA toolkit. CUDA is opt-in: the CPU build does not require the toolkit.

From PowerShell in the `filo2-cuda` directory:

```powershell
# CPU build with original KDTree neighbor ordering.
cmake -S . -B build-cpu-parity -DFILO2_USE_CUDA=OFF -DBUILD_TESTING=ON
cmake --build build-cpu-parity --config Release --parallel 8

# CUDA build. Architecture 89 is for the benchmark machine's RTX 4070;
# set CMAKE_CUDA_ARCHITECTURES for your own GPU.
cmake -S . -B build-cuda -DFILO2_USE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DBUILD_TESTING=ON
cmake --build build-cuda --config Release --parallel 8
```

On Linux, use the same CMake commands and run `build-cpu-parity/filo2` or `build-cuda/filo2` rather than the Windows `Release\filo2.exe` paths. OpenMP is detected by CMake; if it is unavailable, the build remains usable but CPU preprocessing is sequential.

## Reproduce the comparison

Place the benchmark instances under `instances/I/`. The user-facing commands below use the same seed and parameters as the measured table. Run each solver separately so its performance is not affected by another solver process. Replace `Lazio` with `Abruzzo` or `Campania` to reproduce the other rows.

```powershell
$env:OMP_NUM_THREADS = "16"

..\filo2\build\Release\filo2.exe instances/I/Lazio.vrp --neighbors-num 1500 --coreopt-iterations 5000 --seed 0 --outpath benchmark-results/Lazio/original
.\build-cpu-parity\Release\filo2.exe instances/I/Lazio.vrp --neighbors-num 1500 --coreopt-iterations 5000 --seed 0 --outpath benchmark-results/Lazio/omp
.\build-cuda\Release\filo2.exe instances/I/Lazio.vrp --neighbors-num 1500 --coreopt-iterations 5000 --seed 0 --outpath benchmark-results/Lazio/cuda
```

The adjacent `..\filo2` executable is the original build used for this comparison; it is not built by this repository. Each `--outpath` contains a `*.out` file with objective and solver-reported total time, and a `*.vrp.sol` route file. For example, to check exact parity:

```powershell
$original = 'benchmark-results/Lazio/original/Lazio.vrp_seed-0.vrp.sol'
$parallel = 'benchmark-results/Lazio/omp/Lazio.vrp_seed-0.vrp.sol'
(Get-FileHash $original -Algorithm SHA256).Hash -eq (Get-FileHash $parallel -Algorithm SHA256).Hash
```

`Parameters.hpp` lists the solver's other options and defaults. Build and benchmark directories are local artifacts and are excluded from Git.

## Tests and scope of verification

```powershell
ctest --test-dir build-cpu-parity -C Release --output-on-failure
ctest --test-dir build-cuda -C Release --output-on-failure
```

The test suite covers the CPU grid reference, CUDA memory planning, `KDTree` versus grid-neighbor cases, and (with CUDA enabled) the GPU neighbor implementation. The current CUDA build passed all four configured tests. The three full solver comparisons above provide the end-to-end evidence for original/OpenMP output parity. Neither the unit tests nor three instance runs establish a universal parity guarantee for the CUDA build.

## Attribution and license

This project builds on Luca Accorsi and Daniele Vigo, “Routing one million customers in a handful of minutes,” *Computers & Operations Research* 164 (2024), 106562 ([DOI](https://doi.org/10.1016/j.cor.2024.106562); [original repository](https://github.com/acco93/filo2)). The code is distributed under the [GNU General Public License v3.0](LICENSE).
