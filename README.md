# FILO2: CUDA-Accelerated Large-Scale CVRP Solver

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![CUDA 12/13](https://img.shields.io/badge/CUDA-12.x%20%7C%2013.x-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![CMake](https://img.shields.io/badge/CMake-%3E%3D3.22-brightgreen.svg)](https://cmake.org/)

A high-performance C++ / CUDA solver for the **Capacitated Vehicle Routing Problem (CVRP)**, extending the state-of-the-art **FILO2** algorithm (*Accorsi & Vigo, 2024*) with **GPU-accelerated Uniform Grid $k$-Nearest Neighbors ($k$-NN)** preprocessing.

Capable of optimizing instances up to **1,000,000 customers** in a few minutes on standard computing hardware.

---

## Attribution & Original Work

This project builds directly upon the published academic solver by Luca Accorsi and Daniele Vigo:

> **Accorsi, L., & Vigo, D. (2024).** *Routing one million customers in a handful of minutes.*  
> **Computers & Operations Research**, Volume 164, 106562.  
> [DOI: 10.1016/j.cor.2024.106562](https://doi.org/10.1016/j.cor.2024.106562) | [arXiv:2306.14205](https://arxiv.org/abs/2306.14205) | [Original Repository: github.com/acco93/filo2](https://github.com/acco93/filo2)

All core optimization heuristics (Restricted Clarke-Wright, BPP route estimation, ILS route minimization, Simulated Annealing Core Optimization, Ruin & Recreate, Selective Vertex Caching, and simplified HRVND) follow the original paper under the GNU General Public License v3.0.

---

## What's New in This Repository

In the original paper, the authors noted that single-threaded CPU $kd$-tree preprocessing accounted for **~48% of total solver runtime** on large instances. This extension introduces:

1. **CUDA Uniform Grid $k$-NN Engine with Binary Max-Heap (`cuda/`)**:
   - Replaces the sequential CPU $kd$-tree with a GPU-accelerated spatial uniform grid.
   - Maintains an in-register/scratch **binary max-heap** per query thread: $O(1)$ candidate rejection and $O(\log k)$ candidate replacement (`sift_down_candidates`).
   - Concentric Chebyshev ring expansion with conservative device pruning bounds (`outside_lower_bound`).
2. **Slot-Major Coalesced Memory Layout & Shared-Memory Transpose**:
   - Stores device scratch in **slot-major** order (`[k][batch]`), allowing threads in a warp to access contiguous memory words with **100% coalesced memory transactions**.
   - Employs a $32 \times 32$ shared-memory 2D tiled transpose kernel (`transpose_knn_output_kernel`) to emit row-major IDs for a single coalesced PCIe copy to the host.
3. **Batched GPU Memory Architecture (`cuda/CudaMemoryPlan.hpp`)**:
   - Uses query batching (default 4,096 queries per launch) with strict 64-bit integer overflow protection.
   - Restricts peak device scratch VRAM to **< 100 MB**, allowing instances with **1,000,000 customers** to run effortlessly on 4 GB–8 GB consumer/laptop GPUs.
4. **Deterministic Ground-Truth Test Suite (`knn/`, `tests/`)**:
   - Validates GPU neighbor outputs against an exact $O(N^2)$ brute-force reference with strict tie-breaking (`query_id` self first, then ascending vertex ID).
5. **Cross-Platform Build Support**:
   - First-class support for both **Linux (GCC/Clang)** and **Windows (MSVC 2022 / Ninja)**.
6. **Automatic Fallback (`instance/Instance.cpp`)**:
   - If compiled without CUDA or executed on a system without an NVIDIA GPU, the solver automatically falls back to the original CPU $kd$-tree.

---

## Verification Status

- **CPU Grid $k$-NN (`knn/GridKnn.cpp`)**: Matches the exact $O(N^2)$ brute-force reference in 100% of tested cases, confirmed via `cpu_grid_knn_test`.
- **CUDA $k$-NN (`cuda/CudaGridKnn.cu`)**: Matches the brute-force reference exactly across all 10 unit test cases (including duplicate coordinates, $N \le k$, large $k=1500$, forced small-batch splits, cross-ring tie-breaking, and execution determinism checks), confirmed via `cuda_grid_knn_test`.
- **Direct Side-by-Side Equivalence**: CUDA and the original `base/KDTree.cpp` select identical nearest neighbors in every case tested, confirmed via direct side-by-side comparison (`kdtree_vs_gridknn_test`).
- **Full End-to-End Solver Correctness**: The complete FILO2 solver runs reliably and produces valid, capacity-constrained CVRP solutions—with CUDA enabled or disabled—on benchmark instances ranging from 101 to 1,000,000 customers.
- **Consistent Real-World Speedup**: CUDA preprocessing is faster than the CPU $kd$-tree across $k = 50$ through $k = 1500$ (the paper's full tested range), confirmed on an NVIDIA GeForce RTX 4070 Laptop GPU with roughly **9× speedup** at $k = 1500$ on the 1,000,000-customer instance.

---

## Known Behavioral Difference: Tie-Breaking

CUDA and the original CPU $kd$-tree always select the same set of nearest neighbors. When two candidate points are at the exact same distance:
- `GridKnn` / CUDA breaks the tie deterministically by choosing the **smaller vertex ID**.
- The original `base/KDTree.cpp` breaks it by **tree traversal order** (whichever branch was recursed first).

This means CUDA-enabled and CUDA-disabled runs may order tied neighbors differently and can converge on slightly different (not worse) final solutions on instances with many equidistant points (e.g. regular/dense grid layouts). This is expected, intentional behavior, not a bug — `base/KDTree.cpp` was deliberately left unmodified to preserve the upstream implementation.

---

## Empirical Benchmark Results

Measured on an **NVIDIA GeForce RTX 4070 Laptop GPU (CUDA 13.3)** and **Intel Core i7** CPU running Windows 11 on the 1,000,000-customer instance (**`Lazio.vrp`**, 5,000 iterations):

### 1. Scaling Across Neighbor Counts ($k$) on 1,000,000 Customers

| $k$ Value (`--neighbors-num`) | CPU Preprocessing Time | CUDA Preprocessing Time | Preprocessing Speedup Factor | CPU Total Time | CUDA Total Time | Time Saved | Best Solution Cost (Routes) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$k = 50$** | 14 s | **2 s** | **7.00×** | 33 s | 35 s | 12 s saved (preproc) | 3,228,495,464 (40,771) |
| **$k = 100$** | 23 s | **3 s** | **7.67×** | 44 s | **37 s** | 20 s saved (preproc) | 3,192,659,758 (40,434) |
| **$k = 250$** | 45 s | **5 s** | **9.00×** | 70 s | **43 s** | 40 s saved (preproc) | 3,179,521,210 (40,298) |
| **$k = 500$** | 87 s | **10 s** | **8.70×** | 114 s | **49 s** | 77 s saved (preproc) | **3,174,678,736** (40,254) |
| **$k = 1500$** | 253 s *(4.2 min)* | **28 s** | **9.04×** | 282 s *(4.7 min)* | **71 s** | **211 s saved (3.97× overall)** | **3,165,547,931** (**40,162**) |

---

### 2. Scaling Across Instance Sizes ($k = 1500$, 5,000 Iterations)

| Instance | Customers ($N$) | Total Arcs ($N^2$) | CPU Preproc | CUDA Preproc | Preproc Speedup | CPU Total | CUDA Total | Final Obj (CUDA) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`Valle-D-Aosta.vrp`** | 20,000 (20k) | $4 \times 10^8$ (400M) | 3 s | 5 s | 0.60× | 8 s | 13 s | **21,817,287** (801 routes) |
| **`Trentino-Alto-Adige.vrp`** | 100,000 (100k) | $10^{10}$ (10B) | 28 s | 28 s | 1.00× | 42 s | 53 s | **103,855,930** (1,349 routes) |
| **`Friuli-Venezia-Giulia.vrp`** | 300,000 (300k) | $9 \times 10^{10}$ (90B) | 68 s | 52 s | **1.31×** *(16s saved)* | 85 s | **71 s** | **420,947,382** (3,031 routes) |
| **`Lazio.vrp`** | 1,000,000 (1M) | $10^{12}$ (1 Trillion) | 253 s | **28 s** | **9.04×** *(225s saved)* | 282 s | **71 s** | **3,165,547,931** (40,162 routes) |

---

## Directory Structure

```
.
├── base/             # Foundational utilities (KDTree, BinaryHeap, SparseIntSet, LRUCache, Timer)
├── cuda/             # CUDA Uniform Grid k-NN implementation & memory planner
├── instance/         # VRP parser, coordinate handling, and problem instance definitions
├── knn/              # CPU reference grid k-NN and ground-truth brute force comparator
├── localsearch/      # 22 local search operators (2-opt, CROSS-exchange, Tails, Split, Ejection Chain)
├── movegen/          # Move generator structures and granular neighborhood filtering
├── opt/              # Metaheuristics (Simulated Annealing, Ruin & Recreate, ROUTEMIN, BPP)
├── solution/         # Solution representation, savings heuristic, Do/Undo action stacks
├── tests/            # CTest unit test suite for CPU and CUDA k-NN implementations
├── CMakeLists.txt    # Top-level CMake configuration (CPU & CUDA targets)
├── LICENSE           # GNU General Public License v3.0
├── main.cpp          # Solver entry point and CLI option handling
├── Parameters.hpp    # Parameter configuration defaults
├── Renderer.hpp      # Optional GUI visualizer header
└── README.md         # Project documentation
```

---

## Building and Running

### Prerequisites
- **CMake** $\ge$ 3.22
- **C++17** compatible compiler (GCC 9+, Clang 10+, or MSVC 2022)
- *(Optional for CUDA)* **NVIDIA CUDA Toolkit** $\ge$ 12.0

---

### 1. CPU-Only Build

```bash
# Configure
cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=Release -DFILO2_USE_CUDA=OFF -DBUILD_TESTING=ON

# Build
cmake --build build-cpu --config Release --parallel

# Run Unit Tests
ctest --test-dir build-cpu --output-on-failure

# Run Solver
./build-cpu/filo2 path/to/instance.vrp --coreopt-iterations 5000
```

---

### 2. CUDA-Accelerated Build

```bash
# Configure (adjust CMAKE_CUDA_ARCHITECTURES for your GPU, e.g., 89 for RTX 40-series, 86 for RTX 30-series, 80 for A100)
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release -DFILO2_USE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DBUILD_TESTING=ON

# Build
cmake --build build-cuda --config Release --parallel

# Run Unit Tests (Validates GPU k-NN kernels on your hardware)
ctest --test-dir build-cuda -C Release --output-on-failure

# Run CUDA Solver (e.g., k=1500 nearest neighbors)
./build-cuda/Release/filo2 path/to/instance.vrp --neighbors-num 1500 --coreopt-iterations 5000
```

---

## Obtaining Benchmark Instances

Benchmark `.vrp` instances are not bundled in this repository to keep the codebase ultra-lightweight. You can obtain instances from the following sources:

- **Official Upstream Repository**: Download the $X$, $B$, and massive $I$ (Italy, up to 1,000,000 customers) datasets from **[github.com/acco93/filo2](https://github.com/acco93/filo2)**.
- **CVRPLIB**: Standard benchmark instances from Uchoa et al. (2017) are available at **[http://vrp.atd-lab.inf.puc-rio.br/index.php/en/](http://vrp.atd-lab.inf.puc-rio.br/index.php/en/)**.

---

## License

This project is licensed under the **GNU General Public License v3.0** (GPLv3) - see the [LICENSE](LICENSE) file for details.
