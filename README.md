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
> **Computers & Operations Research**, Volume 164, 106541.  
> [DOI: 10.1016/j.cor.2023.106541](https://doi.org/10.1016/j.cor.2023.106541) | [arXiv:2306.14205](https://arxiv.org/abs/2306.14205) | [Original Repository: github.com/acco93/filo2](https://github.com/acco93/filo2)

All core optimization heuristics (Restricted Clarke-Wright, BPP route estimation, ILS route minimization, Simulated Annealing Core Optimization, Ruin & Recreate, Selective Vertex Caching, and simplified HRVND) follow the original paper under the GNU General Public License v3.0.

---

## What's New in This Repository

In the original paper, the authors noted that single-threaded CPU $kd$-tree preprocessing accounted for **~48% of total solver runtime** on large instances. This extension introduces:

1. **CUDA Uniform Grid $k$-NN Engine (`cuda/`)**:
   - Replaces the sequential CPU $kd$-tree with a GPU-accelerated spatial uniform grid.
   - Bins coordinates into spatial grid cells and processes all queries concurrently (one thread per query point).
   - Concentric Chebyshev ring expansion with conservative device pruning bounds (`outside_lower_bound`).
2. **Batched GPU Memory Architecture (`cuda/CudaMemoryPlan.hpp`)**:
   - Uses query batching (default 4,096 queries per launch) with strict 64-bit integer overflow protection.
   - Restricts peak device scratch VRAM to **< 100 MB**, allowing instances with **1,000,000 customers** to run effortlessly on 4 GB–8 GB consumer/laptop GPUs.
3. **Deterministic Ground-Truth Test Suite (`knn/`, `tests/`)**:
   - Validates GPU neighbor outputs against an exact $O(N^2)$ brute-force reference with strict tie-breaking (`query_id` self first, then ascending vertex ID).
4. **Cross-Platform Build Support**:
   - First-class support for both **Linux (GCC/Clang)** and **Windows (MSVC 2022 / Ninja)**.
5. **Automatic Fallback (`instance/Instance.cpp`)**:
   - If compiled without CUDA or executed on a system without an NVIDIA GPU, the solver automatically falls back to the original CPU $kd$-tree.

---

## Benchmark Results & Empirical Analysis

The solver was verified and benchmarked across instances of increasing size on an **NVIDIA GeForce RTX 4070 Laptop GPU (CUDA 13.3)** and **Intel Core i7** CPU running Windows 11.

### 1. Scaling Across Problem Sizes ($k = 1500$, 5,000 Iterations)

| Instance | Customers ($N$) | Total Arcs ($N^2$) | CPU Preproc | CUDA Preproc | Preproc Speedup | CPU Total | CUDA Total | Final Obj (CUDA) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`Valle-D-Aosta.vrp`** | 20,000 (20k) | $4 \times 10^8$ (400M) | 3 s | 5 s | 0.60× | 9 s | 13 s | **21,817,287** (801 routes) |
| **`Trentino-Alto-Adige.vrp`** | 100,000 (100k) | $10^{10}$ (10B) | 28 s | 28 s | 1.00× | 42 s | 53 s | **103,855,930** (1,349 routes) |
| **`Friuli-Venezia-Giulia.vrp`** | 300,000 (300k) | $9 \times 10^{10}$ (90B) | 68 s | 52 s | **1.31×** *(16s saved)* | 85 s | **71 s** | **420,947,382** (3,031 routes) |
| **`Lazio.vrp`** | 1,000,000 (1M) | $10^{12}$ (1 Trillion) | 256 s | 291 s | 0.88× | 285 s | 333 s | **3,165,547,931** (40,162 routes) |

---

### 2. Isolation Study: Effect of $k$ on 1,000,000-Customer Preprocessing (`Lazio.vrp`)

To isolate the relationship between neighbor list capacity ($k$) and GPU parallelism, `Lazio.vrp` was benchmarked at varying $k$:

| $k$ Value (`--neighbors-num`) | CPU Preprocessing Time | CUDA Preprocessing Time | Preprocessing Speedup Factor | CPU Total Time | CUDA Total Time |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **$k = 100$** | **26 s** | **5 s** | **5.20× faster on GPU** *(21s saved)* | 56 s | **39 s** |
| **$k = 250$** | **46 s** | **18 s** | **2.56× faster on GPU** *(28s saved)* | 70 s | **55 s** |
| **$k = 1500$** | **256 s** | **291 s** | **0.88× (CPU is faster)** | 285 s | 333 s |

### Technical Analysis & Known Limitation
> [!NOTE]
> GPU preprocessing delivers **up to 5.2× speedup at $k \le 250$**.
> At very large $k$ ($k = 1500$), CUDA preprocessing experiences a slowdown relative to CPU.
> 
> **Root Cause**: In `cuda/CudaGridKnn.cu`, the device function `insert_candidate()` maintains a sorted top-$k$ list via linear insertion in global memory:
> ```cpp
> ids[position] = ids[position - 1]; // Shifts up to k elements per accepted candidate
> ```
> At $k=1500$, every candidate accepted near the front of the list causes up to 1,500 memory shifts per thread across uncoalesced global memory. At $k \le 250$, this shifting work is minimal, allowing the GPU's massive thread parallelism to dominate.
> 
> *Future Work*: Replace global-memory linear insertion with a device-side binary heap, bitonic sort register network, or shared-memory staging.

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

# Run CUDA Solver (e.g., k=250 nearest neighbors)
./build-cuda/Release/filo2 path/to/instance.vrp --neighbors-num 250 --coreopt-iterations 5000
```

---

## Obtaining Benchmark Instances

Benchmark `.vrp` instances are not bundled in this repository to keep the codebase ultra-lightweight. You can obtain instances from the following sources:

- **Official Upstream Repository**: Download the $X$, $B$, and massive $I$ (Italy, up to 1,000,000 customers) datasets from **[github.com/acco93/filo2](https://github.com/acco93/filo2)**.
- **CVRPLIB**: Standard benchmark instances from Uchoa et al. (2017) are available at **[http://vrp.atd-lab.inf.puc-rio.br/index.php/en/](http://vrp.atd-lab.inf.puc-rio.br/index.php/en/)**.

---

## License

This project is licensed under the **GNU General Public License v3.0** (GPLv3) - see the [LICENSE](LICENSE) file for details.
