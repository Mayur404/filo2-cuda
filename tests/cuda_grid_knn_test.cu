#include "../cuda/CudaGridKnn.hpp"
#include "../knn/GridKnn.hpp"

#include <cuda_runtime.h>

#include <iostream>
#include <random>
#include <utility>
#include <vector>

namespace {

std::vector<std::vector<int>> brute_force_reference(const std::vector<double>& xs,
                                                    const std::vector<double>& ys, int k) {
    // Use the delivered CPU ground truth directly.  Keeping a second,
    // CUDA-test-local implementation could let the two implementations share
    // the same mistake without the comparison detecting it.
    return cobra::knn::BruteForceKnn(xs, ys, k);
}

bool run_fixture(const char* name, const std::vector<double>& xs,
                 const std::vector<double>& ys, int k,
                 std::uint64_t max_batch_queries = 4096,
                 double cell_size = 0.0) {
    std::vector<std::vector<int>> actual;
    filo2_cuda::CudaGridKnnOptions options;
    options.max_batch_queries = max_batch_queries;
    options.cell_size = cell_size;
    const filo2_cuda::CudaGridKnnResult status =
        filo2_cuda::compute_cuda_grid_knn(xs, ys, k, actual, options);
    if (status.status != filo2_cuda::CudaGridKnnStatus::Success) {
        std::cerr << name << ": CUDA call failed: " << status.message << "\n";
        return false;
    }
    const auto expected = brute_force_reference(xs, ys, k);
    if (actual != expected) {
        std::cerr << name << ": GPU output differs from CPU brute-force reference\n";
        for (std::size_t query = 0; query < actual.size(); ++query) {
            if (actual[query] != expected[query]) {
                std::cerr << "first mismatch query " << query << "\n";
                break;
            }
        }
        return false;
    }
    std::cout << name << ": PASS\n";
    return true;
}

bool run_determinism_check(const char* name, const std::vector<double>& xs,
                           const std::vector<double>& ys, int k,
                           std::uint64_t max_batch_queries) {
    filo2_cuda::CudaGridKnnOptions options;
    options.max_batch_queries = max_batch_queries;
    std::vector<std::vector<int>> first;
    std::vector<std::vector<int>> second;
    const filo2_cuda::CudaGridKnnResult first_status =
        filo2_cuda::compute_cuda_grid_knn(xs, ys, k, first, options);
    const filo2_cuda::CudaGridKnnResult second_status =
        filo2_cuda::compute_cuda_grid_knn(xs, ys, k, second, options);
    if (!first_status.succeeded() || !second_status.succeeded()) {
        std::cerr << name << ": repeated CUDA call failed\n";
        return false;
    }
    if (first != second) {
        std::cerr << name << ": repeated CUDA calls produced different output\n";
        return false;
    }
    std::cout << name << ": PASS\n";
    return true;
}

}  // namespace

int main() {
    int device_count = 0;
    const cudaError_t device_status = cudaGetDeviceCount(&device_count);
    if (device_status != cudaSuccess || device_count == 0) {
        std::cout << "cuda_grid_knn_test: SKIP (no CUDA device/runtime)\n";
        return 77;
    }

    std::mt19937_64 generator(0xF102ULL);
    std::uniform_real_distribution<double> coordinate(-100.0, 100.0);
    auto random_fixture = [&](int n) {
        std::vector<double> xs(static_cast<std::size_t>(n));
        std::vector<double> ys(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) {
            xs[static_cast<std::size_t>(i)] = coordinate(generator);
            ys[static_cast<std::size_t>(i)] = coordinate(generator);
        }
        return std::make_pair(std::move(xs), std::move(ys));
    };

    auto fixture_100 = random_fixture(100);
    auto fixture_1000 = random_fixture(1000);
    auto fixture_10000 = random_fixture(10000);
    if (!run_fixture("N=100", fixture_100.first, fixture_100.second, 25) ||
        !run_fixture("k=1", fixture_100.first, fixture_100.second, 1) ||
        !run_fixture("N=1000", fixture_1000.first, fixture_1000.second, 64) ||
        !run_fixture("N=10000", fixture_10000.first, fixture_10000.second, 128)) {
        return 1;
    }

    // Duplicate coordinates stress equal-distance ordering and the self-first
    // invariant that FILO2's existing Instance implementation requires.
    std::vector<double> duplicates_x(128);
    std::vector<double> duplicates_y(128);
    for (int i = 0; i < 128; ++i) {
        duplicates_x[static_cast<std::size_t>(i)] = static_cast<double>(i % 4);
        duplicates_y[static_cast<std::size_t>(i)] = static_cast<double>((i / 4) % 4);
    }
    if (!run_fixture("duplicates", duplicates_x, duplicates_y, 32)) return 1;

    // With a fixed unit cell and the out-of-the-way (0, 3) anchor setting the
    // grid origin, query 0's ring-1 candidate (ID 2) and ring-2 candidate
    // (ID 1) are both exactly 1.25 units away, or 1.5625 in squared-distance
    // terms.  The smaller ID is deliberately encountered in the later ring,
    // validating deterministic tie ordering independently of traversal order.
    const std::vector<double> tie_x{0.75, 2.0, 1.5, 3.0, 0.0};
    const std::vector<double> tie_y{0.75, 0.75, 1.75, 0.0, 3.0};
    if (!run_fixture("later-ring tie", tie_x, tie_y, 2, 3, 1.0)) return 1;

    auto small_fixture = random_fixture(17);
    if (!run_fixture("N<=k", small_fixture.first, small_fixture.second, 100)) return 1;

    // Keep this moderate enough for the O(N^2) reference while exercising
    // the large-k path that previously performed long global-memory shifts
    // for every accepted candidate.  The odd batch cap also verifies that a
    // k-sized scratch region is correctly reused across a final partial batch.
    auto large_k_fixture = random_fixture(2048);
    if (!run_fixture("k=1500", large_k_fixture.first, large_k_fixture.second, 1500, 73)) {
        return 1;
    }

    // Force several small batches through the same fixture.  This catches
    // query-offset, final-short-batch, and scratch-row reuse defects that a
    // default-sized single batch would leave untested.
    if (!run_fixture("forced batching", fixture_100.first, fixture_100.second, 25, 7)) return 1;
    if (!run_determinism_check("determinism", duplicates_x, duplicates_y, 32, 11)) return 1;
    std::cout << "cuda_grid_knn_test: PASS\n";
    return 0;
}
