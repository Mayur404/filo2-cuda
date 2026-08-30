#include "../cuda/CudaMemoryPlan.hpp"

#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

}  // namespace

int main() {
    using filo2_cuda::CudaMemoryPlan;
    using filo2_cuda::make_cuda_memory_plan;
    using filo2_cuda::max_cuda_batch_for_budget;

    const CudaMemoryPlan one = make_cuda_memory_plan(1000, 25, 1024, 1);
    require(one.ok, "ordinary plan rejected");
    require(one.coordinates_bytes == 1000ULL * 2ULL * sizeof(double), "coordinate ledger mismatch");
    require(one.point_ids_bytes == 1000ULL * sizeof(int), "point-ID ledger mismatch");
    require(one.offsets_bytes == 1025ULL * sizeof(int), "offset ledger mismatch");
    require(one.scratch_ids_bytes == 25ULL * sizeof(int), "scratch-ID ledger mismatch");
    require(one.scratch_dist_bytes == 25ULL * sizeof(double), "scratch-distance ledger mismatch");
    require(one.total_bytes == one.coordinates_bytes + one.point_ids_bytes + one.offsets_bytes +
                                   one.scratch_ids_bytes + one.scratch_dist_bytes,
            "total ledger mismatch");

    const std::uint64_t static_bytes = one.total_bytes - one.scratch_ids_bytes - one.scratch_dist_bytes;
    const std::uint64_t per_query = one.scratch_ids_bytes + one.scratch_dist_bytes;
    // Exactly two rows fit below this boundary; a third row is one byte too large.
    const std::uint64_t boundary_budget = static_bytes + 2ULL * per_query;
    require(max_cuda_batch_for_budget(1000, 25, 1024, boundary_budget, 1000) == 2,
            "two-row budget boundary mismatch");
    require(max_cuda_batch_for_budget(1000, 25, 1024, boundary_budget - 1, 1000) == 1,
            "one-row budget boundary mismatch");
    require(max_cuda_batch_for_budget(1000, 25, 1024, one.total_bytes - 1, 1000) == 0,
            "undersized budget accepted");
    require(max_cuda_batch_for_budget(1000, 25, 1024, boundary_budget, 1) == 1,
            "requested batch cap ignored");

    require(!make_cuda_memory_plan(0, 1, 1, 1).ok, "zero N accepted");
    require(!make_cuda_memory_plan(1, 0, 1, 1).ok, "zero k accepted");
    require(!make_cuda_memory_plan(1, 1, 0, 1).ok, "zero cells accepted");
    require(!make_cuda_memory_plan(1, 1, 1, 0).ok, "zero batch accepted");
    require(!make_cuda_memory_plan(std::numeric_limits<std::uint64_t>::max(), 1, 1, 1).ok,
            "out-of-range N accepted");
    require(!make_cuda_memory_plan(1, std::numeric_limits<std::uint64_t>::max(), 1, 1).ok,
            "overflowing k accepted");
    require(!make_cuda_memory_plan(1, 1, std::numeric_limits<std::uint64_t>::max(), 1).ok,
            "overflowing cells accepted");
    require(!make_cuda_memory_plan(1, 1, 1, std::numeric_limits<std::uint64_t>::max()).ok,
            "overflowing batch accepted");

    std::cout << "cuda_memory_plan_test: PASS\n";
    return 0;
}
