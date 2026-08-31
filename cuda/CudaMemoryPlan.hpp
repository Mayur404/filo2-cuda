#ifndef FILO2_CUDA_MEMORY_PLAN_HPP
#define FILO2_CUDA_MEMORY_PLAN_HPP

#include <cstddef>
#include <cstdint>
#include <limits>

namespace filo2_cuda {

// A host-only description of the allocations made by CudaGridKnn.cu.  Keep
// this type free of CUDA headers: it is intentionally usable by a normal
// C++17 build and by the boundary/overflow tests on a machine without nvcc.
struct CudaMemoryPlan {
    std::uint64_t n = 0;
    std::uint64_t k = 0;
    std::uint64_t cells = 0;
    std::uint64_t batch_queries = 0;

    std::uint64_t coordinates_bytes = 0;  // x and y, both double[N]
    std::uint64_t point_ids_bytes = 0;    // int[N]
    std::uint64_t offsets_bytes = 0;      // int[cells + 1]
    // Heap scratch is stored slot-major as [k][batch] so neighboring query
    // threads address neighboring words when they inspect the same heap slot.
    std::uint64_t scratch_ids_bytes = 0;  // int[k * batch]
    std::uint64_t scratch_dist_bytes = 0; // double[k * batch]
    // The final tiled transpose writes row-major IDs for one coalesced D2H
    // copy.  This is separate from scratch_ids because an in-place transpose
    // would introduce unnecessary synchronization and permutation overhead.
    std::uint64_t output_ids_bytes = 0;  // int[batch * k]
    std::uint64_t total_bytes = 0;

    bool ok = false;
    const char* error = nullptr;
};

constexpr std::uint64_t kCudaDefaultMemoryCeilingBytes = 8ULL * 1024ULL * 1024ULL * 1024ULL;

namespace detail {

constexpr bool add_overflow(std::uint64_t a, std::uint64_t b, std::uint64_t* out) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) return true;
    *out = a + b;
    return false;
}

constexpr bool mul_overflow(std::uint64_t a, std::uint64_t b, std::uint64_t* out) {
    if (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a) return true;
    *out = a * b;
    return false;
}

inline bool add_field(std::uint64_t* total, std::uint64_t value) {
    return add_overflow(*total, value, total);
}

}  // namespace detail

// Compute the exact device allocation ledger for one batch.  The CUDA path
// calls this before its first cudaMalloc.  This function intentionally uses
// uint64_t arithmetic throughout so malformed or adversarial N/k values
// cannot wrap into an apparently-small allocation.
inline CudaMemoryPlan make_cuda_memory_plan(std::uint64_t n, std::uint64_t k,
                                            std::uint64_t cells,
                                            std::uint64_t batch_queries) {
    CudaMemoryPlan plan;
    plan.n = n;
    plan.k = k;
    plan.cells = cells;
    plan.batch_queries = batch_queries;
    if (n == 0 || k == 0 || batch_queries == 0 || cells == 0) {
        plan.error = "N, k, cells, and batch_queries must all be non-zero";
        return plan;
    }
    std::uint64_t cell_count_plus_one = 0;
    if (detail::add_overflow(cells, 1, &cell_count_plus_one)) {
        plan.error = "grid offset count overflows uint64_t";
        return plan;
    }
    if (n > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
        plan.error = "N or grid cell count does not fit the 32-bit CUDA index representation";
        return plan;
    }
    if (cells > static_cast<std::uint64_t>(std::numeric_limits<int>::max() - 1)) {
        plan.error = "N or grid cell count does not fit the 32-bit CUDA index representation";
        return plan;
    }

    if (detail::mul_overflow(n, 2ULL * sizeof(double), &plan.coordinates_bytes) ||
        detail::mul_overflow(n, sizeof(int), &plan.point_ids_bytes) ||
        detail::mul_overflow(cell_count_plus_one, sizeof(int), &plan.offsets_bytes)) {
        plan.error = "static grid allocation size overflows uint64_t";
        return plan;
    }
    std::uint64_t batch_k = 0;
    if (detail::mul_overflow(batch_queries, k, &batch_k) ||
        detail::mul_overflow(batch_k, sizeof(int), &plan.scratch_ids_bytes) ||
        detail::mul_overflow(batch_k, sizeof(double), &plan.scratch_dist_bytes) ||
        detail::mul_overflow(batch_k, sizeof(int), &plan.output_ids_bytes)) {
        plan.error = "per-batch allocation size overflows uint64_t";
        return plan;
    }
    plan.total_bytes = 0;
    if (detail::add_field(&plan.total_bytes, plan.coordinates_bytes) ||
        detail::add_field(&plan.total_bytes, plan.point_ids_bytes) ||
        detail::add_field(&plan.total_bytes, plan.offsets_bytes) ||
        detail::add_field(&plan.total_bytes, plan.scratch_ids_bytes) ||
        detail::add_field(&plan.total_bytes, plan.scratch_dist_bytes) ||
        detail::add_field(&plan.total_bytes, plan.output_ids_bytes)) {
        plan.error = "total CUDA allocation size overflows uint64_t";
        return plan;
    }
    plan.ok = true;
    return plan;
}

// Return the largest batch that fits under available_bytes.  available_bytes
// is expected to have already had the explicit safety reserve removed by the
// caller.  The function never returns zero for a valid query if even one
// query fits; this makes the CUDA caller's batch loop easy to audit.
inline std::uint64_t max_cuda_batch_for_budget(std::uint64_t n, std::uint64_t k,
                                               std::uint64_t cells,
                                               std::uint64_t available_bytes,
                                               std::uint64_t requested_cap = 4096) {
    if (n == 0 || k == 0 || cells == 0 || available_bytes == 0 || requested_cap == 0) return 0;
    const CudaMemoryPlan one = make_cuda_memory_plan(n, k, cells, 1);
    if (!one.ok || one.total_bytes > available_bytes) return 0;
    const std::uint64_t static_bytes = one.total_bytes - one.scratch_ids_bytes -
                                       one.scratch_dist_bytes - one.output_ids_bytes;
    const std::uint64_t per_query = one.scratch_ids_bytes + one.scratch_dist_bytes +
                                    one.output_ids_bytes;
    if (per_query == 0 || available_bytes < static_bytes) return 0;
    std::uint64_t by_budget = (available_bytes - static_bytes) / per_query;
    if (by_budget == 0) return 0;
    if (by_budget > n) by_budget = n;
    if (by_budget > requested_cap) by_budget = requested_cap;
    // Guard the arithmetic in case this helper is changed to use a different
    // allocation set in the future.
    while (by_budget > 0) {
        const CudaMemoryPlan candidate = make_cuda_memory_plan(n, k, cells, by_budget);
        if (candidate.ok && candidate.total_bytes <= available_bytes) return by_budget;
        --by_budget;
    }
    return 0;
}

}  // namespace filo2_cuda

#endif
