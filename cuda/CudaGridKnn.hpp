#ifndef FILO2_CUDA_GRID_KNN_HPP
#define FILO2_CUDA_GRID_KNN_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "CudaMemoryPlan.hpp"

namespace filo2_cuda {

enum class CudaGridKnnStatus {
    Success,
    NoDevice,
    Error,
};

struct CudaGridKnnResult {
    CudaGridKnnStatus status = CudaGridKnnStatus::Error;
    std::string message;

    bool succeeded() const { return status == CudaGridKnnStatus::Success; }
};

struct CudaGridKnnOptions {
    // Zero selects the deterministic sqrt(N) uniform-grid heuristic.  A
    // positive value is useful for controlled GPU experiments.
    double cell_size = 0.0;

    // The worker uses one CUDA thread per query and a bounded [id,distance]
    // max-heap in slot-major batch scratch.  A 4096-query cap bounds launch
    // latency and host staging while still exposing many blocks.  That value
    // was chosen without empirical verification — benchmark batch size first.
    std::uint64_t max_batch_queries = 32768;

    // Eight GiB is the default ceiling, independently of total device RAM.
    // FILO2_CUDA_MEMORY_CEILING_MB, when set, overrides this value in MiB.
    std::uint64_t memory_ceiling_bytes = kCudaDefaultMemoryCeilingBytes;

    // Keep this explicit reserve out of the allocation budget so driver and
    // unrelated application allocations do not consume the last bytes.
    std::uint64_t memory_reserve_bytes = 256ULL * 1024ULL * 1024ULL;

    // Target average occupancy of a non-empty spatial cell.  Using a little
    // more than one point per cell greatly reduces empty-ring traversal for
    // large k while retaining a cheap conservative grid lower bound.  This
    // field is last to preserve aggregate initialization of older options.
    double target_cell_occupancy = 8.0;
};

// Builds a compact host grid, processes query points in GPU batches, and
// writes exactly min(k, N) IDs per row.  The source point is always row[0],
// matching FILO2's existing neighbor-list invariant, including ties.
//
// This header deliberately contains no CUDA includes.  Consumers may compile
// and link their normal CPU path without the CUDA toolkit; the implementation
// is supplied by CudaGridKnn.cu only in the opt-in CUDA build.
CudaGridKnnResult compute_cuda_grid_knn(const std::vector<double>& xs,
                                        const std::vector<double>& ys,
                                        int k,
                                        std::vector<std::vector<int>>& out,
                                        const CudaGridKnnOptions& options = {});

}  // namespace filo2_cuda

#endif
