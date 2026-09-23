#include "CudaGridKnn.hpp"

#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <new>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace filo2_cuda {
namespace {

constexpr int kThreadsPerBlock = 256;

struct HostGrid {
    double min_x = 0.0;
    double min_y = 0.0;
    double cell_size = 1.0;
    int nx = 1;
    int ny = 1;
    std::vector<int> point_ids;
    std::vector<int> offsets;
};

struct BuildResult {
    bool ok = false;
    std::string message;
    HostGrid grid;
};

std::string cuda_error(const char* operation, cudaError_t error) {
    std::ostringstream stream;
    stream << operation << " failed: " << cudaGetErrorString(error) << " ("
           << static_cast<int>(error) << ")";
    return stream.str();
}

bool checked_mul_u64(std::uint64_t a, std::uint64_t b, std::uint64_t* result) {
    if (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a) return false;
    *result = a * b;
    return true;
}

bool checked_add_u64(std::uint64_t a, std::uint64_t b, std::uint64_t* result) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) return false;
    *result = a + b;
    return true;
}

bool parse_memory_ceiling_from_env(std::uint64_t default_value, std::uint64_t* value,
                                   std::string* error) {
    *value = default_value;
    const char* env = std::getenv("FILO2_CUDA_MEMORY_CEILING_MB");
    if (env == nullptr || *env == '\0') return true;

    errno = 0;
    char* end = nullptr;
    const unsigned long long mb = std::strtoull(env, &end, 10);
    if (errno == ERANGE || end == env || *end != '\0' ||
        mb > std::numeric_limits<std::uint64_t>::max() / (1024ULL * 1024ULL)) {
        *error = "FILO2_CUDA_MEMORY_CEILING_MB must be an unsigned integer MiB value";
        return false;
    }
    *value = static_cast<std::uint64_t>(mb) * 1024ULL * 1024ULL;
    return true;
}

BuildResult build_host_grid(const std::vector<double>& xs, const std::vector<double>& ys,
                            const CudaGridKnnOptions& options) {
    BuildResult result;
    HostGrid& grid = result.grid;
    const std::size_t n = xs.size();
    if (n == 0) {
        result.message = "N must be positive";
        return result;
    }
    if (n > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        result.message = "N exceeds the 32-bit index range used by the CUDA grid";
        return result;
    }

    // [CPU-1] Validate and summarize the input point set.  The CUDA path
    // deliberately rejects non-finite coordinates before allocating device
    // memory; otherwise floor(), cell IDs, and distance ordering are undefined.
    double max_x = std::numeric_limits<double>::lowest();
    double max_y = std::numeric_limits<double>::lowest();
    grid.min_x = std::numeric_limits<double>::max();
    grid.min_y = std::numeric_limits<double>::max();
    for (std::size_t i = 0; i < n; ++i) {
        if (!std::isfinite(xs[i]) || !std::isfinite(ys[i])) {
            result.message = "coordinates must be finite";
            return result;
        }
        grid.min_x = std::min(grid.min_x, xs[i]);
        grid.min_y = std::min(grid.min_y, ys[i]);
        max_x = std::max(max_x, xs[i]);
        max_y = std::max(max_y, ys[i]);
    }
    const double range_x = max_x - grid.min_x;
    const double range_y = max_y - grid.min_y;
    if (!std::isfinite(range_x) || !std::isfinite(range_y)) {
        result.message = "coordinate range overflows double";
        return result;
    }
    // The device stores squared Euclidean distances in double.  Refuse a
    // span that could make dx*dx + dy*dy overflow to infinity and silently
    // corrupt the ordering; the caller can then use FILO2's CPU fallback.
    const double safe_span = std::sqrt(std::numeric_limits<double>::max() / 2.0);
    if (range_x > safe_span || range_y > safe_span) {
        result.message = "coordinate span is too large for finite CUDA squared distances";
        return result;
    }

    // [CPU-1] Choose a uniform grid. A small, explicit target occupancy keeps
    // large-k queries from walking thousands of mostly empty rings while
    // preserving the exact ring/lower-bound proof in the device kernel. A
    // caller can supply a fixed positive cell_size for controlled experiments.
    if (options.cell_size < 0.0 || !std::isfinite(options.cell_size) ||
        (options.cell_size == 0.0 &&
         (options.target_cell_occupancy <= 0.0 || !std::isfinite(options.target_cell_occupancy)))) {
        result.message = "cell_size must be finite and non-negative and target occupancy must be positive";
        return result;
    }
    if (options.cell_size > 0.0) {
        grid.cell_size = options.cell_size;
    } else {
        const double target_cells =
            std::max(1.0, static_cast<double>(n) / options.target_cell_occupancy);
        const double heuristic = std::max(range_x, range_y) /
                                 std::sqrt(target_cells);
        grid.cell_size = heuristic;
    }
    if (!(grid.cell_size > 0.0) || !std::isfinite(grid.cell_size)) grid.cell_size = 1.0;

    auto axis_cells = [&](double range, int* axis, const char* axis_name) -> bool {
        if (range == 0.0) {
            *axis = 1;
            return true;
        }
        const double count = std::ceil(range / grid.cell_size);
        if (!std::isfinite(count) || count < 1.0 ||
            count > static_cast<double>(std::numeric_limits<int>::max())) {
            result.message = std::string(axis_name) + " cell count is outside the supported range";
            return false;
        }
        *axis = static_cast<int>(count);
        return true;
    };
    if (!axis_cells(range_x, &grid.nx, "x") || !axis_cells(range_y, &grid.ny, "y")) return result;

    std::uint64_t cells = 0;
    if (!checked_mul_u64(static_cast<std::uint64_t>(grid.nx),
                         static_cast<std::uint64_t>(grid.ny), &cells) ||
        cells == 0 || cells > static_cast<std::uint64_t>(std::numeric_limits<int>::max() - 1)) {
        result.message = "uniform grid has too many cells for compact 32-bit offsets";
        return result;
    }

    // Compute each cell ID once and reuse it for counting and scatter.  The
    // transient int array is only N elements (and is released before the
    // device phase); avoiding a second pair of floating-point division/floor
    // operations is worthwhile during million-point preprocessing while the
    // counting layout still removes the old O(N log N) point sort.
    auto to_axis = [&](double coordinate, double origin, int extent) -> int {
        const double ratio = (coordinate - origin) / grid.cell_size;
        // Clamp before converting to an integer.  This keeps an unexpected
        // finite/overflowed ratio or NaN from reaching an out-of-range
        // floating-to-integer conversion.
        if (!(ratio > 0.0)) return 0;
        if (ratio >= static_cast<double>(extent)) return extent - 1;
        return static_cast<int>(std::floor(ratio));
    };
    const auto cell_for_point = [&](std::size_t point) -> int {
        const int ix = to_axis(xs[point], grid.min_x, grid.nx);
        const int iy = to_axis(ys[point], grid.min_y, grid.ny);
        // `cells <= INT_MAX - 1` was checked above, so this product and sum
        // are bounded before the int arithmetic and cannot wrap.
        return iy * grid.nx + ix;
    };

    std::vector<int> cell_ids(n);
    grid.point_ids.resize(n);
    grid.offsets.assign(static_cast<std::size_t>(cells) + 1, 0);
    for (std::size_t point = 0; point < n; ++point) {
        const int cell = cell_for_point(point);
        if (cell < 0) {
            result.message = "cell index overflow";
            return result;
        }
        cell_ids[point] = cell;
        ++grid.offsets[static_cast<std::size_t>(cell) + 1];
    }
    for (std::size_t cell = 1; cell < grid.offsets.size(); ++cell) {
        grid.offsets[cell] += grid.offsets[cell - 1];
    }
    std::vector<int> write_positions(grid.offsets.begin(), grid.offsets.end() - 1);
    // Points are visited in increasing ID order, so each CSR cell is stable
    // without a comparison sort. This makes equal-distance ties deterministic
    // and matches the original (cell ID, point ID) ordering.
    for (std::size_t point = 0; point < n; ++point) {
        const int cell = cell_ids[point];
        grid.point_ids[static_cast<std::size_t>(write_positions[static_cast<std::size_t>(cell)]++)] =
            static_cast<int>(point);
    }
    result.ok = true;
    return result;
}

__device__ __forceinline__ bool better_candidate(int candidate_id, double candidate_distance,
                                                   int current_id, double current_distance,
                                                   int query_id) {
    // Match the CPU reference's lexicographic order exactly: distance is the
    // primary key, then the query itself wins an exact tie, then the smaller
    // vertex ID wins.  The self-distance is normally zero, but keeping the
    // tie condition explicit also preserves the ordering if a future metric
    // or finite-distance guard changes that assumption.
    if (candidate_distance < current_distance) return true;
    if (candidate_distance > current_distance) return false;
    if (candidate_id == query_id && current_id != query_id) return true;
    if (candidate_id != query_id && current_id == query_id) return false;
    return candidate_id < current_id;
}

// The bounded heap is a max heap under the exact inverse of the neighbor
// ordering: its root is the current worst retained candidate. This changes
// each accepted candidate from a potentially k-element global-memory shift
// into O(log k) swaps while keeping the same distance/self/ID tie rule.
__device__ __forceinline__ void heap_sift_down(int* ids, double* distances, int count,
                                                int index, int query_id) {
    for (;;) {
        const long long left_ll = static_cast<long long>(index) * 2LL + 1LL;
        if (left_ll >= count) break;
        const int left = static_cast<int>(left_ll);
        int worst_child = left;
        const int right = left + 1;
        if (right < count &&
            better_candidate(ids[left], distances[left], ids[right], distances[right], query_id)) {
            // The left child is better, so the right child is worse and must
            // be selected to preserve the max-heap invariant.
            worst_child = right;
        }
        if (!better_candidate(ids[index], distances[index], ids[worst_child],
                              distances[worst_child], query_id)) {
            break;
        }
        const int id = ids[index];
        ids[index] = ids[worst_child];
        ids[worst_child] = id;
        const double distance = distances[index];
        distances[index] = distances[worst_child];
        distances[worst_child] = distance;
        index = worst_child;
    }
}

__device__ __forceinline__ void heapify(int* ids, double* distances, int count, int query_id) {
    // The first k candidates are appended without ordering; one bottom-up
    // pass builds the bounded heap in O(k), avoiding O(k log k) sift-ups for
    // every query before the heap is full.
    for (int index = (count >> 1) - 1; index >= 0; --index) {
        heap_sift_down(ids, distances, count, index, query_id);
    }
}

__device__ __forceinline__ void insert_candidate(int* ids, double* distances, int* count, int k,
                                                  int candidate_id, double candidate_distance,
                                                  int query_id) {
    if (*count < k) {
        const int position = *count;
        ids[position] = candidate_id;
        distances[position] = candidate_distance;
        ++(*count);
        if (*count == k) heapify(ids, distances, *count, query_id);
    } else if (better_candidate(candidate_id, candidate_distance, ids[0], distances[0], query_id)) {
        ids[0] = candidate_id;
        distances[0] = candidate_distance;
        heap_sift_down(ids, distances, *count, 0, query_id);
    }
}

__device__ __forceinline__ void heap_sort_best_first(int* ids, double* distances, int count,
                                                      int query_id) {
    // Root is worst, so moving it to the end on each pass produces the exact
    // best-to-worst ordering expected by the host neighbor lists.
    for (int end = count - 1; end > 0; --end) {
        const int id = ids[0];
        ids[0] = ids[end];
        ids[end] = id;
        const double distance = distances[0];
        distances[0] = distances[end];
        distances[end] = distance;
        heap_sift_down(ids, distances, end, 0, query_id);
    }
}

__device__ __forceinline__ double point_distance(double qx, double qy, double px, double py) {
    const double dx = qx - px;
    const double dy = qy - py;
    return dx * dx + dy * dy;
}

__device__ __forceinline__ double outside_lower_bound(double qx, double qy,
                                                       int qix, int qiy, int radius,
                                                       int nx, int ny, double min_x,
                                                       double min_y, double cell_size) {
    // Do +/- radius arithmetic in 64 bits before clamping.  A deliberately
    // very fine one-dimensional grid can make qix + radius overflow int.
    const long long qix_ll = qix;
    const long long qiy_ll = qiy;
    const long long radius_ll = radius;
    const int low_x = qix_ll - radius_ll > 0 ? static_cast<int>(qix_ll - radius_ll) : 0;
    const int high_x = qix_ll + radius_ll < static_cast<long long>(nx - 1)
                           ? static_cast<int>(qix_ll + radius_ll) : nx - 1;
    const int low_y = qiy_ll - radius_ll > 0 ? static_cast<int>(qiy_ll - radius_ll) : 0;
    const int high_y = qiy_ll + radius_ll < static_cast<long long>(ny - 1)
                           ? static_cast<int>(qiy_ll + radius_ll) : ny - 1;
    double lower = CUDART_INF;
    if (low_x > 0) {
        const double delta = qx - (min_x + static_cast<double>(low_x) * cell_size);
        lower = fmin(lower, delta > 0.0 ? delta * delta : 0.0);
    }
    if (high_x < nx - 1) {
        const double delta = min_x + static_cast<double>(high_x + 1) * cell_size - qx;
        lower = fmin(lower, delta > 0.0 ? delta * delta : 0.0);
    }
    if (low_y > 0) {
        const double delta = qy - (min_y + static_cast<double>(low_y) * cell_size);
        lower = fmin(lower, delta > 0.0 ? delta * delta : 0.0);
    }
    if (high_y < ny - 1) {
        const double delta = min_y + static_cast<double>(high_y + 1) * cell_size - qy;
        lower = fmin(lower, delta > 0.0 ? delta * delta : 0.0);
    }
    // Make the floating-point box bound conservative by one representable
    // step.  This preserves the CPU reference's "strictly closer" proof even
    // when a boundary multiplication rounds upward on the device.
    return nextafter(lower, 0.0);
}

__device__ __forceinline__ void scan_grid_cell(
        int ix, int iy, int nx, const int* __restrict__ offsets, const int* __restrict__ point_ids,
        const double* __restrict__ xs, const double* __restrict__ ys, double qx, double qy, int query,
        int k, int* __restrict__ best_ids, double* __restrict__ best_distances, int* count, double* worst_distance) {
    const int cell = iy * nx + ix;
    const int start = offsets[cell];
    const int end = offsets[cell + 1];
    for (int position = start; position < end; ++position) {
        const int point = point_ids[position];
        const double distance = point_distance(qx, qy, xs[point], ys[point]);
        // Bounded max-heap optimization: if heap is full and distance is worse than worst in heap, skip immediately
        if (*count == k && distance > *worst_distance) continue;
        insert_candidate(best_ids, best_distances, count, k, point, distance, query);
        if (*count == k) *worst_distance = best_distances[0];
    }
}

__global__ void grid_knn_kernel(const double* __restrict__ xs, const double* __restrict__ ys, const int* __restrict__ point_ids,
                                const int* __restrict__ offsets, int nx, int ny, double min_x,
                                double min_y, double cell_size, int n, int k,
                                int query_offset, int query_count, int* __restrict__ scratch_ids,
                                double* __restrict__ scratch_distances) {
    const int local_query = blockIdx.x * blockDim.x + threadIdx.x;
    if (local_query >= query_count) return;
    const int query = query_offset + local_query;
    const double qx = xs[query];
    const double qy = ys[query];
    // [CPU-2] Query/top-k initialization.  The bounded scratch row is the
    // device equivalent of the CPU reference's empty max heap.
    const double inv_cell_size = 1.0 / cell_size;
    const double raw_qix = floor((qx - min_x) * inv_cell_size);
    const double raw_qiy = floor((qy - min_y) * inv_cell_size);
    const int qix = !(raw_qix > 0.0) ? 0
                                    : (raw_qix >= static_cast<double>(nx)
                                           ? nx - 1
                                           : static_cast<int>(raw_qix));
    const int qiy = !(raw_qiy > 0.0) ? 0
                                    : (raw_qiy >= static_cast<double>(ny)
                                           ? ny - 1
                                           : static_cast<int>(raw_qiy));

    int* best_ids = scratch_ids + static_cast<std::size_t>(local_query) * k;
    double* best_distances = scratch_distances + static_cast<std::size_t>(local_query) * k;
    int count = 0;
    double worst_distance = CUDART_INF;

    // [CPU-3] Expand the query cell ring by ring.  Each ring visits exactly
    // the newly exposed boundary cells, so no point is examined twice.
    int max_ring = qix > nx - 1 - qix ? qix : nx - 1 - qix;
    const int y_ring = qiy > ny - 1 - qiy ? qiy : ny - 1 - qiy;
    if (y_ring > max_ring) max_ring = y_ring;
    for (int radius = 0; radius <= max_ring; ++radius) {
        const long long r = radius;
        const long long raw_low_x = static_cast<long long>(qix) - r;
        const long long raw_high_x = static_cast<long long>(qix) + r;
        const long long raw_low_y = static_cast<long long>(qiy) - r;
        const long long raw_high_y = static_cast<long long>(qiy) + r;
        const int first_x = raw_low_x > 0 ? static_cast<int>(raw_low_x) : 0;
        const int last_x = raw_high_x < static_cast<long long>(nx - 1)
                               ? static_cast<int>(raw_high_x) : nx - 1;
        const int first_y = raw_low_y + 1 > 0 ? static_cast<int>(raw_low_y + 1) : 0;
        const int last_y = raw_high_y - 1 < static_cast<long long>(ny - 1)
                               ? static_cast<int>(raw_high_y - 1) : ny - 1;

        // Use the raw (unclamped) ring edges to decide whether an edge is in
        // range.  Clamping the edges first would revisit boundary cells on
        // every later ring for a query near the grid boundary, inserting the
        // same point ID more than once.  This is the device counterpart of
        // UniformGrid::ForEachRing in [CPU-3].
        if (raw_low_y >= 0 && raw_low_y < ny) {
            for (int ix = first_x; ix <= last_x; ++ix) {
                scan_grid_cell(ix, static_cast<int>(raw_low_y), nx, offsets, point_ids,
                               xs, ys, qx, qy, query, k, best_ids, best_distances, &count, &worst_distance);
            }
        }
        if (raw_high_y >= 0 && raw_high_y < ny && raw_high_y != raw_low_y) {
            for (int ix = first_x; ix <= last_x; ++ix) {
                scan_grid_cell(ix, static_cast<int>(raw_high_y), nx, offsets, point_ids,
                               xs, ys, qx, qy, query, k, best_ids, best_distances, &count, &worst_distance);
            }
        }
        if (raw_low_x >= 0 && raw_low_x < nx) {
            for (int iy = first_y; iy <= last_y; ++iy) {
                scan_grid_cell(static_cast<int>(raw_low_x), iy, nx, offsets, point_ids,
                               xs, ys, qx, qy, query, k, best_ids, best_distances, &count, &worst_distance);
            }
        }
        if (raw_high_x >= 0 && raw_high_x < nx && raw_high_x != raw_low_x) {
            for (int iy = first_y; iy <= last_y; ++iy) {
                scan_grid_cell(static_cast<int>(raw_high_x), iy, nx, offsets, point_ids,
                               xs, ys, qx, qy, query, k, best_ids, best_distances, &count, &worst_distance);
            }
        }
        if (count == k) {
            const double lower = outside_lower_bound(qx, qy, qix, qiy, radius,
                                                     nx, ny, min_x, min_y, cell_size);
            // [CPU-5] Strict inequality is intentional: equal-distance unsearched
            // points can win the ID tie-break, so they must still be visited.
            if (worst_distance < lower) break;
        }
    }

    // A valid grid visits every point and therefore reaches k candidates, but
    // keep the helper exact even if a future pruning/validation change leaves
    // a short row: the append-only prefix must be heapified before sorting.
    if (count > 1 && count < k) heapify(best_ids, best_distances, count, query);
    heap_sort_best_first(best_ids, best_distances, count, query);

    // [CPU-6] Emit the bounded list into the batch scratch row.  The host
    // copies only IDs for this batch and immediately reuses the same storage.
    for (int position = 0; position < k; ++position) {
        if (position >= count) {
            best_ids[position] = -1;
        }
    }
    (void)n;
}

bool as_size_t(std::uint64_t bytes, std::size_t* result) {
    if (bytes > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) return false;
    *result = static_cast<std::size_t>(bytes);
    return true;
}

CudaGridKnnResult make_result(CudaGridKnnStatus status, std::string message) {
    CudaGridKnnResult result;
    result.status = status;
    result.message = std::move(message);
    return result;
}

}  // namespace

CudaGridKnnResult compute_cuda_grid_knn(const std::vector<double>& xs,
                                        const std::vector<double>& ys,
                                        int k,
                                        std::vector<std::vector<int>>& out,
                                        const CudaGridKnnOptions& options) {
    // [CPU-1] Match the reference input contract before querying the device.
    if (xs.size() != ys.size()) {
        return make_result(CudaGridKnnStatus::Error, "x and y coordinate arrays have different sizes");
    }
    if (xs.empty()) return make_result(CudaGridKnnStatus::Error, "N must be positive");
    if (k <= 0) return make_result(CudaGridKnnStatus::Error, "k must be positive");
    const std::size_t n_size = xs.size();
    const std::uint64_t effective_k = static_cast<std::uint64_t>(
        std::min<std::size_t>(static_cast<std::size_t>(k), n_size));
    if (effective_k == 0 || n_size > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        return make_result(CudaGridKnnStatus::Error, "N/k are outside the supported range");
    }

    int device_count = 0;
    const cudaError_t count_error = cudaGetDeviceCount(&device_count);
    if (count_error == cudaErrorNoDevice || count_error == cudaErrorInsufficientDriver ||
        (count_error == cudaSuccess && device_count == 0)) {
        return make_result(CudaGridKnnStatus::NoDevice, "no CUDA device is available; use the CPU kd-tree path");
    }
    if (count_error != cudaSuccess) return make_result(CudaGridKnnStatus::Error,
                                                        cuda_error("cudaGetDeviceCount", count_error));

    // [CPU-1] Build all host grid metadata before entering the CUDA
    // allocation section.  The sorted compact representation is shared by
    // every query batch.
    BuildResult built;
    try {
        built = build_host_grid(xs, ys, options);
    } catch (const std::bad_alloc&) {
        return make_result(CudaGridKnnStatus::Error, "host compact-grid allocation failed");
    } catch (const std::length_error& error) {
        return make_result(CudaGridKnnStatus::Error,
                           std::string("host compact-grid allocation is too large: ") + error.what());
    }
    if (!built.ok) return make_result(CudaGridKnnStatus::Error, built.message);
    HostGrid& grid = built.grid;
    const std::uint64_t cells = static_cast<std::uint64_t>(grid.nx) *
                                static_cast<std::uint64_t>(grid.ny);

    std::uint64_t ceiling = 0;
    std::string parse_error;
    if (!parse_memory_ceiling_from_env(options.memory_ceiling_bytes, &ceiling, &parse_error)) {
        return make_result(CudaGridKnnStatus::Error, parse_error);
    }
    // The CUDA API uses size_t for these values.  Keep that ABI-facing type
    // at the call boundary, then move into the checked uint64 ledger.
    std::size_t free_size = 0;
    std::size_t total_device_size = 0;
    const cudaError_t info_error = cudaMemGetInfo(&free_size, &total_device_size);
    if (info_error != cudaSuccess) return make_result(CudaGridKnnStatus::Error,
                                                       cuda_error("cudaMemGetInfo", info_error));
    const std::uint64_t free_bytes = static_cast<std::uint64_t>(free_size);
    (void)total_device_size;

    // No cudaMalloc occurs above this point.  The ledger includes every
    // allocation made below: x/y coordinates, point IDs, offsets, and both
    // per-query scratch arrays.  Explicit reserve is removed after applying
    // min(configured ceiling, cudaMemGetInfo free bytes).
    const std::uint64_t available_before_reserve = std::min(ceiling, free_bytes);
    if (options.memory_reserve_bytes >= available_before_reserve) {
        return make_result(CudaGridKnnStatus::Error,
                           "CUDA memory ceiling/free memory is smaller than the configured reserve");
    }
    const std::uint64_t available = available_before_reserve - options.memory_reserve_bytes;
    const std::uint64_t n = static_cast<std::uint64_t>(n_size);
    // One thread per query uses global [id,distance] scratch rows.  This maps
    // the CPU bounded-list steps directly and avoids a warp-wide merge plus
    // shared memory proportional to k; that traceability is why it was chosen
    // over a warp-cooperative top-k here.
    const std::uint64_t batch = max_cuda_batch_for_budget(
        n, effective_k, cells, available, options.max_batch_queries);
    if (batch == 0) {
        return make_result(CudaGridKnnStatus::Error,
                           "CUDA memory budget cannot fit static grid storage plus one query scratch row");
    }
    const CudaMemoryPlan plan = make_cuda_memory_plan(n, effective_k, cells, batch);
    if (!plan.ok || plan.total_bytes > available) {
        return make_result(CudaGridKnnStatus::Error,
                           plan.error == nullptr ? "internal CUDA memory ledger failure" : plan.error);
    }

    std::size_t coordinates_bytes = 0;
    std::size_t ids_bytes = 0;
    std::size_t offsets_bytes = 0;
    std::size_t scratch_ids_bytes = 0;
    std::size_t scratch_dist_bytes = 0;
    if (!as_size_t(plan.coordinates_bytes, &coordinates_bytes) ||
        !as_size_t(plan.point_ids_bytes, &ids_bytes) ||
        !as_size_t(plan.offsets_bytes, &offsets_bytes) ||
        !as_size_t(plan.scratch_ids_bytes, &scratch_ids_bytes) ||
        !as_size_t(plan.scratch_dist_bytes, &scratch_dist_bytes)) {
        return make_result(CudaGridKnnStatus::Error, "CUDA allocation size does not fit size_t");
    }

    // The output is intentionally allocated on the host one row at a time in
    // semantic terms, while device scratch remains only batch*k.  FILO2's
    // final neighbor list necessarily has N*k host entries; no N*k device
    // output allocation is made here.
    try {
        // Resize in place so a caller that reuses `out` retains row capacity;
        // constructing one temporary row and copying it N times adds an
        // avoidable allocation/copy pass at million-customer scale.
        out.resize(n_size);
#if defined(_OPENMP)
        #pragma omp parallel for schedule(static)
#endif
        for (std::int64_t i = 0; i < static_cast<std::int64_t>(n_size); ++i) {
            out[i].resize(static_cast<std::size_t>(effective_k));
        }
    } catch (const std::bad_alloc&) {
        return make_result(CudaGridKnnStatus::Error, "host neighbor-list output allocation failed");
    } catch (const std::length_error&) {
        return make_result(CudaGridKnnStatus::Error, "host neighbor-list output size is too large");
    }

    double* d_coordinates = nullptr;
    double* d_x = nullptr;
    double* d_y = nullptr;
    int* d_point_ids = nullptr;
    int* d_offsets = nullptr;
    int* d_scratch_ids = nullptr;
    double* d_scratch_distances = nullptr;
    int* pinned_batch_ids = nullptr;
    int* pinned_next_batch_ids = nullptr;
    std::vector<int> fallback_batch_ids;
    auto release = [&]() {
        if (pinned_next_batch_ids != nullptr) cudaFreeHost(pinned_next_batch_ids);
        if (pinned_batch_ids != nullptr) cudaFreeHost(pinned_batch_ids);
        if (d_scratch_distances != nullptr) cudaFree(d_scratch_distances);
        if (d_scratch_ids != nullptr) cudaFree(d_scratch_ids);
        if (d_offsets != nullptr) cudaFree(d_offsets);
        if (d_point_ids != nullptr) cudaFree(d_point_ids);
        if (d_coordinates != nullptr) cudaFree(d_coordinates);
    };
    auto allocate = [&](void** pointer, std::size_t bytes, const char* name) -> CudaGridKnnResult {
        const cudaError_t error = cudaMalloc(pointer, bytes);
        if (error != cudaSuccess) {
            release();
            return make_result(CudaGridKnnStatus::Error, cuda_error(name, error));
        }
        return make_result(CudaGridKnnStatus::Success, "");
    };

    // Allocation order matches the ledger above.  This is the first actual
    // device allocation in the function.
    CudaGridKnnResult allocation_result = allocate(reinterpret_cast<void**>(&d_coordinates),
                                                    coordinates_bytes, "cudaMalloc(coordinates)");
    if (!allocation_result.succeeded()) return allocation_result;
    d_x = d_coordinates;
    d_y = d_coordinates + n_size;
    allocation_result = allocate(reinterpret_cast<void**>(&d_point_ids), ids_bytes,
                                 "cudaMalloc(d_point_ids)");
    if (!allocation_result.succeeded()) return allocation_result;
    allocation_result = allocate(reinterpret_cast<void**>(&d_offsets), offsets_bytes,
                                 "cudaMalloc(d_offsets)");
    if (!allocation_result.succeeded()) return allocation_result;
    allocation_result = allocate(reinterpret_cast<void**>(&d_scratch_ids), scratch_ids_bytes,
                                 "cudaMalloc(d_scratch_ids)");
    if (!allocation_result.succeeded()) return allocation_result;
    allocation_result = allocate(reinterpret_cast<void**>(&d_scratch_distances), scratch_dist_bytes,
                                 "cudaMalloc(d_scratch_distances)");
    if (!allocation_result.succeeded()) return allocation_result;

    auto copy_to_device = [&](void* destination, const void* source, std::size_t bytes,
                              const char* name) -> CudaGridKnnResult {
        const cudaError_t error = cudaMemcpy(destination, source, bytes, cudaMemcpyHostToDevice);
        if (error != cudaSuccess) {
            release();
            return make_result(CudaGridKnnStatus::Error, cuda_error(name, error));
        }
        return make_result(CudaGridKnnStatus::Success, "");
    };
    // The source coordinate arrays remain alive for the duration of this
    // call; avoid a second host-side x/y copy in HostGrid.
    const std::size_t coordinate_row_bytes = coordinates_bytes / 2;
    allocation_result = copy_to_device(d_x, xs.data(), coordinate_row_bytes, "cudaMemcpy(x)");
    if (!allocation_result.succeeded()) return allocation_result;
    allocation_result = copy_to_device(d_y, ys.data(), coordinate_row_bytes, "cudaMemcpy(y)");
    if (!allocation_result.succeeded()) return allocation_result;
    allocation_result = copy_to_device(d_point_ids, grid.point_ids.data(), ids_bytes,
                                       "cudaMemcpy(point_ids)");
    if (!allocation_result.succeeded()) return allocation_result;
    allocation_result = copy_to_device(d_offsets, grid.offsets.data(), offsets_bytes,
                                       "cudaMemcpy(offsets)");
    if (!allocation_result.succeeded()) return allocation_result;

    const std::size_t max_batch_ids = static_cast<std::size_t>(batch) * static_cast<std::size_t>(effective_k);
    if (cudaMallocHost(&pinned_batch_ids, max_batch_ids * sizeof(int)) != cudaSuccess) {
        pinned_batch_ids = nullptr;
        try {
            fallback_batch_ids.resize(max_batch_ids);
        } catch (...) {
            release();
            return make_result(CudaGridKnnStatus::Error, "host batch output allocation failed");
        }
    }
    // With two pinned buffers, the GPU can produce the next batch while the
    // CPU scatters the completed batch into its final neighbor rows. A second
    // allocation is optional: the existing synchronous path remains usable
    // when pinned host memory is scarce.
    if (pinned_batch_ids != nullptr && n > batch &&
        cudaMallocHost(&pinned_next_batch_ids, max_batch_ids * sizeof(int)) != cudaSuccess) {
        pinned_next_batch_ids = nullptr;
    }
    const bool pipelined = pinned_next_batch_ids != nullptr;
    auto scatter_batch = [&](std::uint64_t query_offset, std::uint64_t query_count,
                             const int* source_batch) {
#if defined(_OPENMP)
        #pragma omp parallel for schedule(static)
#endif
        for (std::int64_t local = 0; local < static_cast<std::int64_t>(query_count); ++local) {
            const std::size_t query = static_cast<std::size_t>(query_offset) + static_cast<std::size_t>(local);
            const int* source = source_batch +
                                static_cast<std::size_t>(local) * static_cast<std::size_t>(effective_k);
            std::copy_n(source, static_cast<std::size_t>(effective_k), out[query].data());
        }
    };
    std::uint64_t previous_offset = 0;
    std::uint64_t previous_count = 0;
    const int* previous_batch = nullptr;
    bool use_next_buffer = false;

    for (std::uint64_t query_offset = 0; query_offset < n; query_offset += batch) {
        const std::uint64_t query_count_u64 = std::min(batch, n - query_offset);
        const int query_offset_i = static_cast<int>(query_offset);
        const int query_count = static_cast<int>(query_count_u64);
        // Avoid query_count + (block_size - 1) overflow for a caller that
        // supplies a very large batch cap near INT_MAX.
        const int blocks = query_count / kThreadsPerBlock +
                           (query_count % kThreadsPerBlock != 0 ? 1 : 0);
        grid_knn_kernel<<<blocks, kThreadsPerBlock>>>(
            d_x, d_y, d_point_ids, d_offsets, grid.nx, grid.ny, grid.min_x, grid.min_y,
            grid.cell_size, static_cast<int>(n), static_cast<int>(effective_k), query_offset_i,
            query_count, d_scratch_ids, d_scratch_distances);
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            release();
            return make_result(CudaGridKnnStatus::Error, cuda_error("grid_knn_kernel launch", error));
        }
        const std::size_t copied_ids = static_cast<std::size_t>(query_count_u64) *
                                       static_cast<std::size_t>(effective_k);
        int* dest_batch = pinned_batch_ids == nullptr ? fallback_batch_ids.data() :
                          (use_next_buffer ? pinned_next_batch_ids : pinned_batch_ids);
        if (pipelined) {
            error = cudaMemcpyAsync(dest_batch, d_scratch_ids,
                                    copied_ids * sizeof(int), cudaMemcpyDeviceToHost);
        } else {
            error = cudaMemcpy(dest_batch, d_scratch_ids,
                               copied_ids * sizeof(int), cudaMemcpyDeviceToHost);
        }
        if (error != cudaSuccess) {
            release();
            return make_result(CudaGridKnnStatus::Error, cuda_error("cudaMemcpy(batch IDs)", error));
        }
        if (pipelined) {
            // The previous transfer was synchronized at the end of the last
            // iteration. Its host buffer is safe to read while this kernel
            // and D2H transfer run in the default CUDA stream.
            if (previous_batch != nullptr) {
                scatter_batch(previous_offset, previous_count, previous_batch);
            }
            error = cudaStreamSynchronize(nullptr);
            if (error != cudaSuccess) {
                release();
                return make_result(CudaGridKnnStatus::Error, cuda_error("cudaStreamSynchronize(batch)", error));
            }
            previous_offset = query_offset;
            previous_count = query_count_u64;
            previous_batch = dest_batch;
            use_next_buffer = !use_next_buffer;
        } else {
            scatter_batch(query_offset, query_count_u64, dest_batch);
        }
    }
    if (pipelined) scatter_batch(previous_offset, previous_count, previous_batch);
    release();
    std::ostringstream message;
    message << "CUDA grid k-NN succeeded (N=" << n << ", k=" << effective_k
            << ", cells=" << cells << ", batch=" << batch << ", device bytes="
            << plan.total_bytes << ")";
    return make_result(CudaGridKnnStatus::Success, message.str());
}

}  // namespace filo2_cuda
