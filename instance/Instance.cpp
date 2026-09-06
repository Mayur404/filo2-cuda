#include "Instance.hpp"

#include <algorithm>
#include <iostream>
#include <utility>

#include "base/KDTree.hpp"
#include "base/Timer.hpp"

#ifdef FILO2_USE_CUDA
    #include "cuda/CudaGridKnn.hpp"
#endif

namespace cobra {

    // static
    std::optional<Instance> Instance::make(const std::string& filepath, int neighbors_num, int threads_num,
                                           const std::vector<std::vector<int>>& neighbors_) {

        Parser parser(filepath);

        std::optional<InstanceData> maybe_data = parser.Parse();
        if (!maybe_data.has_value()) {
            return std::nullopt;
        }

        return Instance(std::move(maybe_data.value()), neighbors_num, threads_num, neighbors_);
    }

    // static
    Instance Instance::make(const InstanceData& data, int num_neighbors, int threads_num,
                            const std::vector<std::vector<int>>& neighbors_) {
        return Instance(data, num_neighbors, threads_num, neighbors_);
    }

    // static
    Instance Instance::make(InstanceData&& data, int num_neighbors, int threads_num,
                            const std::vector<std::vector<int>>& neighbors_) {
        return Instance(std::move(data), num_neighbors, threads_num, neighbors_);
    }

    Instance::Instance(InstanceData data, int neighbors_num, int threads_num,
                       const std::vector<std::vector<int>>& neighbors_) {

        neighbors_num = std::max(0, std::min(neighbors_num, static_cast<int>(data.demands.size())));

        // Copy info from parsed data.
        vehicle_capacity = data.vehicle_capacity;
        xcoords = std::move(data.xcoords);
        ycoords = std::move(data.ycoords);
        demands = std::move(data.demands);

        // Cache the depot distance row that is repeatedly queried by
        // initial construction, savings, routemin, and ruin-and-recreate.
        depot_costs.resize(xcoords.size());
        if (!xcoords.empty()) {
            for (std::size_t i = 0; i < xcoords.size(); ++i) {
                const double dx = xcoords[i] - xcoords[get_depot()];
                const double dy = ycoords[i] - ycoords[get_depot()];
                depot_costs[i] = fastround(std::sqrt(dx * dx + dy * dy));
            }
        }

        if (!neighbors_.empty()) {
            assert(static_cast<int>(neighbors_.size()) == get_vertices_num());
#ifndef NDEBUG
            for (int i = get_vertices_begin(); i < get_vertices_end(); ++i) {
                assert(!neighbors_[i].empty());
            }
#endif
            neighbors = neighbors_;
            return;
        }

        neighbors.resize(get_vertices_num());

#ifdef FILO2_USE_CUDA
        // The CUDA implementation has the same row layout as FILO2's
        // existing neighbor lists: one vector per vertex, sorted by the
        // coordinate distance, with the query vertex in position zero.  A
        // CUDA build is still allowed to run on a CPU-only host: the API
        // reports NoDevice (or Error), and the OpenMP kd-tree code below
        // remains the authoritative fallback.
        std::vector<std::vector<int>> cuda_neighbors;
        const auto cuda_result = filo2_cuda::compute_cuda_grid_knn(
            xcoords, ycoords, neighbors_num, cuda_neighbors);
        bool cuda_shape_is_valid =
            cuda_neighbors.size() == static_cast<std::size_t>(get_vertices_num());
        if (cuda_shape_is_valid && get_vertices_num() > 0) {
            const std::size_t n = static_cast<std::size_t>(get_vertices_num());
            const std::size_t samples[] = {0, n / 4, n / 2, 3 * n / 4, n - 1};
            for (std::size_t s : samples) {
                if (cuda_neighbors[s].size() != static_cast<std::size_t>(neighbors_num) ||
                    (neighbors_num > 0 && cuda_neighbors[s].front() != static_cast<int>(s))) {
                    cuda_shape_is_valid = false;
                    break;
                }
            }
        }
        if (cuda_result.succeeded() && cuda_shape_is_valid) {
            neighbors = std::move(cuda_neighbors);
            return;
        }

        std::cerr << "WARNING: CUDA neighbor-list preprocessing unavailable; "
                     "falling back to CPU kd-tree path";
        if (!cuda_result.message.empty()) {
            std::cerr << " (" << cuda_result.message << ")";
        }
        std::cerr << ".\n";
#endif

        auto set_first_neighbor = [&](const int i) {
            if (neighbors[i][0] != i) {
                const auto self = std::find(neighbors[i].begin() + 1, neighbors[i].end(), i);
                if (self == neighbors[i].end()) {
                    neighbors[i].back() = i;
                    std::swap(neighbors[i].front(), neighbors[i].back());
                } else {
                    std::iter_swap(neighbors[i].begin(), self);
                }
            }
        };

        KDTree kd_tree(xcoords, ycoords);

#if defined(_OPENMP)
        #pragma omp parallel for schedule(static) num_threads(threads_num)
#endif
        for (int i = get_vertices_begin(); i < get_vertices_end(); ++i) {
            neighbors[i] = kd_tree.GetNearestNeighbors(xcoords[i], ycoords[i], neighbors_num);
            if (neighbors_num > 0) {
                set_first_neighbor(i);
                assert(neighbors[i][0] == i);
            }
        }
    }

}  // namespace cobra
