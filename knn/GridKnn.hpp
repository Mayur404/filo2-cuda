#ifndef FILO2_KNN_GRID_KNN_HPP_
#define FILO2_KNN_GRID_KNN_HPP_

#include <cstddef>
#include <vector>

namespace cobra {
namespace knn {

// A coordinate-only representation is useful to both the CPU reference and
// the CUDA comparison harness.  Vertex IDs are the zero-based positions in
// the input vector; no coordinate-based renumbering is performed.
struct Point2D {
    double x;
    double y;
};

using NeighborList = std::vector<int>;
using NeighborLists = std::vector<NeighborList>;

// Compute the effective number of neighbors.  The implementation accepts an
// int to match FILO2's existing n_nn parameter and rejects negative values.
// k == 0 is valid and produces an empty list for every query.
int EffectiveK(std::size_t point_count, int k);

// Exact CPU reference implementation.  For each input point i, the result is
// sorted by (squared distance, i first on equal distance, vertex ID).  Thus
// result[i][0] is i whenever k > 0.  The grid is an acceleration structure;
// this function has the same result as BruteForceKnn for every finite input.
NeighborLists GridKnn(const std::vector<Point2D>& points, int k);

// Convenience overload matching the coordinate storage used by Instance.
NeighborLists GridKnn(const std::vector<double>& xcoords,
                      const std::vector<double>& ycoords,
                      int k);

// O(N^2) ground truth used by the executable CPU tests and by later
// GPU-vs-CPU comparison tests.  It uses precisely the same deterministic
// ordering rule as GridKnn.
NeighborLists BruteForceKnn(const std::vector<Point2D>& points, int k);
NeighborLists BruteForceKnn(const std::vector<double>& xcoords,
                            const std::vector<double>& ycoords,
                            int k);

// Lower-case aliases make the API convenient in small external test harnesses
// while retaining the FILO2-style capitalized entry points above.
inline NeighborLists grid_knn(const std::vector<Point2D>& points, int k) {
    return GridKnn(points, k);
}

inline NeighborLists brute_force_knn(const std::vector<Point2D>& points, int k) {
    return BruteForceKnn(points, k);
}

}  // namespace knn

// Also make the two principal functions available in cobra, which is the
// namespace used by the existing FILO2 code.  The nested namespace remains
// available for callers that prefer an explicit k-NN namespace.
using knn::BruteForceKnn;
using knn::GridKnn;
using knn::NeighborList;
using knn::NeighborLists;
using knn::Point2D;

}  // namespace cobra

#endif  // FILO2_KNN_GRID_KNN_HPP_
