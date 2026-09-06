#include "GridKnn.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <queue>
#include <stdexcept>

namespace cobra {
namespace knn {
namespace {

using Wide = long double;

struct Candidate {
    int index;
    Wide distance;
};

// The order is deliberately independent of the order in which grid cells
// are visited.  A query vertex wins an exact distance tie with another vertex
// (self-first), then the smaller vertex ID wins all remaining ties.
bool IsBetter(const Candidate& a, const Candidate& b, int query_index) {
    if (a.distance < b.distance) return true;
    if (b.distance < a.distance) return false;
    const bool a_is_self = a.index == query_index;
    const bool b_is_self = b.index == query_index;
    if (a_is_self != b_is_self) return a_is_self;
    return a.index < b.index;
}

struct WorseFirst {
    int query_index;

    // std::priority_queue puts the element for which Compare returns false
    // against every other element at the top.  Returning true when a is
    // better than b therefore makes top() the current worst member of the
    // bounded top-k set (the same convention as std::less for a max heap).
    bool operator()(const Candidate& a, const Candidate& b) const {
        return IsBetter(a, b, query_index);
    }
};

class CandidateHeap : public std::priority_queue<Candidate,
                                                   std::vector<Candidate>,
                                                   WorseFirst> {
    using Base = std::priority_queue<Candidate, std::vector<Candidate>, WorseFirst>;

public:
    using Base::Base;

    void exchange_storage(std::vector<Candidate>& reusable) {
        this->c.swap(reusable);
    }
};

Wide SquaredDistance(const Point2D& a, const Point2D& b) {
    const Wide dx = static_cast<Wide>(a.x) - static_cast<Wide>(b.x);
    const Wide dy = static_cast<Wide>(a.y) - static_cast<Wide>(b.y);
    return dx * dx + dy * dy;
}

void ValidatePoints(const std::vector<Point2D>& points) {
    // Neighbor IDs are part of the public API as int, matching FILO2's
    // existing KDTree output.  Refuse an input that cannot be represented.
    if (points.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::length_error("GridKnn supports at most INT_MAX points");
    }
    for (const Point2D& point : points) {
        if (!std::isfinite(point.x) || !std::isfinite(point.y)) {
            throw std::invalid_argument("GridKnn requires finite coordinates");
        }
    }
}

void ValidateCoordinateArrays(const std::vector<double>& xcoords,
                              const std::vector<double>& ycoords) {
    if (xcoords.size() != ycoords.size()) {
        throw std::invalid_argument("x and y coordinate arrays must have equal size");
    }
}

struct Bounds {
    Wide min_x = 0;
    Wide max_x = 0;
    Wide min_y = 0;
    Wide max_y = 0;
};

// Uniform-grid index and storage.  Cells are retained as vectors rather than
// sorted point ranges: this keeps duplicate coordinates and vertex IDs intact
// and makes the CPU block boundaries directly traceable in the CUDA port.
class UniformGrid {
public:
    explicit UniformGrid(const std::vector<Point2D>& points) : points_(points) {
        // [CPU-1] Grid setup/binning.  The target grid has about sqrt(N)
        // cells per axis.  The cell-size heuristic is a reasoned choice, not
        // empirically verified here — benchmark this first on FILO2's data;
        // it bounds the number of allocated cells while making a roughly
        // uniform point cloud put O(1) points in a query's early rings.
        if (points_.empty()) return;

        bounds_.min_x = bounds_.max_x = static_cast<Wide>(points_[0].x);
        bounds_.min_y = bounds_.max_y = static_cast<Wide>(points_[0].y);
        for (const Point2D& point : points_) {
            const Wide x = static_cast<Wide>(point.x);
            const Wide y = static_cast<Wide>(point.y);
            bounds_.min_x = std::min(bounds_.min_x, x);
            bounds_.max_x = std::max(bounds_.max_x, x);
            bounds_.min_y = std::min(bounds_.min_y, y);
            bounds_.max_y = std::max(bounds_.max_y, y);
        }

        const Wide span_x = bounds_.max_x - bounds_.min_x;
        const Wide span_y = bounds_.max_y - bounds_.min_y;
        const Wide span = std::max(span_x, span_y);
        const Wide root = std::sqrt(static_cast<Wide>(points_.size()));
        target_axis_ = static_cast<std::size_t>(std::ceil(root));
        if (target_axis_ == 0) target_axis_ = 1;

        // A zero-span axis gets one cell.  For a nonzero span, max-span /
        // target-axis makes the number of cells on each axis no greater than
        // target-axis (the +1 is handled by the final-cell clamp below).
        if (span > 0) {
            cell_size_ = span / static_cast<Wide>(target_axis_);
            if (!(cell_size_ > 0) || !std::isfinite(cell_size_)) {
                throw std::length_error("GridKnn could not represent grid cell size");
            }
        } else {
            cell_size_ = 1;
        }

        nx_ = AxisCellCount(span_x);
        ny_ = AxisCellCount(span_y);
        if (nx_ > std::numeric_limits<std::size_t>::max() / ny_) {
            throw std::length_error("GridKnn grid cell count overflows size_t");
        }
        const std::size_t total_cells = nx_ * ny_;
        if (total_cells > std::numeric_limits<std::size_t>::max() - 1) {
            throw std::length_error("GridKnn grid cell count overflows size_t");
        }
        offsets_.assign(total_cells + 1, 0);
        std::vector<std::size_t> cell_ids(points_.size());
        for (std::size_t i = 0; i < points_.size(); ++i) {
            const std::size_t cx = CellX(points_[i].x);
            const std::size_t cy = CellY(points_[i].y);
            const std::size_t cell = FlatIndex(cx, cy);
            cell_ids[i] = cell;
            ++offsets_[cell + 1];
        }
        for (std::size_t cell = 1; cell < offsets_.size(); ++cell) {
            offsets_[cell] += offsets_[cell - 1];
        }
        point_ids_.resize(points_.size());
        std::vector<std::size_t> write_positions(offsets_.begin(), offsets_.end() - 1);
        for (std::size_t i = 0; i < points_.size(); ++i) {
            const std::size_t cell = cell_ids[i];
            point_ids_[write_positions[cell]++] = static_cast<int>(i);
        }
    }

    std::size_t max_ring(std::size_t cx, std::size_t cy) const {
        if (nx_ == 0 || ny_ == 0) return 0;
        return std::max(std::max(cx, nx_ - 1 - cx),
                        std::max(cy, ny_ - 1 - cy));
    }

    std::size_t nx() const { return nx_; }
    std::size_t ny() const { return ny_; }
    const Bounds& bounds() const { return bounds_; }
    Wide cell_size() const { return cell_size_; }

    std::size_t cell_begin(std::size_t cx, std::size_t cy) const {
        return offsets_[FlatIndex(cx, cy)];
    }

    std::size_t cell_end(std::size_t cx, std::size_t cy) const {
        return offsets_[FlatIndex(cx, cy) + 1];
    }

    int point_id(std::size_t position) const {
        return point_ids_[position];
    }

    std::size_t CellX(double x) const {
        return CellCoordinate(static_cast<Wide>(x), bounds_.min_x, nx_);
    }

    std::size_t CellY(double y) const {
        return CellCoordinate(static_cast<Wide>(y), bounds_.min_y, ny_);
    }

    // Visit each cell in one Chebyshev ring exactly once.  The callback gets
    // only in-range cells, so an edge query naturally has fewer cells in a
    // ring without any special cases in the caller.
    template <typename Callback>
    bool ForEachRing(std::size_t cx, std::size_t cy, std::size_t radius,
                     Callback&& callback) const {
        if (nx_ == 0 || ny_ == 0) return false;

        const std::int64_t center_x = static_cast<std::int64_t>(cx);
        const std::int64_t center_y = static_cast<std::int64_t>(cy);
        const std::int64_t r = static_cast<std::int64_t>(radius);
        bool visited = false;

        auto visit = [&](std::int64_t x, std::int64_t y) {
            if (x < 0 || y < 0 || x >= static_cast<std::int64_t>(nx_) ||
                y >= static_cast<std::int64_t>(ny_)) {
                return;
            }
            visited = true;
            callback(static_cast<std::size_t>(x), static_cast<std::size_t>(y));
        };

        if (radius == 0) {
            visit(center_x, center_y);
            return visited;
        }

        // Top and bottom edges.
        for (std::int64_t x = center_x - r; x <= center_x + r; ++x) {
            visit(x, center_y - r);
            visit(x, center_y + r);
        }
        // Left and right edges, excluding corners already visited above.
        for (std::int64_t y = center_y - r + 1; y <= center_y + r - 1; ++y) {
            visit(center_x - r, y);
            visit(center_x + r, y);
        }
        return visited;
    }

    // Distance from a point to the closed bounding box of a cell.  Every
    // point assigned to that cell is in this box, so this is a safe lower
    // bound for the exact point distance.
    Wide CellLowerBoundSquared(const Point2D& query,
                               std::size_t cx, std::size_t cy) const {
        const Wide x0 = bounds_.min_x + static_cast<Wide>(cx) * cell_size_;
        const Wide y0 = bounds_.min_y + static_cast<Wide>(cy) * cell_size_;
        const Wide x1 = (cx + 1 >= nx_) ? bounds_.max_x
                                        : bounds_.min_x + static_cast<Wide>(cx + 1) * cell_size_;
        const Wide y1 = (cy + 1 >= ny_) ? bounds_.max_y
                                        : bounds_.min_y + static_cast<Wide>(cy + 1) * cell_size_;

        const Wide qx = static_cast<Wide>(query.x);
        const Wide qy = static_cast<Wide>(query.y);
        Wide dx = 0;
        Wide dy = 0;
        if (qx < x0) dx = x0 - qx;
        else if (qx > x1) dx = qx - x1;
        if (qy < y0) dy = y0 - qy;
        else if (qy > y1) dy = qy - y1;
        // Keep the bound conservative across the rounded cell-boundary
        // arithmetic. Equality must remain searchable because the final
        // vertex-ID tie-break is part of the exact result.
        return std::nextafter(dx * dx + dy * dy, static_cast<Wide>(0));
    }

private:
    std::size_t AxisCellCount(Wide span) const {
        if (!(span > 0)) return 1;
        // Floating-point roundoff at max-span can produce target-axis + 1.
        // Clamp to preserve the allocation bound and let CellCoordinate clamp
        // the endpoint into the final cell.
        const Wide ratio = span / cell_size_;
        const Wide raw = std::floor(ratio) + 1;
        std::size_t result = raw >= static_cast<Wide>(target_axis_)
                                 ? target_axis_
                                 : static_cast<std::size_t>(raw);
        return std::max<std::size_t>(1, result);
    }

    std::size_t CellCoordinate(Wide coordinate, Wide minimum,
                               std::size_t axis_count) const {
        if (axis_count <= 1 || cell_size_ <= 0) return 0;
        const Wide raw = (coordinate - minimum) / cell_size_;
        if (!(raw > 0)) return 0;
        if (raw >= static_cast<Wide>(axis_count)) return axis_count - 1;
        const std::size_t result = static_cast<std::size_t>(raw);
        return std::min(result, axis_count - 1);
    }

    std::size_t FlatIndex(std::size_t cx, std::size_t cy) const {
        return cy * nx_ + cx;
    }

    const std::vector<Point2D>& points_;
    Bounds bounds_;
    Wide cell_size_ = 1;
    std::size_t target_axis_ = 0;
    std::size_t nx_ = 0;
    std::size_t ny_ = 0;
    std::vector<int> point_ids_;
    std::vector<std::size_t> offsets_;
};

NeighborLists MakeCoordinatePoints(const std::vector<double>& xcoords,
                                   const std::vector<double>& ycoords,
                                   int k,
                                   bool grid) {
    ValidateCoordinateArrays(xcoords, ycoords);
    std::vector<Point2D> points;
    points.reserve(xcoords.size());
    for (std::size_t i = 0; i < xcoords.size(); ++i) {
        points.push_back({xcoords[i], ycoords[i]});
    }
    return grid ? GridKnn(points, k) : BruteForceKnn(points, k);
}

}  // namespace

int EffectiveK(std::size_t point_count, int k) {
    if (k < 0) throw std::invalid_argument("k must be non-negative");
    if (point_count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::length_error("GridKnn supports at most INT_MAX points");
    }
    return std::min(k, static_cast<int>(point_count));
}

NeighborLists BruteForceKnn(const std::vector<Point2D>& points, int k) {
    ValidatePoints(points);
    const int effective_k = EffectiveK(points.size(), k);
    NeighborLists result(points.size());
    if (effective_k == 0) return result;

    for (std::size_t query = 0; query < points.size(); ++query) {
        std::vector<Candidate> candidates;
        candidates.reserve(points.size());
        for (std::size_t candidate = 0; candidate < points.size(); ++candidate) {
            candidates.push_back({static_cast<int>(candidate),
                                  SquaredDistance(points[query], points[candidate])});
        }
        std::sort(candidates.begin(), candidates.end(),
                  [query](const Candidate& a, const Candidate& b) {
                      return IsBetter(a, b, static_cast<int>(query));
                  });
        result[query].reserve(static_cast<std::size_t>(effective_k));
        for (int rank = 0; rank < effective_k; ++rank) {
            result[query].push_back(candidates[static_cast<std::size_t>(rank)].index);
        }
    }
    return result;
}

NeighborLists BruteForceKnn(const std::vector<double>& xcoords,
                            const std::vector<double>& ycoords, int k) {
    return MakeCoordinatePoints(xcoords, ycoords, k, false);
}

NeighborLists GridKnn(const std::vector<Point2D>& points, int k) {
    ValidatePoints(points);
    const int effective_k = EffectiveK(points.size(), k);
    NeighborLists result(points.size());
    if (effective_k == 0 || points.empty()) return result;

    UniformGrid grid(points);
    // Queries are independent, but each bounded heap has the same maximum
    // size. Exchange one reusable backing vector instead of allocating and
    // freeing a k-sized heap buffer for every point in the fallback path.
    std::vector<Candidate> reusable_heap_storage;
    reusable_heap_storage.reserve(static_cast<std::size_t>(effective_k));
    std::vector<Candidate> ordered;
    ordered.reserve(static_cast<std::size_t>(effective_k));

    for (std::size_t query_index = 0; query_index < points.size(); ++query_index) {
        // [CPU-2] Query/top-k initialization.  The heap is bounded throughout
        // the search; once full, top() is the exact current worst candidate.
        const int query_id = static_cast<int>(query_index);
        CandidateHeap heap(WorseFirst{query_id});
        heap.exchange_storage(reusable_heap_storage);
        const std::size_t query_cx = grid.CellX(points[query_index].x);
        const std::size_t query_cy = grid.CellY(points[query_index].y);
        const std::size_t final_ring = grid.max_ring(query_cx, query_cy);

        std::size_t radius = 0;
        for (;;) {
            // [CPU-3] Ring enumeration.  Every point is visited exactly once
            // because every nonempty cell has one Chebyshev ring number.
            grid.ForEachRing(query_cx, query_cy, radius,
                             [&](std::size_t cx, std::size_t cy) {
                                 for (std::size_t position = grid.cell_begin(cx, cy);
                                      position < grid.cell_end(cx, cy); ++position) {
                                     const int candidate_id = grid.point_id(position);
                                     // [CPU-4] Bounded candidate update.
                                     const Candidate candidate{
                                         candidate_id,
                                         SquaredDistance(points[query_index],
                                                         points[static_cast<std::size_t>(candidate_id)])};
                                     if (heap.size() < static_cast<std::size_t>(effective_k)) {
                                         heap.push(candidate);
                                     } else if (IsBetter(candidate, heap.top(), query_id)) {
                                         heap.pop();
                                         heap.push(candidate);
                                     }
                                 }
                             });

            // [CPU-5] Exact lower-bound stop.  If another ring exists, the
            // nearest possible point in all unsearched cells is attained by
            // one of the next-ring cell bounding boxes: moving farther in
            // either grid axis cannot decrease its point-to-box distance.
            // Strict inequality is required; equality remains searchable so
            // vertex-ID tie ordering cannot be lost.  If there is no next
            // ring, every cell has been enumerated and termination is exact.
            if (radius >= final_ring) break;  // explicit all-cells termination

            const std::size_t next_radius = radius + 1;
            Wide lower_bound = std::numeric_limits<Wide>::infinity();
            grid.ForEachRing(query_cx, query_cy, next_radius,
                             [&](std::size_t cx, std::size_t cy) {
                                 lower_bound = std::min(
                                     lower_bound,
                                     grid.CellLowerBoundSquared(points[query_index], cx, cy));
                             });

            if (heap.size() == static_cast<std::size_t>(effective_k) &&
                heap.top().distance < lower_bound) {
                break;
            }
            radius = next_radius;
        }

        // [CPU-6] Ordered output.  priority_queue order is intentionally not
        // exposed: sorting establishes the same deterministic order as the
        // brute-force reference regardless of ring/cell insertion order.
        ordered.clear();
        ordered.reserve(heap.size());
        while (!heap.empty()) {
            ordered.push_back(heap.top());
            heap.pop();
        }
        std::sort(ordered.begin(), ordered.end(),
                  [query_id](const Candidate& a, const Candidate& b) {
                      return IsBetter(a, b, query_id);
                  });
        result[query_index].reserve(ordered.size());
        for (const Candidate& candidate : ordered) {
            result[query_index].push_back(candidate.index);
        }

        // The heap is empty after the drain; return its capacity to the
        // reusable vector before the next query constructs its comparator.
        heap.exchange_storage(reusable_heap_storage);
    }
    return result;
}

NeighborLists GridKnn(const std::vector<double>& xcoords,
                      const std::vector<double>& ycoords, int k) {
    return MakeCoordinatePoints(xcoords, ycoords, k, true);
}

}  // namespace knn
}  // namespace cobra
