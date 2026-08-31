#include "../base/KDTree.hpp"
#include "../knn/GridKnn.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {

double squared_dist(double x1, double y1, double x2, double y2) {
    double dx = x1 - x2;
    double dy = y1 - y2;
    return dx * dx + dy * dy;
}

void compare_kdtree_vs_gridknn(const std::string& test_name,
                               const std::vector<double>& xs,
                               const std::vector<double>& ys,
                               int k) {
    const int n = static_cast<int>(xs.size());
    std::cout << "==================================================\n";
    std::cout << "TEST: " << test_name << " (N=" << n << ", k=" << k << ")\n";
    std::cout << "==================================================\n";

    // 1. Run base/KDTree.cpp exactly as done in Instance.cpp
    cobra::KDTree kd_tree(xs, ys);
    std::vector<std::vector<int>> kdtree_out(n);
    for (int i = 0; i < n; ++i) {
        kdtree_out[i] = kd_tree.GetNearestNeighbors(xs[i], ys[i], k);
        // Apply FILO2 Instance.cpp self-first rule
        if (kdtree_out[i][0] != i) {
            int p = 1;
            while (p < static_cast<int>(kdtree_out[i].size())) {
                if (kdtree_out[i][p] == i) break;
                p++;
            }
            if (p < static_cast<int>(kdtree_out[i].size())) {
                std::swap(kdtree_out[i][0], kdtree_out[i][p]);
            }
        }
    }

    // 2. Run knn/GridKnn.cpp (CPU) directly
    std::vector<std::vector<int>> gridknn_out = cobra::knn::GridKnn(xs, ys, k);

    // 3. Direct head-to-head comparison
    int disagreements = 0;
    std::vector<int> mismatched_queries;

    for (int i = 0; i < n; ++i) {
        if (kdtree_out[i] != gridknn_out[i]) {
            disagreements++;
            mismatched_queries.push_back(i);
        }
    }

    if (disagreements == 0) {
        std::cout << n << " points tested, 0 disagreements, outputs identical\n\n";
    } else {
        std::cout << disagreements << " disagreements found (out of " << n << " points tested)\n";
        std::cout << "First " << std::min<int>(3, static_cast<int>(mismatched_queries.size()))
                  << " mismatch details:\n";

        for (int m = 0; m < std::min<int>(3, static_cast<int>(mismatched_queries.size())); ++m) {
            int q = mismatched_queries[m];
            std::cout << "\n--- Query Point " << q << " at (" << xs[q] << ", " << ys[q] << ") ---\n";
            std::cout << std::left << std::setw(8) << "Rank"
                      << std::setw(15) << "KDTree ID"
                      << std::setw(15) << "KDTree DistSq"
                      << std::setw(15) << "GridKnn ID"
                      << std::setw(15) << "GridKnn DistSq"
                      << "Match?\n";
            std::cout << std::string(75, '-') << "\n";

            for (int r = 0; r < k; ++r) {
                int kid = kdtree_out[q][r];
                int gid = gridknn_out[q][r];
                double kdist = squared_dist(xs[q], ys[q], xs[kid], ys[kid]);
                double gdist = squared_dist(xs[q], ys[q], xs[gid], ys[gid]);

                std::cout << std::left << std::setw(8) << r
                          << std::setw(15) << kid
                          << std::setw(15) << kdist
                          << std::setw(15) << gid
                          << std::setw(15) << gdist
                          << (kid == gid ? "YES" : "MISMATCH") << "\n";
            }
        }
        std::cout << "\n";
    }
}

}  // namespace

int main() {
    // Case 1: Continuous Random 2D coordinates without ties (N=1000, k=25)
    {
        std::mt19937_64 rng(42);
        std::uniform_real_distribution<double> dist(-1000.0, 1000.0);
        int n = 1000;
        std::vector<double> xs(n), ys(n);
        for (int i = 0; i < n; ++i) {
            xs[i] = dist(rng);
            ys[i] = dist(rng);
        }
        compare_kdtree_vs_gridknn("Random Non-Tied Point Set", xs, ys, 25);
    }

    // Case 2: Symmetric 2D Grid Pattern Designed Specifically to Contain Tied Distances (5x5 Grid, k=8)
    // Points at integer coordinates (x, y) where x in {0..4}, y in {0..4}
    // Query at center (2, 2) has 4 points at distance 1.0 (ties), 4 points at distance 2.0 (ties), etc.
    {
        int grid_dim = 5;
        std::vector<double> xs, ys;
        for (int y = 0; y < grid_dim; ++y) {
            for (int x = 0; x < grid_dim; ++x) {
                xs.push_back(static_cast<double>(x));
                ys.push_back(static_cast<double>(y));
            }
        }
        compare_kdtree_vs_gridknn("Symmetric 5x5 Grid Pattern (Heavy Ties)", xs, ys, 8);
    }

    // Case 3: 10x10 Symmetric Grid Pattern (N=100, k=16)
    {
        int grid_dim = 10;
        std::vector<double> xs, ys;
        for (int y = 0; y < grid_dim; ++y) {
            for (int x = 0; x < grid_dim; ++x) {
                xs.push_back(static_cast<double>(x));
                ys.push_back(static_cast<double>(y));
            }
        }
        compare_kdtree_vs_gridknn("Symmetric 10x10 Grid Pattern (Heavy Ties)", xs, ys, 16);
    }

    return 0;
}
