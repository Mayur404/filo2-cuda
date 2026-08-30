#include "../knn/GridKnn.hpp"

#include <cmath>
#include <cstddef>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using cobra::knn::BruteForceKnn;
using cobra::knn::GridKnn;
using cobra::knn::NeighborLists;
using cobra::knn::Point2D;

std::vector<Point2D> RandomPoints(std::size_t count, unsigned seed) {
    std::mt19937 generator(seed);
    std::uniform_real_distribution<double> distribution(-1000.0, 1000.0);
    std::vector<Point2D> points;
    points.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        points.push_back({distribution(generator), distribution(generator)});
    }
    return points;
}

void RequireEqual(const std::string& name, const std::vector<Point2D>& points,
                  int k) {
    const NeighborLists expected = BruteForceKnn(points, k);
    const NeighborLists actual = GridKnn(points, k);
    if (actual != expected) {
        std::cerr << "FAIL " << name << ": grid result differs from brute force\n";
        for (std::size_t i = 0; i < actual.size(); ++i) {
            if (actual[i] != expected[i]) {
                std::cerr << "  first mismatch at query " << i << "\n"
                          << "  expected:";
                for (int id : expected[i]) std::cerr << ' ' << id;
                std::cerr << "\n  actual:  ";
                for (int id : actual[i]) std::cerr << ' ' << id;
                std::cerr << '\n';
                break;
            }
        }
        throw std::runtime_error("CPU grid k-NN mismatch");
    }
    std::cout << "PASS " << std::left << std::setw(34) << name
              << " N=" << points.size() << " k=" << k << '\n';
}

void RequireValidationFailures() {
    bool caught = false;
    try {
        GridKnn(std::vector<Point2D>{{0.0, 0.0}}, -1);
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    if (!caught) throw std::runtime_error("negative k was not rejected");

    caught = false;
    try {
        GridKnn(std::vector<Point2D>{{std::numeric_limits<double>::quiet_NaN(), 0.0}}, 1);
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    if (!caught) throw std::runtime_error("non-finite coordinate was not rejected");

    caught = false;
    try {
        GridKnn(std::vector<double>{0.0}, std::vector<double>{}, 1);
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    if (!caught) throw std::runtime_error("mismatched coordinate arrays were not rejected");

    RequireEqual("empty input and k=0", {}, 0);
    RequireEqual("k=0 nonempty input", RandomPoints(17, 101), 0);
    std::cout << "PASS validation and effective-k checks\n";
}

std::vector<Point2D> DuplicatePoints() {
    std::vector<Point2D> points;
    points.reserve(512);
    // Many exact duplicates deliberately exercise the equality path in both
    // the bounded heap and the strict lower-bound stopping rule.
    for (int i = 0; i < 512; ++i) {
        const int group = i % 8;
        points.push_back({static_cast<double>(group % 4),
                          static_cast<double>(group / 4)});
    }
    return points;
}

std::vector<Point2D> BoundaryAndOutlierPoints() {
    std::vector<Point2D> points;
    points.reserve(240);
    for (int i = -100; i <= 100; ++i) {
        points.push_back({static_cast<double>(i), 0.0});
    }
    points.push_back({-1.0e9, 1.0e9});
    points.push_back({1.0e9, -1.0e9});
    points.push_back({0.0, 1.0e9});
    points.push_back({0.0, -1.0e9});
    return points;
}

}  // namespace

int main() {
    try {
        RequireEqual("random N=100", RandomPoints(100, 1), 17);
        RequireEqual("random N=1000", RandomPoints(1000, 2), 31);
        RequireEqual("random N=10000", RandomPoints(10000, 3), 47);
        RequireEqual("duplicate points", DuplicatePoints(), 100);
        RequireEqual("N <= k", RandomPoints(9, 4), 25);
        RequireEqual("boundary collinear outliers", BoundaryAndOutlierPoints(), 13);

        // The coordinate-array overload is the shape consumed by FILO2's
        // Instance and by the later GPU comparison harness.
        const std::vector<Point2D> coordinate_points = RandomPoints(127, 5);
        std::vector<double> xcoords;
        std::vector<double> ycoords;
        xcoords.reserve(coordinate_points.size());
        ycoords.reserve(coordinate_points.size());
        for (const Point2D& point : coordinate_points) {
            xcoords.push_back(point.x);
            ycoords.push_back(point.y);
        }
        if (GridKnn(xcoords, ycoords, 23) != GridKnn(coordinate_points, 23) ||
            BruteForceKnn(xcoords, ycoords, 23) != BruteForceKnn(coordinate_points, 23)) {
            throw std::runtime_error("coordinate-array overload mismatch");
        }
        std::cout << "PASS coordinate-array overload\n";

        RequireValidationFailures();
        std::cout << "ALL CPU GRID KNN TESTS PASSED\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "CPU GRID KNN TEST FAILURE: " << error.what() << '\n';
        return 1;
    }
}
