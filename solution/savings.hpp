#ifndef _FILO2_SOLUTIONALGORITHMS_HPP_
#define _FILO2_SOLUTIONALGORITHMS_HPP_

#include "../base/Timer.hpp"
#include "Solution.hpp"

namespace cobra {

    // Limited savings algorithm.
    inline void clarke_and_wright(const Instance &instance, Solution &solution, const double lambda, int neighbors_num) {

        solution.reset();

        for (auto i = instance.get_customers_begin(); i < instance.get_customers_end(); i++) {
            solution.build_one_customer_route</*record_acion=*/false>(i);
        }
        assert(solution.is_feasible());

        neighbors_num = std::min(instance.get_customers_num() - 1, neighbors_num);

        struct Saving {
            int i;
            int j;
            double value;
        };

        auto savings = std::vector<Saving>();

        const int depot = instance.get_depot();
        std::vector<double> depot_costs(instance.get_vertices_num());
        for (int i = instance.get_customers_begin(); i < instance.get_customers_end(); ++i) {
            depot_costs[i] = instance.get_cost(i, depot);
        }

        // Each customer owns a disjoint output range. Count first so workers
        // can fill those ranges without locks or concurrent vector growth.
        const int first = instance.get_customers_begin();
        const int last = instance.get_customers_end();
        std::vector<std::size_t> offsets(static_cast<std::size_t>(last) + 1, 0);
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) if(last - first >= 10000)
#endif
        for (int i = first; i < last; ++i) {
            const auto &ineighbors = instance.get_neighbors_of(i);
            unsigned int added = 0;
            for (std::size_t n = 1; added < static_cast<unsigned int>(neighbors_num) && n < ineighbors.size(); ++n) {
                added += i < ineighbors[n];
            }
            offsets[i + 1] = added;
        }
        for (int i = first; i < last; ++i) {
            offsets[i + 1] += offsets[i];
        }
        savings.resize(offsets[last]);
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) if(last - first >= 10000)
#endif
        for (int i = first; i < last; ++i) {
            const auto &ineighbors = instance.get_neighbors_of(i);
            std::size_t position = offsets[i];
            for (std::size_t n = 1; position < offsets[i + 1] && n < ineighbors.size(); ++n) {
                const int j = ineighbors[n];
                if (i < j) {
                    const double value = depot_costs[i] + depot_costs[j] - lambda * instance.get_cost(i, j);
                    savings[position++] = {i, j, value};
                }
            }
        }

        // Keep the original FILO2 comparison and std::sort ordering. Tied
        // savings can lead to different route merges if a parallel sort
        // permutes them, even though their numeric values are equal.
        std::sort(savings.begin(), savings.end(),
                  [](const Saving &a, const Saving &b) { return a.value > b.value; });

#ifdef VERBOSE
        Timer timer;
#endif

        for (std::size_t n = 0; n < savings.size(); ++n) {

            const auto &saving = savings[n];

            const auto i = saving.i;
            const auto j = saving.j;

            const auto iRoute = solution.get_route_index(i);
            const auto jRoute = solution.get_route_index(j);

            if (iRoute == jRoute) {
                continue;
            }

            if (solution.get_last_customer(iRoute) == i && solution.get_first_customer(jRoute) == j &&
                solution.get_route_load(iRoute) + solution.get_route_load(jRoute) <= instance.get_vehicle_capacity()) {

                solution.append_route(iRoute, jRoute);


            } else if (solution.get_last_customer(jRoute) == j && solution.get_first_customer(iRoute) == i &&
                       solution.get_route_load(iRoute) + solution.get_route_load(jRoute) <= instance.get_vehicle_capacity()) {

                solution.append_route(jRoute, iRoute);
            }

#ifdef VERBOSE
            if (timer.elapsed_time<std::chrono::seconds>() > 2) {
                std::cout << "Progress: " << 100.0 * (n + 1) / savings.size() << "%, Solution cost: " << solution.get_cost() << " \n";
                timer.reset();
            }
#endif
        }
        assert(solution.is_feasible());
    }

}  // namespace cobra

#endif
