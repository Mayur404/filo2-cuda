#ifndef _FILO2_BPP_HPP_
#define _FILO2_BPP_HPP_

#include <algorithm>
#include <vector>

#include "../instance/Instance.hpp"

namespace bpp {

    // Fast O(N log K) greedy first-fit decreasing bin packing using a segment tree.
    inline int greedy_first_fit_decreasing(const cobra::Instance& instance) {
        const int n = instance.get_customers_num();
        if (n == 0) return 0;

        std::vector<int> customers(n);
        for (int i = 0; i < n; ++i) {
            customers[i] = instance.get_customers_begin() + i;
        }

        std::sort(customers.begin(), customers.end(),
                  [&instance](int i, int j) { return instance.get_demand(i) > instance.get_demand(j); });

        const int capacity = instance.get_vehicle_capacity();

        // Build a segment tree over at most n bins, tracking max remaining capacity in each subtree.
        int tree_size = 1;
        while (tree_size < n) {
            tree_size <<= 1;
        }

        std::vector<int> tree(2 * tree_size, 0);
        for (int i = 0; i < n; ++i) {
            tree[tree_size + i] = capacity;
        }
        for (int i = tree_size - 1; i > 0; --i) {
            tree[i] = std::max(tree[2 * i], tree[2 * i + 1]);
        }

        int used_bins = 0;
        for (int c : customers) {
            const int demand = instance.get_demand(c);
            int node = 1;
            int l = 0;
            int r = tree_size - 1;

            while (l < r) {
                const int mid = (l + r) / 2;
                if (tree[2 * node] >= demand) {
                    node = 2 * node;
                    r = mid;
                } else {
                    node = 2 * node + 1;
                    l = mid + 1;
                }
            }

            tree[node] -= demand;
            const int bin_idx = l;
            if (bin_idx + 1 > used_bins) {
                used_bins = bin_idx + 1;
            }

            // Push update back up to the root.
            node >>= 1;
            while (node > 0) {
                tree[node] = std::max(tree[2 * node], tree[2 * node + 1]);
                node >>= 1;
            }
        }

        return used_bins;
    }

}  // namespace bpp

#endif