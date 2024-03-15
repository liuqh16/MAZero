#ifndef UTILS_H
#define UTILS_H
// Avoid a gcc warning below:
// warning: #warning "Using deprecated NumPy API, disable it with " "#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <iostream>
#include <vector>
#include <set>

const float FLOAT_MAX = 1000000.0;
const float FLOAT_MIN = -FLOAT_MAX;

namespace tools
{
    void my_assert(bool flag, const char *msg);

    struct SubTreeValueSet{
        /*
        res = Mean{ lambda^d * x | d, x in big[d] }

        big[d] = MultiSet { val | val above quantile at depth d }
        small[d] = MultiSet { val | val below quantile at depth d }
        count[d] = # vals at depth d = |big[d]| + |small[d]|

        weighted_sum = Sum{ lambda^d * x | d, x in big[d] }
        tot_weight = Sum{ lambda^d * big[d].size() }
        */
        float weighted_sum, tot_weight, quantile, lambda;

        std::vector<std::multiset<float>> big;
        std::vector<std::multiset<float>> small;
        std::vector<int> count;
        std::vector<float> lam_pow;

        SubTreeValueSet(float rho, float lam):weighted_sum(0.), tot_weight(0.), quantile(rho), lambda(lam){}

        void update(float key, int depth);
        float value_estimation();
    };

    struct CMinMaxStats
    {
        std::multiset<float> se;
        float delta_lb;

        CMinMaxStats(float delta_lb);
        ~CMinMaxStats();

        void remove(float old_val);
        void insert(float new_val);
        float normalize(float value);
    };

    template <typename T>
    class Array2D
    {
        // no bound check
    public:
        T *p;
        size_t d1, d2;

        Array2D() {}
        Array2D(T *p, const size_t &d1, const size_t &d2) : p(p), d1(d1), d2(d2) {}
        T &operator()(size_t i, size_t j);
    };

    template <typename T>
    class Array3D
    {
        // no bound check
    public:
        T *p;
        size_t d1, d2, d3;

        Array3D() {}
        Array3D(T *p, const size_t &d1, const size_t &d2, const size_t &d3) : p(p), d1(d1), d2(d2), d3(d3) {}
        T &operator()(size_t i, size_t j, size_t k);
    };
}

#endif