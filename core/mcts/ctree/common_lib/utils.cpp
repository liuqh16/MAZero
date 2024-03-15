#include "utils.h"
#include <iostream>
#include <cmath>

namespace tools
{

    void my_assert(bool flag, const char *msg = nullptr)
    {
        /*
        WARNING: never use `assert` here, since an assertion is not an exception and cannot be caught.
                use `my_assert` instead.
        */
        if (!flag)
        {
            throw std::runtime_error(msg);
        }
    }

    void SubTreeValueSet::update(float key, int depth){
        if ((int)(count.size()) <= depth){
            big.push_back(std::multiset<float>());
            small.push_back(std::multiset<float>());
            count.push_back(0);
            if(depth == 0) lam_pow.push_back(1.);
                else lam_pow.push_back(lam_pow[depth-1] * lambda);
        }
        count[depth] += 1;

        int cur_size = big[depth].size();
        int size_lim = std::max(1, (int)(ceil(count[depth] * (1-this->quantile))));

        if(cur_size == size_lim){
            auto it = big[depth].begin();
            if(key < *it){
                small[depth].insert(key);
            }else{
                small[depth].insert(*it);

                weighted_sum -= lam_pow[depth] * (*it);
                tot_weight -= lam_pow[depth];
                big[depth].erase(it);

                big[depth].insert(key);
                tot_weight += lam_pow[depth];
                weighted_sum += lam_pow[depth] * key;
            }
        }else{
            my_assert(cur_size + 1 == size_lim, "SubTreeValueSet::update: cur_size+1!=size_lim.");

            if(small[depth].begin() == small[depth].end()){
                big[depth].insert(key);
                tot_weight += lam_pow[depth];
                weighted_sum += lam_pow[depth] * key;
            }else{
                auto it = small[depth].end();
                it--;
                if(key > *it){
                    big[depth].insert(key);
                    tot_weight += lam_pow[depth];
                    weighted_sum += lam_pow[depth] * key;
                }else{
                    big[depth].insert(*it);
                    tot_weight += lam_pow[depth];
                    weighted_sum += lam_pow[depth] * (*it);
                    small[depth].erase(it);
                    small[depth].insert(key);
                }
            }
        }
    }

    float SubTreeValueSet::value_estimation(){
        int d = big.size();
        my_assert(d > 0, "SubTreeValueSet::value_estimation: empty set.");
        return weighted_sum / tot_weight;
    }

    CMinMaxStats::CMinMaxStats(float delta_lb) : delta_lb(delta_lb) {}

    CMinMaxStats::~CMinMaxStats() {}

    void CMinMaxStats::remove(float old_val)
    {
        auto it = se.find(old_val);
        my_assert(it != se.end(), "CMinMaxStats::remove: value not found");
        se.erase(it);
    }

    void CMinMaxStats::insert(float new_val)
    {
        se.insert(new_val);
    }

    float CMinMaxStats::normalize(float value)
    {
        if (se.begin() == se.end())
            return value;
        float maximum = *se.rbegin();
        float minimum = *se.begin();
        float delta = maximum - minimum;
        return (value - minimum) / std::max(delta_lb, delta);
    }

    // Array indexing

    template <typename T>
    T &Array2D<T>::operator()(size_t i, size_t j)
    {
        return this->p[i * d2 + j];
    }
    template <typename T>
    T &Array3D<T>::operator()(size_t i, size_t j, size_t k)
    {
        return this->p[(i * d2 + j) * d3 + k];
    }
}