#ifndef dnntool_Utils_hpp
#define dnntool_Utils_hpp

#include <cmath>
#include <vector>

namespace dnntool
{

namespace utils {

template <typename T>
struct IndexComparator {
    IndexComparator(std::vector<T> vec) : vec_(vec) {}

    bool operator() (size_t i, size_t j) { return vec_[i]<vec_[j]; }

private:
    std::vector<T> vec_;
};

template <typename T>
void find_max(std::vector<T> vals, T& max_val, size_t& idx)
{
    std::vector<size_t> indices(vals.size());
    for (size_t x = 0; x < indices.size(); x++) indices[x]=x;
    std::sort(indices.begin(), indices.end(), IndexComparator<T>(vals));
    std::reverse(indices.begin(), indices.end());
    max_val = vals[indices[0]];
    idx = indices[0];
}

inline double rad2deg( double rad )
{
    return rad / M_PI * 180.0;
}

static inline double deg2rad( double deg )
{
    return deg / 180.0 * M_PI;
}

} // namespace utils

} // namespace dnntool

#endif
