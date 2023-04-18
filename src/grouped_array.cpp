#include <memory>
#include <vector>

#include "grouped_array.h"
#include "rolling.h"
#include "boost_rolling_mean.h"

class GroupedArray {
private:
    float* data_;
    int32_t* indptr_;
    int32_t n_data_;
    int32_t n_groups_;
    int num_threads_;
public:
    GroupedArray(float* data, int32_t n_data, int32_t* indptr, int32_t n_indptr, int num_threads) :
        data_(data), indptr_(indptr), n_data_(n_data), n_groups_(n_indptr - 1), num_threads_(num_threads) {}
    void GetGroupSizes(int32_t out[]) {
        for (int i = 0; i < n_groups_; ++i) {
            out[i] = indptr_[i + 1] - indptr_[i];
        }
    }
    void ComputeRollingMean(int32_t lag, int window_size, int min_samples, float* out) {
        RollingMean roll_mean = RollingMean(window_size, min_samples);
        # pragma omp parallel for schedule(static) num_threads(num_threads_)
        for (int i = 0; i < n_groups_; ++i) {
            int32_t start = indptr_[i];
            int32_t end = indptr_[i + 1];
            for (int j = 0; j < lag; ++j) {
                out[start + j] = std::numeric_limits<float>::quiet_NaN();
            }
            roll_mean.transform(data_ + start,
                                end - start - lag,
                                out + start + lag);
        }
    }

    void UpdateRollingMean(int lag, int window_size, int min_samples, float* out) {
        RollingMean roll_mean = RollingMean(window_size, min_samples);
        # pragma omp parallel for schedule(static) num_threads(num_threads_)
        for (int i = 0; i < n_groups_; ++i) {
            int32_t start = indptr_[i];
            int32_t end = indptr_[i + 1];
            out[i] = roll_mean.update(data_ + start, end - start - lag + 1);
        }
    }

    void BoostComputeRollingMean(int32_t lag, int32_t window_size, int32_t min_samples, float* out) {
        for (int i = 0; i < n_groups_; ++i) {
            int32_t start = indptr_[i];
            int32_t end = indptr_[i + 1];
            for (int j = 0; j < lag; ++j) {
                out[start + j] = std::numeric_limits<float>::quiet_NaN();
            }
            boost_rolling_mean(data_ + start,
                               end - start - lag,
                               window_size,
                               min_samples,
                               out + start + lag);
        }
    }
};

extern "C" int GroupedArray_CreateFromArrays(float* data,
                                             int32_t n_data,
                                             int32_t* indptr,
                                             int32_t n_groups,
                                             int num_threads,
                                             GroupedArrayHandle* out) {
    *out = new GroupedArray(data, n_data, indptr, n_groups, num_threads);
    return 0;
}

extern "C" int GroupedArray_GetGroupSizes(GroupedArrayHandle* handle,
                                          int32_t *out) {
    auto ga = reinterpret_cast<GroupedArray*>(handle);
    ga->GetGroupSizes(out);
    return 0;
}

extern "C" int GroupedArray_ComputeRollingMean(GroupedArrayHandle* handle,
                                               int lag,
                                               int window_size,
                                               int min_samples,
                                               float* out) {
    auto ga = reinterpret_cast<GroupedArray*>(handle);
    ga->ComputeRollingMean(lag, window_size, min_samples, out);
    return 1;
}

extern "C" int GroupedArray_BoostComputeRollingMean(GroupedArrayHandle* handle,
                                                    int32_t lag,
                                                    int32_t window_size,
                                                    int32_t min_samples,
                                                    float* out) {
    auto ga = reinterpret_cast<GroupedArray*>(handle);
    ga->BoostComputeRollingMean(lag, window_size, min_samples, out);
    return 0;
}

extern "C" int GroupedArray_UpdateRollingMean(GroupedArrayHandle* handle,
                                              int lag,
                                              int window_size,
                                              int min_samples,
                                              float* out) {
    auto ga = reinterpret_cast<GroupedArray*>(handle);
    ga->UpdateRollingMean(lag, window_size, min_samples, out);
    return 0;
}