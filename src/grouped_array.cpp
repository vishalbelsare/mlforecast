#include <memory>
#include <vector>

#include "grouped_array.h"
#include "rolling.h"

class GroupedArray {
private:
    std::unique_ptr<std::vector<float>> data_;
    std::unique_ptr<std::vector<int32_t>> indptr_;
    int32_t n_groups_;
public:
    GroupedArray(float* data,
                 int32_t n_data,
                 int32_t* indptr,
                 int32_t n_indptr) {
        data_ = std::make_unique<std::vector<float>>(data, data + n_data);
        indptr_ = std::make_unique<std::vector<int32_t>>(indptr, indptr + n_indptr);
        n_groups_ = n_indptr - 1;
    }
    void GetGroupSizes(int32_t out[]) {
        for (int i = 0; i < n_groups_; ++i) {
            out[i] = indptr_->operator[](i + 1) - indptr_->operator[](i);
        }
    }
    void ComputeRollingMean(int32_t lag, int32_t window_size, int32_t min_samples, float* out) {
        float* data = data_->data();
        for (int i = 0; i < n_groups_; ++i) {
            int32_t start = indptr_->operator[](i);
            int32_t end = indptr_->operator[](i + 1);
            for (int j = 0; j < lag; ++j) {
                out[start + j] = std::numeric_limits<float>::quiet_NaN();
            }
            rolling_mean(data + start,
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
                                             GroupedArrayHandle* out) {
    *out = new GroupedArray(data, n_data, indptr, n_groups);
    return 1;
}

extern "C" int GroupedArray_GetGroupSizes(GroupedArrayHandle* handle,
                                          int32_t *out) {
    auto ga = reinterpret_cast<GroupedArray*>(handle);
    ga->GetGroupSizes(out);
    return 1;
}

extern "C" int GroupedArray_ComputeRollingMean(GroupedArrayHandle* handle,
                                               int32_t lag,
                                               int32_t window_size,
                                               int32_t min_samples,
                                               float* out) {
    auto ga = reinterpret_cast<GroupedArray*>(handle);
    ga->ComputeRollingMean(lag, window_size, min_samples, out);
    return 1;
}