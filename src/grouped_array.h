#ifndef MLFORECAST_GROUPED_ARRAY_H
#define MLFORECAST_GROUPED_ARRAY_H
#endif //MLFORECAST_GROUPED_ARRAY_H

typedef void* GroupedArrayHandle;

extern "C" int GroupedArray_CreateFromArrays(float* data,
                                             int32_t n_data,
                                             int32_t* indptr,
                                             int32_t n_groups,
                                             int num_threads,
                                             GroupedArrayHandle* out);

extern "C" int GroupedArray_GetGroupSizes(GroupedArrayHandle* handle, int32_t *out);
extern "C" int GroupedArray_ComputeRollingMean(GroupedArrayHandle* handle,
                                               int32_t lag,
                                               int32_t window_size,
                                               int32_t min_samples,
                                               float* out);