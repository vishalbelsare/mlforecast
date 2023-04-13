import ctypes
import random
import time

import numpy as np
from window_ops.rolling import rolling_mean

from mlforecast.core import GroupedArray, TimeSeries

# Setup
_LIB = ctypes.cdll.LoadLibrary('cmake-build-release/libmlforecast.so')
n_series = 50_000
min_length = 500
max_length = 1_000
seed = 0

rng = np.random.RandomState(0)
random.seed(0)
series_lengths = rng.randint(min_length, max_length + 1, n_series)
total_length = series_lengths.sum()
data = rng.rand(total_length).astype(np.float32)
indptr = np.append(0, series_lengths.cumsum()).astype(np.int32)

# Python
ga = GroupedArray(data, indptr)
lag = 2
window_size = 5
min_samples = 1
ts = TimeSeries(
    freq=1,
    lag_transforms={
        lag: [(rolling_mean, window_size, min_samples),
              (rolling_mean, window_size + 1, min_samples),
              (rolling_mean, window_size + 2, min_samples),
              (rolling_mean, window_size + 3, min_samples),]},
    num_threads=4,
)
ts.ga = ga
# to trigger compilation
ts._apply_multithreaded_transforms(False)
# actual execution
start = time.perf_counter()
out = ts._apply_multithreaded_transforms(False)
rm = np.empty((data.size, 4), dtype=np.float32)
for i, tfm in enumerate(ts.transforms.keys()):
    rm[:, i] = out[tfm]
python_time = time.perf_counter() - start
print(f'Python: {1000 * python_time:.0f}ms')

# C++
handle = ctypes.c_void_p()
_LIB.GroupedArray_CreateFromArrays(
    data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    ctypes.c_int32(data.size),
    indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
    ctypes.c_int32(indptr.size),
    ctypes.byref(handle),
)
group_sizes = np.empty(indptr.size - 1, dtype=np.int32)
_LIB.GroupedArray_GetGroupSizes(
    handle,
    group_sizes.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
)
np.testing.assert_equal(group_sizes, np.diff(indptr))
start = time.perf_counter()
rms = np.empty(shape=(4, data.shape[0]), dtype=np.float32)
_LIB.GroupedArray_ComputeRollingMeans(
    handle,
    ctypes.c_int32(lag),
    ctypes.c_int32(window_size),
    ctypes.c_int32(min_samples),
    rms.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
)
rms = rms.T
cpp_time = time.perf_counter() - start
print(f'C++: {1000 * cpp_time:.0f}ms')
np.testing.assert_allclose(rm, rms, atol=1e-4)
print(f'Speedup: {python_time / cpp_time:.1f}x')

start = time.perf_counter()
boost_rm = np.empty_like(data)
_LIB.GroupedArray_BoostComputeRollingMean(
    handle,
    ctypes.c_int32(lag),
    ctypes.c_int32(window_size),
    ctypes.c_int32(min_samples),
    boost_rm.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
)
boost_time = time.perf_counter() - start
print(f'Boost: {1000 * boost_time:.0f}ms')
np.testing.assert_allclose(rm[:, 0], boost_rm, atol=1e-4, rtol=1e-4)