import ctypes
import random
import time

import numpy as np
from window_ops.rolling import rolling_mean

from mlforecast.core import TimeSeries
from mlforecast.grouped_array import GroupedArray

# Setup
_LIB = ctypes.cdll.LoadLibrary('cmake-build-release/libmlforecast.so')
# n_series = 1_000
n_series = 20_000
min_length = 1_000
max_length = 2_000
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
num_threads = 2
ts = TimeSeries(
    freq=1,
    lag_transforms={  # type: ignore
        lag: [
            (rolling_mean, window_size, min_samples),
            (rolling_mean, window_size + 1, min_samples),
            (rolling_mean, window_size + 2, min_samples),
            (rolling_mean, window_size + 3, min_samples),
        ]
    },
    num_threads=num_threads,
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

# updates
ts._apply_multithreaded_transforms(True)
start = time.perf_counter()
updates = ts._apply_multithreaded_transforms(True)
py_upd = np.empty((n_series, 4), dtype=np.float32)
for i, tfm in enumerate(ts.transforms.keys()):
    py_upd[:, i] = updates[tfm]
python_upd_time = time.perf_counter() - start

# C++
handle = ctypes.c_void_p()
_LIB.GroupedArray_CreateFromArrays(
    data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    ctypes.c_int32(data.size),
    indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
    ctypes.c_int32(indptr.size),
    ctypes.c_int(num_threads),
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
for i in range(4):
    _LIB.GroupedArray_ComputeRollingMean(
        handle,
        ctypes.c_int(lag),
        ctypes.c_int(window_size + i),
        ctypes.c_int(min_samples),
        rms[i].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
rms = rms.T
cpp_time = time.perf_counter() - start
print(f'C++: {1000 * cpp_time:.0f}ms')
np.testing.assert_allclose(rm, rms, atol=1e-4)
print(f'Speedup: {python_time / cpp_time:.1f}x')

print(f'Python updates: {1000 * python_upd_time:.0f}ms')
start = time.perf_counter()
c_upd = np.empty(shape=(4, n_series), dtype=np.float32)
for offset in range(4):
    _LIB.GroupedArray_UpdateRollingMean(
        handle,
        ctypes.c_int(lag),
        ctypes.c_int(window_size + offset),
        ctypes.c_int(min_samples),
        c_upd[offset].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
c_upd = c_upd.T
c_upd_time = time.perf_counter() - start
print(f'C++ updates: {1000 * c_upd_time:.0f}ms')
np.testing.assert_allclose(py_upd, c_upd, atol=1e-4)
print(f'Updates speedup: {python_upd_time / c_upd_time:.1f}x')


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