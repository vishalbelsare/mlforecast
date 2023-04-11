import ctypes
import time

import numpy as np
from window_ops.rolling import rolling_mean

from mlforecast.core import GroupedArray
from mlforecast.utils import generate_daily_series

# Setup
series = generate_daily_series(50_000, min_length=500, max_length=1_000)
data = series['y'].values.astype(np.float32)
sizes = series.groupby('unique_id')['y'].size().values
indptr = np.append(0, np.cumsum(sizes)).astype(np.int32)

# Python
ga = GroupedArray(data, indptr)
lag = 2
window_size = 5
min_samples = 1
# to trigger compilation
ga.transform_series(False, lag, rolling_mean, window_size, min_samples)
# actual execution
start = time.perf_counter()
rm = ga.transform_series(False, lag, rolling_mean, window_size, min_samples)
python_time = time.perf_counter() - start
print(f'Python: {1000 * python_time:.0f}ms')

# C++
_LIB = ctypes.cdll.LoadLibrary('cmake-build-release/libmlforecast.so')
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
rms = np.empty_like(data)
_LIB.GroupedArray_ComputeRollingMean(
    handle,
    ctypes.c_int32(lag),
    ctypes.c_int32(window_size),
    ctypes.c_int32(min_samples),
    rms.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
)
cpp_time = time.perf_counter() - start
print(f'C++: {1000 * cpp_time:.0f}ms')
np.testing.assert_allclose(rm, rms, atol=1e-4)
print(f'Speedup: {python_time / cpp_time:.1f}x')