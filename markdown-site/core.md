---
title: Core
---

export const quartoRawHtml =
[`<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
    .dataframe tbody tr th {
        vertical-align: top;
    }
    .dataframe thead th {
        text-align: right;
    }
</style>
`,`
<p>4874 rows × 5 columns</p>
</div>`,`<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
    .dataframe tbody tr th {
        vertical-align: top;
    }
    .dataframe thead th {
        text-align: right;
    }
</style>
`,`
<p>222 rows × 5 columns</p>
</div>`];

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
%load_ext autoreload
%autoreload 2
```

</details>

:::

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

<details>
<summary>Code</summary>

``` python
import concurrent.futures
import inspect
import warnings
from collections import Counter, OrderedDict
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numba import njit
from sklearn.base import BaseEstimator

from mlforecast.grouped_array import GroupedArray
from mlforecast.target_transforms import BaseTargetTransform, Differences
```

</details>

:::

<details>
<summary>Code</summary>

``` python
import copy

from nbdev import show_doc
from fastcore.test import test_eq, test_fail, test_warns
from window_ops.expanding import expanding_mean
from window_ops.rolling import rolling_mean
from window_ops.shift import shift_array

from mlforecast.utils import generate_daily_series, generate_prices_for_series
```

</details>

## Data format {#data-format}

The required input format is a dataframe with at least the following
columns: \* `unique_id` with a unique identifier for each time serie \*
`ds` with the datestamp and a column \* `y` with the values of the
serie.

Every other column is considered a static feature unless stated
otherwise in `TimeSeries.fit`

<details>
<summary>Code</summary>

``` python
series = generate_daily_series(20, n_static_features=2)
series
```

</details>
<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[0] }} />

|      | unique_id | ds         | y         | static_0 | static_1 |
|------|-----------|------------|-----------|----------|----------|
| 0    | id_00     | 2000-01-01 | 0.740453  | 27       | 53       |
| 1    | id_00     | 2000-01-02 | 3.595262  | 27       | 53       |
| 2    | id_00     | 2000-01-03 | 6.895835  | 27       | 53       |
| 3    | id_00     | 2000-01-04 | 8.499450  | 27       | 53       |
| 4    | id_00     | 2000-01-05 | 11.321981 | 27       | 53       |
| ...  | ...       | ...        | ...       | ...      | ...      |
| 4869 | id_19     | 2000-03-25 | 40.060681 | 97       | 45       |
| 4870 | id_19     | 2000-03-26 | 53.879482 | 97       | 45       |
| 4871 | id_19     | 2000-03-27 | 62.020210 | 97       | 45       |
| 4872 | id_19     | 2000-03-28 | 2.062543  | 97       | 45       |
| 4873 | id_19     | 2000-03-29 | 14.151317 | 97       | 45       |

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[1] }} />

For simplicity we’ll just take one time serie here.

<details>
<summary>Code</summary>

``` python
uids = series['unique_id'].unique()
serie = series[series['unique_id'].eq(uids[0])]
serie
```

</details>
<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[2] }} />

|     | unique_id | ds         | y         | static_0 | static_1 |
|-----|-----------|------------|-----------|----------|----------|
| 0   | id_00     | 2000-01-01 | 0.740453  | 27       | 53       |
| 1   | id_00     | 2000-01-02 | 3.595262  | 27       | 53       |
| 2   | id_00     | 2000-01-03 | 6.895835  | 27       | 53       |
| 3   | id_00     | 2000-01-04 | 8.499450  | 27       | 53       |
| 4   | id_00     | 2000-01-05 | 11.321981 | 27       | 53       |
| ... | ...       | ...        | ...       | ...      | ...      |
| 217 | id_00     | 2000-08-05 | 1.326319  | 27       | 53       |
| 218 | id_00     | 2000-08-06 | 3.823198  | 27       | 53       |
| 219 | id_00     | 2000-08-07 | 5.955518  | 27       | 53       |
| 220 | id_00     | 2000-08-08 | 8.698637  | 27       | 53       |
| 221 | id_00     | 2000-08-09 | 11.925481 | 27       | 53       |

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[3] }} />

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’ 6=‘i’}

<details>
<summary>Code</summary>

``` python
date_features_dtypes = {
    'year': np.uint16,
    'month': np.uint8,
    'day': np.uint8,
    'hour': np.uint8,
    'minute': np.uint8,
    'second': np.uint8,
    'dayofyear': np.uint16,
    'day_of_year': np.uint16,
    'weekofyear': np.uint8,
    'week': np.uint8,
    'dayofweek': np.uint8,
    'day_of_week': np.uint8,
    'weekday': np.uint8,
    'quarter': np.uint8,
    'daysinmonth': np.uint8,
    'is_month_start': np.uint8,
    'is_month_end': np.uint8,
    'is_quarter_start': np.uint8,
    'is_quarter_end': np.uint8,
    'is_year_start': np.uint8,
    'is_year_end': np.uint8,
}
```

</details>

:::

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’ 6=‘i’}

<details>
<summary>Code</summary>

``` python
def _build_transform_name(lag, tfm, *args) -> str:
    """Creates a name for a transformation based on `lag`, the name of the function and its arguments."""
    tfm_name = f'{tfm.__name__}_lag{lag}'
    func_params = inspect.signature(tfm).parameters
    func_args = list(func_params.items())[1:]  # remove input array argument
    changed_params = [
        f'{name}{value}'
        for value, (name, arg) in zip(args, func_args)
        if arg.default != value
    ]
    if changed_params:
        tfm_name += '_' + '_'.join(changed_params)
    return tfm_name
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
test_eq(_build_transform_name(1, expanding_mean), 'expanding_mean_lag1')
test_eq(_build_transform_name(2, rolling_mean, 7), 'rolling_mean_lag2_window_size7')
```

</details>

:::

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’ 6=‘i’}

<details>
<summary>Code</summary>

``` python
def _name_models(current_names):
    ctr = Counter(current_names)
    if not ctr:
        return []
    if max(ctr.values()) < 2:
        return current_names
    names = current_names.copy()
    for i, x in enumerate(reversed(current_names), start=1):
        count = ctr[x]
        if count > 1:
            name = f'{x}{count}'
            ctr[x] -= 1
        else:
            name = x
        names[-i] = name
    return names
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
# one duplicate
names = ['a', 'b', 'a', 'c']
expected = ['a', 'b', 'a2', 'c']
actual = _name_models(names)
assert actual == expected

# no duplicates
names = ['a', 'b', 'c']
actual = _name_models(names)
assert actual == names
```

</details>

:::

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’ 6=‘i’}

<details>
<summary>Code</summary>

``` python
@njit
def _identity(x: np.ndarray) -> np.ndarray:
    """Do nothing to the input."""
    return x


def _as_tuple(x):
    """Return a tuple from the input."""
    if isinstance(x, tuple):
        return x
    return (x,)


@njit
def _expand_target(data, indptr, max_horizon):
    out = np.empty((data.size, max_horizon), dtype=data.dtype)
    n_series = len(indptr) - 1
    n = 0
    for i in range(n_series):
        serie = data[indptr[i] : indptr[i+1]]
        for j in range(serie.size):
            upper = min(serie.size - j, max_horizon)
            for k in range(upper):
                out[n, k] = serie[j + k]
            for k in range(upper, max_horizon):
                out[n, k] = np.nan
            n += 1
    return out
```

</details>

:::

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’ 6=‘i’}

<details>
<summary>Code</summary>

``` python
Freq = Union[int, str, pd.offsets.BaseOffset]
Lags = Iterable[int]
LagTransform = Union[Callable, Tuple[Callable, Any]]
LagTransforms = Dict[int, List[LagTransform]]
DateFeature = Union[str, Callable]
Models = Union[BaseEstimator, List[BaseEstimator], Dict[str, BaseEstimator]]
```

</details>

:::

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

<details>
<summary>Code</summary>

``` python
class TimeSeries:
    """Utility class for storing and transforming time series data."""
    
    def __init__(
        self,
        freq: Optional[Freq] = None,
        lags: Optional[Lags] = None,
        lag_transforms: Optional[LagTransforms] = None,
        date_features: Optional[Iterable[DateFeature]] = None,
        differences: Optional[Iterable[int]] = None,
        num_threads: int = 1,
        target_transforms: Optional[List[BaseTargetTransform]] = None,        
    ):
        if isinstance(freq, str):
            self.freq = pd.tseries.frequencies.to_offset(freq)
        elif isinstance(freq, pd.offsets.BaseOffset):
            self.freq = freq
        elif isinstance(freq, int):
            self.freq = freq
        elif freq is None:
            self.freq = 1
        else:
            raise ValueError(
                'Unknown frequency type '
                'Please use a str, int or offset frequency type.'
            )
        if not isinstance(num_threads, int) or num_threads < 1:
            warnings.warn('Setting num_threads to 1.')
            num_threads = 1
        self.lags = [] if lags is None else list(lags)
        self.lag_transforms = {} if lag_transforms is None else lag_transforms
        self.date_features = [] if date_features is None else list(date_features)
        if differences is not None:
            warnings.warn("The differences argument is deprecated and will be removed in a future version.\n"
                "Please pass an `mlforecast.target_transforms.Differences` instance to the `target_transforms` argument instead.""")
            if target_transforms is None:
                target_transforms = [Differences(differences)]
            else:
                target_transforms = [Differences(differences)] + target_transforms
        self.num_threads = num_threads
        self.target_transforms = target_transforms
        for feature in self.date_features:
            if callable(feature) and feature.__name__ == '<lambda>':
                raise ValueError(
                    "Can't use a lambda as a date feature because the function name gets used as the feature name."
                )
        
        self.transforms: Dict[str, Tuple[Any, ...]] = OrderedDict()
        for lag in self.lags:
            self.transforms[f'lag{lag}'] = (lag, _identity)
        for lag in self.lag_transforms.keys():
            for tfm_args in self.lag_transforms[lag]:
                tfm, *args = _as_tuple(tfm_args)
                tfm_name = _build_transform_name(lag, tfm, *args)
                self.transforms[tfm_name] = (lag, tfm, *args)

        self.ga: GroupedArray

    @property
    def _date_feature_names(self):
        return [f.__name__ if callable(f) else f for f in self.date_features]
        
    @property
    def features(self) -> List[str]:
        """Names of all computed features."""
        return list(self.transforms.keys()) + self._date_feature_names
                
    def __repr__(self):
        return (
            f'TimeSeries(freq={self.freq}, '
            f'transforms={list(self.transforms.keys())}, '
            f'date_features={self._date_feature_names}, '
            f'num_threads={self.num_threads})'
        )
        
    def _fit(
        self,
        df: pd.DataFrame,
        id_col: str,
        time_col: str,
        target_col: str,
        static_features: Optional[List[str]] = None,
        keep_last_n: Optional[int] = None,
    ) -> 'TimeSeries':
        """Save the series values, ids and last dates."""
        for col in (id_col, time_col, target_col):
            if col not in df:
                raise ValueError(f"Data doesn't contain {col} column")
        if df[target_col].isnull().any():
            raise ValueError(f'{target_col} column contains null values.')
        if pd.api.types.is_datetime64_dtype(df[time_col]):
            if self.freq == 1:
                raise ValueError('Must set frequency when using a timestamp type column.')
        elif np.issubdtype(df[time_col].dtype.type, np.integer):
            if self.freq != 1:
                warnings.warn('Setting `freq=1` since time col is int.')
                self.freq = 1
        else:
            raise ValueError(f'{time_col} must be either timestamp or integer.')
        self.id_col = id_col
        self.target_col = target_col
        self.time_col = time_col
        to_drop = [id_col, time_col, target_col]
        self.static_features = static_features
        if static_features is None:
            static_features = df.columns.drop([time_col, target_col]).tolist()
        elif id_col not in static_features:
            static_features = [id_col] + static_features
        else:  # static_features defined and contain id_col
            to_drop = [time_col, target_col]
        self.static_features_ = (
            df
            [static_features]
            .groupby(id_col, observed=True)
            .head(1)
            .reset_index(drop=True)
        )
        sort_idxs = pd.core.sorting.lexsort_indexer([df[id_col], df[time_col]])
        self.restore_idxs = np.empty(df.shape[0], dtype=np.int32)        
        self.restore_idxs[sort_idxs] = np.arange(df.shape[0])
        sorted_df = df[[id_col, time_col, target_col]].iloc[sort_idxs]
        if self.target_transforms is not None:
            for tfm in self.target_transforms:
                tfm.set_column_names(id_col, time_col, target_col)
                sorted_df = tfm.fit_transform(sorted_df)
        sorted_df = sorted_df.set_index([id_col, time_col])
        self.uids = sorted_df.index.unique(level=0)
        self.ga = GroupedArray.from_sorted_df(sorted_df, id_col, target_col)
        self.features_ = self._compute_transforms()
        if keep_last_n is not None:
            self.ga = self.ga.take_from_groups(slice(-keep_last_n, None))
        self._ga = GroupedArray(self.ga.data, self.ga.indptr)
        self.last_dates = sorted_df.index.get_level_values(self.time_col)[self.ga.indptr[1:] - 1]
        self.features_order_ = df.columns.drop(to_drop).tolist() + self.features
        return self

    def _apply_transforms(self, updates_only: bool = False) -> Dict[str, np.ndarray]:
        """Apply the transformations using the main process.
        
        If `updates_only` then only the updates are returned.
        """
        results = {}
        offset = 1 if updates_only else 0
        for tfm_name, (lag, tfm, *args) in self.transforms.items():
            results[tfm_name] = self.ga.transform_series(
                updates_only, lag - offset, tfm, *args
            )
        return results

    def _apply_multithreaded_transforms(
        self, updates_only: bool = False
    ) -> Dict[str, np.ndarray]:
        """Apply the transformations using multithreading.
        
        If `updates_only` then only the updates are returned.
        """        
        future_to_result = {}
        results = {}
        offset = 1 if updates_only else 0        
        with concurrent.futures.ThreadPoolExecutor(self.num_threads) as executor:
            for tfm_name, (lag, tfm, *args) in self.transforms.items():
                future = executor.submit(
                    self.ga.transform_series,
                    updates_only,
                    lag - offset,
                    tfm,
                    *args,
                )
                future_to_result[future] = tfm_name
            for future in concurrent.futures.as_completed(future_to_result):
                tfm_name = future_to_result[future]
                results[tfm_name] = future.result()
        return results
    
    def _compute_transforms(self) -> Dict[str, np.ndarray]:
        """Compute the transformations defined in the constructor.

        If `self.num_threads > 1` these are computed using multithreading."""
        if self.num_threads == 1 or len(self.transforms) == 1:
            return self._apply_transforms()
        return self._apply_multithreaded_transforms()
    
    def _compute_date_feature(self, dates, feature): 
        if callable(feature):
            feat_name = feature.__name__
            feat_vals = feature(dates)
        else:
            feat_name = feature
            if feature in ('week', 'weekofyear'):
                dates = dates.isocalendar()
            feat_vals = getattr(dates, feature)
        vals = np.asarray(feat_vals)
        feat_dtype = date_features_dtypes.get(feature)
        if feat_dtype is not None:
            vals = vals.astype(feat_dtype)
        return feat_name, vals

    def _transform(
        self,
        df: pd.DataFrame,
        dropna: bool = True,
        max_horizon: Optional[int] = None,
        return_X_y: bool = False,
    ) -> pd.DataFrame:
        """Add the features to `df`.
        
        if `dropna=True` then all the null rows are dropped."""
        modifies_target = bool(self.target_transforms)
        df = df.copy(deep=modifies_target and not return_X_y)

        # lag transforms
        for feat in self.transforms.keys():
            df[feat] = self.features_[feat][self.restore_idxs]

        # date features
        dates = df[self.time_col]
        if not np.issubdtype(dates.dtype.type, np.integer):
            dates = pd.DatetimeIndex(dates)
        for feature in self.date_features:
            feat_name, feat_vals = self._compute_date_feature(dates, feature)
            df[feat_name] = feat_vals

        # target
        self.max_horizon = max_horizon
        if max_horizon is None:
            if modifies_target:
                target = pd.Series(self.ga.data[self.restore_idxs], index=df.index)
            else:
                target = df[self.target_col]
        else:
            target = pd.DataFrame(
                self.ga.expand_target(max_horizon)[self.restore_idxs],
                index=df.index,
                columns=[f"{self.target_col}{i}" for i in range(max_horizon)],
            )
            
        # determine rows to keep
        if dropna:
            feature_nulls = df[self.features].isnull().any(axis=1)
            target_nulls = target.isnull()
            if target_nulls.ndim == 2:
                # target nulls for each horizon are dropped in MLForecast.fit_models
                # we just drop rows here for which all the target values are null
                target_nulls = target_nulls.all(axis=1)
            keep_rows = ~(feature_nulls | target_nulls).values
        else:
            keep_rows = np.full(df.shape[0], True)

        # assemble return
        xs = df.columns.drop(self.target_col)
        if return_X_y:
            return df.loc[keep_rows, xs], target.loc[keep_rows]
        if max_horizon is None:
            if modifies_target:
                df[self.target_col] = target
        else:
            df = pd.concat([df[xs], target], axis=1)
        return df.loc[keep_rows]


    def fit_transform(
        self,
        data: pd.DataFrame,
        id_col: str,
        time_col: str,
        target_col: str,
        static_features: Optional[List[str]] = None,
        dropna: bool = True,
        keep_last_n: Optional[int] = None,
        max_horizon: Optional[int] = None,
        return_X_y: bool = False,
    ) -> pd.DataFrame:
        """Add the features to `data` and save the required information for the predictions step.
        
        If not all features are static, specify which ones are in `static_features`.
        If you don't want to drop rows with null values after the transformations set `dropna=False`
        If `keep_last_n` is not None then that number of observations is kept across all series for updates.
        """
        self.dropna = dropna
        self.keep_last_n = keep_last_n
        self._fit(data, id_col, time_col, target_col, static_features, keep_last_n)
        return self._transform(data, dropna=dropna, max_horizon=max_horizon, return_X_y=return_X_y)

    def _update_y(self, new: np.ndarray) -> None:
        """Appends the elements of `new` to every time serie.

        These values are used to update the transformations and are stored as predictions."""
        if not hasattr(self, 'y_pred'):
            self.y_pred = []
        self.y_pred.append(new)
        new_arr = np.asarray(new)
        self.ga = self.ga.append(new_arr)     
        
    def _update_features(self) -> pd.DataFrame:
        """Compute the current values of all the features using the latest values of the time series."""
        if not hasattr(self, 'curr_dates'):
            self.curr_dates = self.last_dates.copy()
            self.test_dates = []
        self.curr_dates += self.freq
        self.test_dates.append(self.curr_dates)

        if self.num_threads == 1 or len(self.transforms) == 1:
            features = self._apply_transforms(updates_only=True)
        else:
            features = self._apply_multithreaded_transforms(updates_only=True)

        for feature in self.date_features:
            feat_name, feat_vals = self._compute_date_feature(self.curr_dates, feature)
            features[feat_name] = feat_vals

        features_df = pd.DataFrame(features, columns=self.features)
        features_df[self.id_col] = self.uids
        features_df[self.time_col] = self.curr_dates        
        return self.static_features_.merge(features_df, on=self.id_col)
            
    def _get_raw_predictions(self) -> np.ndarray:
        return np.array(self.y_pred).ravel('F')

    def _get_predictions(self) -> pd.DataFrame:
        """Get all the predicted values with their corresponding ids and datestamps."""
        n_preds = len(self.y_pred)
        uids = pd.Series(
            np.repeat(self.uids, n_preds), name=self.id_col, dtype=self.uids.dtype
        )
        df = pd.DataFrame(
            {
                self.id_col: uids,
                self.time_col: np.array(self.test_dates).ravel('F'),
                f'{self.target_col}_pred': self._get_raw_predictions(),
            },
        )
        return df

    def _predict_setup(self) -> None:
        self.curr_dates = self.last_dates.copy()
        self.test_dates = []
        self.y_pred = []
        self.ga = GroupedArray(self._ga.data, self._ga.indptr)
        
    def _get_features_for_next_step(self, dynamic_dfs):
        new_x = self._update_features()
        if dynamic_dfs:
            for df in dynamic_dfs:
                new_x = new_x.merge(df, how='left')
            new_x = new_x.sort_values(self.id_col)
        nulls = new_x.isnull().any()
        if any(nulls):
            warnings.warn(
                f'Found null values in {", ".join(nulls[nulls].index)}.'
            )
        return new_x[self.features_order_]       
    
    def _predict_recursive(
        self,
        models: Dict[str, BaseEstimator],
        horizon: int,
        dynamic_dfs: Optional[List[pd.DataFrame]] = None,
        before_predict_callback: Optional[Callable] = None,
        after_predict_callback: Optional[Callable] = None,
    ) -> pd.DataFrame:
        """Use `model` to predict the next `horizon` timesteps."""
        if dynamic_dfs is None:
            dynamic_dfs = []
        for i, (name, model) in enumerate(models.items()):
            self._predict_setup() 
            for _ in range(horizon):
                new_x = self._get_features_for_next_step(dynamic_dfs)
                if before_predict_callback is not None:
                    new_x = before_predict_callback(new_x)
                predictions = model.predict(new_x)
                if after_predict_callback is not None:
                    predictions_serie = pd.Series(predictions, index=self.uids)
                    predictions = after_predict_callback(predictions_serie).values
                self._update_y(predictions)
            if i == 0:
                preds = self._get_predictions()
                preds = preds.rename(columns={f'{self.target_col}_pred': name}, copy=False)
            else:           
                preds[name] = self._get_raw_predictions()
        return preds

    def _predict_multi(
        self,
        models: Dict[str, BaseEstimator],
        horizon: int,
        dynamic_dfs: Optional[List[pd.DataFrame]] = None,
        before_predict_callback: Optional[Callable] = None,
    ) -> pd.DataFrame:
        assert self.max_horizon is not None
        if horizon > self.max_horizon:
            raise ValueError(f'horizon must be at most max_horizon ({self.max_horizon})')
        if dynamic_dfs is None:
            dynamic_dfs = []
        uids = np.repeat(self.uids, horizon)
        dates = np.hstack(
            [
                date + (i+1) * self.freq
                for date in self.last_dates                
                for i in range(horizon)
            ]
        )
        result = pd.DataFrame({self.id_col: uids, self.time_col: dates})
        for name, model in models.items():
            self._predict_setup()
            new_x = self._get_features_for_next_step(dynamic_dfs)
            if before_predict_callback is not None:
                new_x = before_predict_callback(new_x)
            predictions = np.empty((new_x.shape[0], horizon))
            for i in range(horizon):
                predictions[:, i] = model[i].predict(new_x)
            raw_preds = predictions.ravel()
            result[name] = raw_preds
        return result
    
    def predict(
        self,
        models: Dict[str, Union[BaseEstimator, List[BaseEstimator]]],
        horizon: int,
        dynamic_dfs: Optional[List[pd.DataFrame]] = None,
        before_predict_callback: Optional[Callable] = None,
        after_predict_callback: Optional[Callable] = None,        
    ) -> pd.DataFrame:
        if getattr(self, 'max_horizon', None) is None:
            preds = self._predict_recursive(
                models,
                horizon,
                dynamic_dfs,
                before_predict_callback,
                after_predict_callback,
            )
        else:
            preds = self._predict_multi(
                models,
                horizon,
                dynamic_dfs,
                before_predict_callback,
            )
        if self.target_transforms is not None:
            for tfm in self.target_transforms[::-1]:
                preds = tfm.inverse_transform(preds)
        return preds
    
    def update(self, df: pd.DataFrame) -> None:
        """Update the values of the stored series."""
        df = df.sort_values([self.id_col, self.time_col])
        new_sizes = df.groupby(self.id_col, observed=True).size()
        prev_sizes = pd.Series(np.full(self.uids.size, 0), index=self.uids)
        sizes = new_sizes.add(prev_sizes, fill_value=0)
        values = df[self.target_col].values
        new_groups = ~sizes.index.isin(self.uids)
        self.last_dates = pd.Index(
            df
            .groupby(self.id_col, observed=True)
            [self.time_col]
            .max()
            .reindex(sizes.index)
            .fillna(dict(zip(self.uids, self.last_dates)))
        ).astype(self.last_dates.dtype)
        self.uids = sizes.index
        new_statics = df.iloc[new_sizes.cumsum() - 1].set_index(self.id_col)
        orig_dtypes = self.static_features_.dtypes
        if pd.api.types.is_categorical_dtype(orig_dtypes[self.id_col]):
            orig_categories = orig_dtypes[self.id_col].categories.tolist()
            missing_categories = set(self.uids) - set(orig_categories)
            if missing_categories:
                orig_dtypes[self.id_col] = pd.CategoricalDtype(categories=orig_categories + list(missing_categories))
        self.static_features_ = self.static_features_.set_index(self.id_col).reindex(self.uids)
        self.static_features_.update(new_statics)
        self.static_features_ = self.static_features_.reset_index().astype(orig_dtypes)
        self.ga = self.ga.append_several(
            new_sizes=sizes.values.astype(np.int32),
            new_values=values,
            new_groups=new_groups,
        )
        self._ga = GroupedArray(self.ga.data, self.ga.indptr)
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
# differences
n = 7 * 14
x = pd.DataFrame(
    {
        'id': np.repeat(0, n),
        'ds': np.arange(n),
        'y': np.arange(7)[[x % 7 for x in np.arange(n)]]
    },
)
x['y'] = x['ds'] * 0.1 + x['y']
ts = TimeSeries(freq=1, target_transforms=[Differences([1, 7])])
ts._fit(x.iloc[:-14], id_col='id', time_col='ds', target_col='y')
np.testing.assert_allclose(
    x['y'].diff(1).diff(7).values[:-14],
    ts.ga.data,
)
ts.y_pred = np.zeros(14)
class A:
    def fit(self, X):
        return self
    def predict(self, X):
        return np.zeros(X.shape[0])
xx = ts.predict({'A': A()}, 14)
np.testing.assert_allclose(xx['A'], x['y'].tail(14).values)
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
test_fail(lambda: TimeSeries(date_features=[lambda: 1]), contains="Can't use a lambda")
```

</details>

:::

The `TimeSeries` class takes care of defining the transformations to be
performed (`lags`, `lag_transforms` and `date_features`). The
transformations can be computed using multithreading if
`num_threads > 1`.

<details>
<summary>Code</summary>

``` python
def month_start_or_end(dates):
    return dates.is_month_start | dates.is_month_end

flow_config = dict(
    freq='W-THU',
    lags=[7],
    lag_transforms={
        1: [expanding_mean, (rolling_mean, 7)]
    },
    date_features=['dayofweek', 'week', month_start_or_end]
)

ts = TimeSeries(**flow_config)
ts
```

</details>

``` text
TimeSeries(freq=<Week: weekday=3>, transforms=['lag7', 'expanding_mean_lag1', 'rolling_mean_lag1_window_size7'], date_features=['dayofweek', 'week', 'month_start_or_end'], num_threads=1)
```

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
test_eq(
    TimeSeries(freq=ts.freq).freq,
    TimeSeries(freq='W-THU').freq
)
```

</details>

:::

The frequency is converted to an offset.

<details>
<summary>Code</summary>

``` python
test_eq(ts.freq, pd.tseries.frequencies.to_offset(flow_config['freq']))
```

</details>

The date features are stored as they were passed to the constructor.

<details>
<summary>Code</summary>

``` python
test_eq(ts.date_features, flow_config['date_features'])
```

</details>

The transformations are stored as a dictionary where the key is the name
of the transformation (name of the column in the dataframe with the
computed features), which is built using `build_transform_name` and the
value is a tuple where the first element is the lag it is applied to,
then the function and then the function arguments.

<details>
<summary>Code</summary>

``` python
test_eq(
    ts.transforms, 
    {
        'lag7': (7, _identity),
        'expanding_mean_lag1': (1, expanding_mean), 
        'rolling_mean_lag1_window_size7': (1, rolling_mean, 7)
        
    }
)
```

</details>

Note that for `lags` we define the transformation as the identity
function applied to its corresponding lag. This is because
`_transform_series` takes the lag as an argument and shifts the array
before computing the transformation.

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
# int y is converted to float32
serie2 = serie.copy()
serie2['y'] = serie2['y'].astype(int)
ts = TimeSeries(num_threads=1, freq='D')
ts._fit(serie2, id_col='unique_id', time_col='ds', target_col='y')
test_eq(ts.ga.data.dtype, np.float32)
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
# _compute_transforms
y = serie.y.values
lag_1 = shift_array(y, 1)

for num_threads in (1, 2):
    ts = TimeSeries(**flow_config)
    ts._fit(serie, id_col='unique_id', time_col='ds', target_col='y')
    transforms = ts._compute_transforms()

    np.testing.assert_equal(transforms['lag7'], shift_array(y, 7))
    np.testing.assert_equal(transforms['expanding_mean_lag1'], expanding_mean(lag_1))
    np.testing.assert_equal(transforms['rolling_mean_lag1_window_size7'], rolling_mean(lag_1, 7))
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
# update_y
ts = TimeSeries(freq='D', lags=[1])
ts._fit(serie, id_col='unique_id', time_col='ds', target_col='y')

max_size = np.diff(ts.ga.indptr)
ts._update_y([1])
ts._update_y([2])

test_eq(np.diff(ts.ga.indptr), max_size + 2)
test_eq(ts.ga.data[-2:], [1, 2])
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
# _update_features
ts = TimeSeries(**flow_config)
ts._fit(serie, id_col='unique_id', time_col='ds', target_col='y')
updates = ts._update_features().drop(columns='ds')

last_date = serie['ds'].max()
first_prediction_date = last_date + ts.freq

# these have an offset becase we can now "see" our last y value
expected = pd.DataFrame({
    'unique_id': ts.uids,
    'lag7': shift_array(y, 6)[-1],
    'expanding_mean_lag1': expanding_mean(y)[-1],
    'rolling_mean_lag1_window_size7': rolling_mean(y, 7)[-1],
    'dayofweek': np.uint8([getattr(first_prediction_date, 'dayofweek')]),
    'week': np.uint8([first_prediction_date.isocalendar()[1]]),
    'month_start_or_end': month_start_or_end(first_prediction_date)
})
statics = serie.tail(1).drop(columns=['ds', 'y'])
pd.testing.assert_frame_equal(updates, statics.merge(expected))


test_eq(ts.curr_dates[0], first_prediction_date)
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
# _get_predictions
ts = TimeSeries(freq='D', lags=[1])
ts._fit(serie, id_col='unique_id', time_col='ds', target_col='y')
ts._update_features()
ts._update_y([1.])
preds = ts._get_predictions()

last_ds = serie['ds'].max()
expected = pd.DataFrame({'unique_id': serie['unique_id'][[0]], 'ds': [last_ds + ts.freq], 'y_pred': [1.]})
pd.testing.assert_frame_equal(preds, expected)
```

</details>

:::

<details>
<summary>Code</summary>

``` python
show_doc(TimeSeries.fit_transform, title_level=2)
```

</details>

------------------------------------------------------------------------

## TimeSeries.fit_transform {#timeseries.fit_transform}

> ``` text
>  TimeSeries.fit_transform (data:pandas.core.frame.DataFrame, id_col:str,
>                            time_col:str, target_col:str,
>                            static_features:Optional[List[str]]=None,
>                            dropna:bool=True,
>                            keep_last_n:Optional[int]=None,
>                            max_horizon:Optional[int]=None,
>                            return_X_y:bool=False)
> ```

Add the features to `data` and save the required information for the
predictions step.

If not all features are static, specify which ones are in
`static_features`. If you don’t want to drop rows with null values after
the transformations set `dropna=False` If `keep_last_n` is not None then
that number of observations is kept across all series for updates.

<details>
<summary>Code</summary>

``` python
flow_config = dict(
    freq='D',
    lags=[7, 14],
    lag_transforms={
        2: [
            (rolling_mean, 7),
            (rolling_mean, 14),
        ]
    },
    date_features=['dayofweek', 'month', 'year'],
    num_threads=2
)

ts = TimeSeries(**flow_config)
_ = ts.fit_transform(series, id_col='unique_id', time_col='ds', target_col='y')
```

</details>

The series values are stored as a GroupedArray in an attribute `ga`. If
the data type of the series values is an int then it is converted to
`np.float32`, this is because lags generate `np.nan`s so we need a float
data type for them.

<details>
<summary>Code</summary>

``` python
np.testing.assert_equal(ts.ga.data, series.y.values)
```

</details>

The series ids are stored in an `uids` attribute.

<details>
<summary>Code</summary>

``` python
test_eq(ts.uids, series['unique_id'].unique())
```

</details>

For each time serie, the last observed date is stored so that
predictions start from the last date + the frequency.

<details>
<summary>Code</summary>

``` python
test_eq(ts.last_dates, series.groupby('unique_id')['ds'].max().values)
```

</details>

The last row of every serie without the `y` and `ds` columns are taken
as static features.

<details>
<summary>Code</summary>

``` python
pd.testing.assert_frame_equal(
    ts.static_features_,
    series.groupby('unique_id').tail(1).drop(columns=['ds', 'y']).reset_index(drop=True),
)
```

</details>

If you pass `static_features` to `TimeSeries.fit_transform` then only
these are kept.

<details>
<summary>Code</summary>

``` python
ts.fit_transform(series, id_col='unique_id', time_col='ds', target_col='y', static_features=['static_0'])

pd.testing.assert_frame_equal(
    ts.static_features_,
    series.groupby('unique_id').tail(1)[['unique_id', 'static_0']].reset_index(drop=True),
)
```

</details>

You can also specify keep_last_n in TimeSeries.fit_transform, which
means that after computing the features for training we want to keep
only the last n samples of each time serie for computing the updates.
This saves both memory and time, since the updates are performed by
running the transformation functions on all time series again and
keeping only the last value (the update).

If you have very long time series and your updates only require a small
sample it’s recommended that you set keep_last_n to the minimum number
of samples required to compute the updates, which in this case is 15
since we have a rolling mean of size 14 over the lag 2 and in the first
update the lag 2 becomes the lag 1. This is because in the first update
the lag 1 is the last value of the series (or the lag 0), the lag 2 is
the lag 1 and so on.

<details>
<summary>Code</summary>

``` python
keep_last_n = 15

ts = TimeSeries(**flow_config)
df = ts.fit_transform(series, id_col='unique_id', time_col='ds', target_col='y', keep_last_n=keep_last_n)

expected_lags = ['lag7', 'lag14']
expected_transforms = ['rolling_mean_lag2_window_size7', 
                       'rolling_mean_lag2_window_size14']
expected_date_features = ['dayofweek', 'month', 'year']

test_eq(ts.features, expected_lags + expected_transforms + expected_date_features)
test_eq(ts.static_features_.columns.tolist() + ts.features, df.columns.drop(['ds', 'y']).tolist())
# we dropped 2 rows because of the lag 2 and 13 more to have the window of size 14
test_eq(df.shape[0], series.shape[0] - (2 + 13) * ts.ga.ngroups)
test_eq(ts.ga.data.size, ts.ga.ngroups * keep_last_n)
```

</details>

`TimeSeries.fit_transform` requires that the *y* column doesn’t have any
null values. This is because the transformations could propagate them
forward, so if you have null values in the *y* column you’ll get an
error.

<details>
<summary>Code</summary>

``` python
series_with_nulls = series.copy()
series_with_nulls.loc[1, 'y'] = np.nan
test_fail(
    lambda: ts.fit_transform(series_with_nulls, id_col='unique_id', time_col='ds', target_col='y'),
    contains='y column contains null values'
)
```

</details>

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
# unsorted df
ts = TimeSeries(**flow_config)
df = ts.fit_transform(series, id_col='unique_id', time_col='ds', target_col='y')
unordered_series = series.sample(frac=1.0)
assert not unordered_series.set_index('ds', append=True).index.is_monotonic_increasing
df2 = ts.fit_transform(unordered_series, id_col='unique_id', time_col='ds', target_col='y')
pd.testing.assert_frame_equal(df, df2.sort_values(['unique_id', 'ds']))
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
# non-standard df
ts = TimeSeries(**flow_config)
df = ts.fit_transform(series, id_col='unique_id', time_col='ds', target_col='y')
non_std_series = series.reset_index().rename(columns={'unique_id': 'some_id', 'ds': 'timestamp', 'y': 'value'})
non_std_res = ts.fit_transform(non_std_series, id_col='some_id', time_col='timestamp', target_col='value')
non_std_res = non_std_res.reset_index(drop=True)
pd.testing.assert_frame_equal(
    df.reset_index(),
    non_std_res.rename(columns={'timestamp': 'ds', 'value': 'y', 'some_id': 'unique_id'})
)
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
# integer timestamps
def identity(x):
    return x

flow_config_int_ds = copy.deepcopy(flow_config)
flow_config_int_ds['date_features'] = [identity]
del flow_config_int_ds['freq']
ts = TimeSeries(**flow_config_int_ds)
int_ds_series = series.copy()
int_ds_series['ds'] = int_ds_series['ds'].astype('int64')
int_ds_res = ts.fit_transform(int_ds_series, id_col='unique_id', time_col='ds', target_col='y')
int_ds_res['ds'] = pd.to_datetime(int_ds_res['ds'])
int_ds_res['identity'] = pd.to_datetime(int_ds_res['ds'])
df2 = df.drop(columns=flow_config['date_features'])
df2['identity'] = df2['ds']
pd.testing.assert_frame_equal(df2, int_ds_res)
```

</details>

:::

<details>
<summary>Code</summary>

``` python
show_doc(TimeSeries.predict, title_level=2)
```

</details>

------------------------------------------------------------------------

## TimeSeries.predict {#timeseries.predict}

> ``` text
>  TimeSeries.predict (models:Dict[str,Union[sklearn.base.BaseEstimator,List
>                      [sklearn.base.BaseEstimator]]], horizon:int, dynamic_
>                      dfs:Optional[List[pandas.core.frame.DataFrame]]=None,
>                      before_predict_callback:Optional[Callable]=None,
>                      after_predict_callback:Optional[Callable]=None)
> ```

Once we have a trained model we can use `TimeSeries.predict` passing the
model and the horizon to get the predictions back.

<details>
<summary>Code</summary>

``` python
class DummyModel:
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X['lag7'].values

horizon = 7
model = DummyModel()
ts = TimeSeries(**flow_config)
ts.fit_transform(series, id_col='unique_id', time_col='ds', target_col='y')
predictions = ts.predict({'DummyModel': model}, horizon)

grouped_series = series.groupby('unique_id')
expected_preds = grouped_series['y'].tail(7)  # the model predicts the lag-7
last_dates = grouped_series['ds'].max()
expected_dsmin = last_dates + ts.freq
expected_dsmax = last_dates + horizon * ts.freq
grouped_preds = predictions.groupby('unique_id')

np.testing.assert_allclose(predictions['DummyModel'], expected_preds)
pd.testing.assert_series_equal(grouped_preds['ds'].min(), expected_dsmin)
pd.testing.assert_series_equal(grouped_preds['ds'].max(), expected_dsmax)
```

</details>

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
model = DummyModel()
ts = TimeSeries(**flow_config)
ts.fit_transform(series, id_col='unique_id', time_col='ds', target_col='y')
predictions = ts.predict({'DummyModel': model}, horizon=horizon)
ts = TimeSeries(**flow_config_int_ds)
ts.fit_transform(int_ds_series, id_col='unique_id', time_col='ds', target_col='y')
int_ds_predictions = ts.predict({'DummyModel': model}, horizon=horizon)
pd.testing.assert_frame_equal(predictions.drop(columns='ds'), int_ds_predictions.drop(columns='ds'))
```

</details>

:::

If we have dynamic features we can pass them to `dynamic_dfs`.

<details>
<summary>Code</summary>

``` python
class PredictPrice:
    def predict(self, X):
        return X['price']

series = generate_daily_series(20, n_static_features=2, equal_ends=True)
dynamic_series = series.rename(columns={'static_1': 'product_id'})
prices_catalog = generate_prices_for_series(dynamic_series)
series_with_prices = dynamic_series.merge(prices_catalog, how='left')

model = PredictPrice()
ts = TimeSeries(**flow_config)
ts.fit_transform(
    series_with_prices,
    id_col='unique_id',
    time_col='ds',
    target_col='y',
    static_features=['static_0', 'product_id'],
)
predictions = ts.predict({'PredictPrice': model}, horizon=1, dynamic_dfs=[prices_catalog])

expected_prices = series_with_prices.reset_index()[['unique_id', 'product_id']].drop_duplicates()
expected_prices['ds'] = series_with_prices['ds'].max() + ts.freq
expected_prices = expected_prices.reset_index()
expected_prices = expected_prices.merge(prices_catalog, on=['product_id', 'ds'], how='left')
expected_prices = expected_prices[['unique_id', 'ds', 'price']]

pd.testing.assert_frame_equal(
    predictions.rename(columns={'PredictPrice': 'price'}),
    expected_prices
)
```

</details>
<details>
<summary>Code</summary>

``` python
show_doc(TimeSeries.update, title_level=2)
```

</details>

------------------------------------------------------------------------

## TimeSeries.update {#timeseries.update}

> ``` text
>  TimeSeries.update (df:pandas.core.frame.DataFrame)
> ```

Update the values of the stored series.

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
class NaiveModel:
    def predict(self, X: pd.DataFrame):
        return X['lag1']

two_series = series[series['unique_id'].isin(['id_00', 'id_19'])]
ts = TimeSeries(freq='D', lags=[1])
ts.fit_transform(
    two_series,
    id_col='unique_id',
    time_col='ds',
    target_col='y',
)
last_vals_two_series = two_series.groupby('unique_id').tail(1)
last_val_id0 = last_vals_two_series[lambda x: x['unique_id'].eq('id_00')].copy()
new_values = last_val_id0.copy()
new_values['ds'] += pd.offsets.Day()
new_serie = pd.DataFrame({
    'unique_id': ['new_idx', 'new_idx'],
    'ds': pd.to_datetime(['2020-01-01', '2020-01-02']),
    'y': [5.0, 6.0],
    'static_0': [0, 0],
    'static_1': [1, 1],
})
new_values = pd.concat([new_values, new_serie])
ts.update(new_values)
test_warns(lambda: ts.predict({'Naive': NaiveModel()}, 1))
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    preds = ts.predict({'Naive': NaiveModel()}, 1)
expected_id0 = last_val_id0.copy()
expected_id0['ds'] += pd.offsets.Day()
expected_id1 = last_vals_two_series[lambda x: x['unique_id'].eq('id_19')].copy()
last_val_new_serie = new_serie.tail(1)[['unique_id', 'ds', 'y']]
expected = pd.concat([expected_id0, expected_id1, last_val_new_serie])
expected = expected[['unique_id', 'ds', 'y']]
expected = expected.rename(columns={'y': 'Naive'}).reset_index(drop=True)
expected['ds'] += pd.offsets.Day()
pd.testing.assert_frame_equal(preds, expected)
pd.testing.assert_frame_equal(
    ts.static_features_,
    (
        pd.concat([last_vals_two_series, new_serie.tail(1)])
        [['unique_id', 'static_0', 'static_1']]
        .astype(ts.static_features_.dtypes)
        .reset_index(drop=True)
    )
)
```

</details>

:::

