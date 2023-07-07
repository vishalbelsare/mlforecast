---
title: Utils
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
<p>12451 rows × 3 columns</p>
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
<p>12451 rows × 5 columns</p>
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
<p>4268 rows × 3 columns</p>
</div>`];

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

<details>
<summary>Code</summary>

``` python
import random
import reprlib
from itertools import chain
from math import ceil, log10
from typing import Optional, Union

import numpy as np
import pandas as pd
```

</details>

:::

<details>
<summary>Code</summary>

``` python
from fastcore.test import test_eq, test_fail
```

</details>

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

<details>
<summary>Code</summary>

``` python
def generate_daily_series(
    n_series: int, 
    min_length: int = 50,
    max_length: int = 500,
    n_static_features: int = 0,
    equal_ends: bool = False,
    static_as_categorical: bool = True,
    with_trend: bool = False,
    seed: int = 0,
) -> pd.DataFrame:
    """Generates `n_series` of different lengths in the interval [`min_length`, `max_length`].
    
    If `n_static_features > 0`, then each serie gets static features with random values.
    If `equal_ends == True` then all series end at the same date."""
    rng = np.random.RandomState(seed)
    random.seed(seed)
    series_lengths = rng.randint(min_length, max_length + 1, n_series)
    total_length = series_lengths.sum()
    n_digits = ceil(log10(n_series))
    
    dates = pd.date_range('2000-01-01', periods=max_length, freq='D').values
    uids = [
        [f'id_{i:0{n_digits}}'] * serie_length
        for i, serie_length in enumerate(series_lengths)
    ]
    if equal_ends:
        ds = [dates[-serie_length:] for serie_length in series_lengths]
    else:
        ds = [dates[:serie_length] for serie_length in series_lengths]
    y = np.arange(total_length) % 7 + rng.rand(total_length) * 0.5
    series = pd.DataFrame(
        {
            'unique_id': list(chain.from_iterable(uids)),
            'ds': list(chain.from_iterable(ds)),
            'y': y,
        }
    )
    for i in range(n_static_features):
        static_values = np.repeat(rng.randint(0, 100, n_series), series_lengths)
        series[f'static_{i}'] = static_values
        if static_as_categorical:
            series[f'static_{i}'] = series[f'static_{i}'].astype('category')
        if i == 0:
            series['y'] = series['y'] * 0.1 * (1 + static_values)
    series['unique_id'] = series['unique_id'].astype('category')
    series['unique_id'] = series['unique_id'].cat.as_ordered()
    if with_trend:
        coefs = pd.Series(rng.rand(n_series), index=[f'id_{i:0{n_digits}}' for i in range(n_series)])
        trends = series.groupby('unique_id').cumcount()
        trends.index = series['unique_id']
        series['y'] += (coefs * trends).values
    return series
```

</details>

:::

Generate 20 series with lengths between 100 and 1,000.

<details>
<summary>Code</summary>

``` python
n_series = 20
min_length = 100
max_length = 1000

series = generate_daily_series(n_series, min_length, max_length)
series
```

</details>
<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[0] }} />

|       | unique_id | ds         | y        |
|-------|-----------|------------|----------|
| 0     | id_00     | 2000-01-01 | 0.395863 |
| 1     | id_00     | 2000-01-02 | 1.264447 |
| 2     | id_00     | 2000-01-03 | 2.284022 |
| 3     | id_00     | 2000-01-04 | 3.462798 |
| 4     | id_00     | 2000-01-05 | 4.035518 |
| ...   | ...       | ...        | ...      |
| 12446 | id_19     | 2002-03-11 | 0.309275 |
| 12447 | id_19     | 2002-03-12 | 1.189464 |
| 12448 | id_19     | 2002-03-13 | 2.325032 |
| 12449 | id_19     | 2002-03-14 | 3.333198 |
| 12450 | id_19     | 2002-03-15 | 4.306117 |

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[1] }} />

<details>
<summary>Code</summary>

``` python
series_sizes = series.groupby('unique_id').size()
assert series_sizes.size == n_series
assert series_sizes.min() >= min_length
assert series_sizes.max() <= max_length
```

</details>

We can also add static features to each serie (these can be things like
product_id or store_id). Only the first static feature (`static_0`) is
relevant to the target.

<details>
<summary>Code</summary>

``` python
n_static_features = 2

series_with_statics = generate_daily_series(n_series, min_length, max_length, n_static_features)
series_with_statics
```

</details>
<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[2] }} />

|       | unique_id | ds         | y         | static_0 | static_1 |
|-------|-----------|------------|-----------|----------|----------|
| 0     | id_00     | 2000-01-01 | 0.752139  | 18       | 10       |
| 1     | id_00     | 2000-01-02 | 2.402450  | 18       | 10       |
| 2     | id_00     | 2000-01-03 | 4.339642  | 18       | 10       |
| 3     | id_00     | 2000-01-04 | 6.579317  | 18       | 10       |
| 4     | id_00     | 2000-01-05 | 7.667484  | 18       | 10       |
| ...   | ...       | ...        | ...       | ...      | ...      |
| 12446 | id_19     | 2002-03-11 | 2.783477  | 89       | 42       |
| 12447 | id_19     | 2002-03-12 | 10.705175 | 89       | 42       |
| 12448 | id_19     | 2002-03-13 | 20.925285 | 89       | 42       |
| 12449 | id_19     | 2002-03-14 | 29.998780 | 89       | 42       |
| 12450 | id_19     | 2002-03-15 | 38.755054 | 89       | 42       |

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[3] }} />

<details>
<summary>Code</summary>

``` python
for i in range(n_static_features):
    assert all(series_with_statics.groupby('unique_id')[f'static_{i}'].nunique() == 1)
```

</details>

If `equal_ends=False` (the default) then every serie has a different end
date.

<details>
<summary>Code</summary>

``` python
assert series_with_statics.groupby('unique_id')['ds'].max().nunique() > 1
```

</details>

We can have all of them end at the same date by specifying
`equal_ends=True`.

<details>
<summary>Code</summary>

``` python
series_equal_ends = generate_daily_series(n_series, min_length, max_length, equal_ends=True)

assert series_equal_ends.groupby('unique_id')['ds'].max().nunique() == 1
```

</details>

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

<details>
<summary>Code</summary>

``` python
def generate_prices_for_series(series: pd.DataFrame, horizon: int = 7, seed: int = 0) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    unique_last_dates = series.groupby('unique_id')['ds'].max().nunique()
    if unique_last_dates > 1:
        raise ValueError('series must have equal ends.')
    if 'product_id' not in series:
        raise ValueError('series must have a product_id column.')
    day_offset = pd.tseries.frequencies.Day()
    starts_ends = series.groupby('product_id')['ds'].agg([min, max])
    dfs = []
    for idx, (start, end) in starts_ends.iterrows():
        product_df = pd.DataFrame(
            {
                'product_id': idx,
                'price': rng.rand((end - start).days + 1 + horizon),
            },
            index=pd.date_range(start, end + horizon * day_offset, name='ds'),
        )
        dfs.append(product_df)
    prices_catalog = pd.concat(dfs).reset_index()
    return prices_catalog
```

</details>

:::

<details>
<summary>Code</summary>

``` python
series_for_prices = generate_daily_series(20, n_static_features=2, equal_ends=True)
series_for_prices.rename(columns={'static_1': 'product_id'}, inplace=True)
prices_catalog = generate_prices_for_series(series_for_prices, horizon=7)
prices_catalog
```

</details>
<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[4] }} />

|      | ds         | product_id | price    |
|------|------------|------------|----------|
| 0    | 2000-05-07 | 9          | 0.548814 |
| 1    | 2000-05-08 | 9          | 0.715189 |
| 2    | 2000-05-09 | 9          | 0.602763 |
| 3    | 2000-05-10 | 9          | 0.544883 |
| 4    | 2000-05-11 | 9          | 0.423655 |
| ...  | ...        | ...        | ...      |
| 4263 | 2001-05-17 | 93         | 0.800781 |
| 4264 | 2001-05-18 | 93         | 0.909013 |
| 4265 | 2001-05-19 | 93         | 0.904419 |
| 4266 | 2001-05-20 | 93         | 0.327888 |
| 4267 | 2001-05-21 | 93         | 0.971973 |

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[5] }} />

<details>
<summary>Code</summary>

``` python
test_eq(set(prices_catalog['product_id']), set(series_for_prices['product_id']))
test_fail(lambda: generate_prices_for_series(series_equal_ends), contains='product_id')
test_fail(lambda: generate_prices_for_series(series), contains='equal ends')
```

</details>

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’ 6=‘i’}

<details>
<summary>Code</summary>

``` python
def single_split(
    data: pd.DataFrame,
    i_window: int,    
    n_windows: int,
    window_size: int,
    id_col: str,
    time_col: str,
    freq: Union[pd.offsets.BaseOffset, int],
    max_dates: pd.Series,  
    step_size: Optional[int] = None,
    input_size: Optional[int] = None,
):
    if step_size is None:
        step_size = window_size
    test_size = window_size + step_size * (n_windows - 1)
    offset = test_size - i_window * step_size
    train_ends = max_dates - offset * freq
    valid_ends = train_ends + window_size * freq
    train_mask = data[time_col].le(train_ends)
    if input_size is not None:
        train_mask &= data[time_col].gt(train_ends - input_size * freq)
    train_sizes = train_mask.groupby(data[id_col], observed=True).sum()
    if train_sizes.eq(0).any():
        ids = reprlib.repr(train_sizes[train_sizes.eq(0)].index.tolist())
        raise ValueError(f'The following series are too short for the window: {ids}')        
    valid_mask = data[time_col].gt(train_ends) & data[time_col].le(valid_ends)
    cutoffs = (
        train_ends
        .set_axis(data[id_col])
        .groupby(id_col, observed=True)
        .head(1)
        .rename('cutoff')
    )
    return cutoffs, train_mask, valid_mask
```

</details>

:::

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

<details>
<summary>Code</summary>

``` python
def backtest_splits(
    data: pd.DataFrame,
    n_windows: int,
    window_size: int,
    id_col: str,
    time_col: str,
    freq: Union[pd.offsets.BaseOffset, int],
    step_size: Optional[int] = None,
    input_size: Optional[int] = None,
):
    max_dates = data.groupby(id_col, observed=True)[time_col].transform('max')    
    for i in range(n_windows):
        cutoffs, train_mask, valid_mask = single_split(
            data,
            i_window=i,
            n_windows=n_windows,
            window_size=window_size,
            id_col=id_col,
            time_col=time_col,
            freq=freq,
            max_dates=max_dates,
            step_size=step_size,
            input_size=input_size,
        )
        train, valid = data[train_mask], data[valid_mask]
        yield cutoffs, train, valid
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
short_series = generate_daily_series(100, max_length=50)
backtest_results = list(
    backtest_splits(
        short_series,
        n_windows=1,
        window_size=49,
        id_col='unique_id',
        time_col='ds',
        freq=pd.offsets.Day(),
    )
)[0]
test_fail(
    lambda: list(
        backtest_splits(
            short_series,
            n_windows=1,
            window_size=50,
            id_col='unique_id',
            time_col='ds',
            freq=pd.offsets.Day(),
        )
    ),
    contains='The following series are too short'
)
short_series_int = short_series.copy()
short_series_int['ds'] = short_series.groupby('unique_id').transform('cumcount')
backtest_int_results = list(
    backtest_splits(
        short_series_int,
        n_windows=1,
        window_size=40,
        id_col='unique_id',
        time_col='ds',
        freq=1
    )
)[0]
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
max_dates = series.groupby('unique_id')['ds'].max()
day_offset = pd.offsets.Day()

def test_backtest_splits(df, n_windows, window_size, step_size, input_size):
    common_kwargs = dict(
        n_windows=n_windows,
        window_size=window_size,
        id_col='unique_id',
        time_col='ds',
        freq=pd.offsets.Day(), 
        step_size=step_size,
        input_size=input_size,        
    )
    permuted_df = df.sample(frac=1.0)
    splits = backtest_splits(df, **common_kwargs)
    splits_on_permuted = list(backtest_splits(permuted_df, **common_kwargs))
    if step_size is None:
        step_size = window_size
    test_size = window_size + step_size * (n_windows - 1)
    for window, (cutoffs, train, valid) in enumerate(splits):
        offset = test_size - window * step_size
        expected_max_train_dates = max_dates - day_offset * offset
        max_train_dates = train.groupby('unique_id')['ds'].max()
        pd.testing.assert_series_equal(max_train_dates, expected_max_train_dates)
        pd.testing.assert_series_equal(cutoffs, max_train_dates.rename('cutoff'))
        
        if input_size is not None:
            expected_min_train_dates = expected_max_train_dates - day_offset * (input_size - 1)
            min_train_dates = train.groupby('unique_id')['ds'].min()
            pd.testing.assert_series_equal(min_train_dates, expected_min_train_dates)

        expected_min_valid_dates = expected_max_train_dates + day_offset
        min_valid_dates = valid.groupby('unique_id')['ds'].min()
        pd.testing.assert_series_equal(min_valid_dates, expected_min_valid_dates)

        expected_max_valid_dates = expected_max_train_dates + day_offset * window_size
        max_valid_dates = valid.groupby('unique_id')['ds'].max()
        pd.testing.assert_series_equal(max_valid_dates, expected_max_valid_dates)

        if window == n_windows - 1:
            pd.testing.assert_series_equal(max_valid_dates, max_dates)

        _, permuted_train, permuted_valid = splits_on_permuted[window]            
        pd.testing.assert_frame_equal(train, permuted_train.sort_values(['unique_id', 'ds']))
    pd.testing.assert_frame_equal(valid, permuted_valid.sort_values(['unique_id', 'ds']))

for step_size in (None, 1, 2):
    for input_size in (None, 4):
        test_backtest_splits(series, n_windows=3, window_size=14, step_size=step_size, input_size=input_size)
```

</details>

:::

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’ 6=‘i’}

<details>
<summary>Code</summary>

``` python
class PredictionIntervals:
    """Class for storing prediction intervals metadata information."""
    def __init__(self, n_windows: int = 2, window_size: int = 1, method: str = 'conformal_distribution'):
        if n_windows < 2:
            raise ValueError('You need at least two windows to compute conformal intervals')
        allowed_methods = ['conformal_error', 'conformal_distribution']            
        if method not in allowed_methods:
            raise ValueError(f'method must be one of {allowed_methods}')
        self.n_windows = n_windows
        self.window_size = window_size
        self.method = method
```

</details>

:::

