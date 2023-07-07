---
title: LightGBMCV
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
<p>4032 rows × 3 columns</p>
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
<p>384 rows × 5 columns</p>
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
<p>192 rows × 4 columns</p>
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

> Time series cross validation with LightGBM.

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

<details>
<summary>Code</summary>

``` python
import copy
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import lightgbm as lgb
import numpy as np
import pandas as pd

from mlforecast.core import (
    DateFeature,
    Freq,
    LagTransforms,
    Lags,
    TimeSeries,
)
from mlforecast.utils import backtest_splits
from mlforecast.target_transforms import BaseTargetTransform
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
from nbdev import show_doc
```

</details>

:::

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’ 6=‘i’}

<details>
<summary>Code</summary>

``` python
def _mape(y_true, y_pred, ids, _dates):
    abs_pct_err = abs(y_true - y_pred) / y_true
    return abs_pct_err.groupby(ids, observed=True).mean().mean()

def _rmse(y_true, y_pred, ids, _dates):
    sq_err = (y_true - y_pred) ** 2
    return sq_err.groupby(ids, observed=True).mean().pow(0.5).mean()

_metric2fn = {'mape': _mape, 'rmse': _rmse}

def _update(bst, n):
    for _ in range(n):
        bst.update()

def _predict(ts, bst, valid, h, before_predict_callback, after_predict_callback):
    ex_cols_to_drop = [ts.id_col, ts.time_col, ts.target_col]
    static_features = ts.static_features_.columns.drop(ts.id_col).tolist()
    ex_cols_to_drop.extend(static_features)
    has_ex = not valid.columns.drop(ex_cols_to_drop).empty
    dynamic_dfs = [valid.drop(columns=static_features + [ts.target_col])] if has_ex else None
    preds = ts.predict({'Booster': bst}, h, dynamic_dfs, before_predict_callback, after_predict_callback)
    return valid.merge(preds, on=[ts.id_col, ts.time_col], how='left')

def _update_and_predict(ts, bst, valid, n, h, before_predict_callback, after_predict_callback):
    _update(bst, n)
    return _predict(ts, bst, valid, h, before_predict_callback, after_predict_callback)
```

</details>

:::

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’ 6=‘i’}

<details>
<summary>Code</summary>

``` python
CVResult = Tuple[int, float]
```

</details>

:::

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

<details>
<summary>Code</summary>

``` python
class LightGBMCV:
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
        """Create LightGBM CV object.

        Parameters
        ----------
        freq : str or int, optional (default=None)
            Pandas offset alias, e.g. 'D', 'W-THU' or integer denoting the frequency of the series.
        lags : list of int, optional (default=None)
            Lags of the target to use as features.
        lag_transforms : dict of int to list of functions, optional (default=None)
            Mapping of target lags to their transformations.
        date_features : list of str or callable, optional (default=None)
            Features computed from the dates. Can be pandas date attributes or functions that will take the dates as input.
        differences : list of int, optional (default=None)
            Differences to take of the target before computing the features. These are restored at the forecasting step.
        num_threads : int (default=1)
            Number of threads to use when computing the features.
        target_transforms : list of transformers, optional(default=None)
            Transformations that will be applied to the target before computing the features and restored after the forecasting step.            
        """            
        self.num_threads = num_threads
        cpu_count = os.cpu_count()
        if cpu_count is None:
            num_cpus = 1
        else:
            num_cpus = cpu_count
        self.bst_threads = max(num_cpus // num_threads, 1)
        self.ts = TimeSeries(
            freq=freq,
            lags=lags,
            lag_transforms=lag_transforms,
            date_features=date_features,
            differences=differences,
            num_threads=self.bst_threads,
            target_transforms=target_transforms,
        )
        
    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'freq={self.ts.freq}, '
            f'lag_features={list(self.ts.transforms.keys())}, '
            f'date_features={self.ts.date_features}, '
            f'num_threads={self.num_threads}, '
            f'bst_threads={self.bst_threads})'
        )
    
    def setup(
        self,
        data: pd.DataFrame,
        n_windows: int,
        window_size: int,
        id_col: str = 'unique_id',
        time_col: str = 'ds',
        target_col: str = 'y',
        step_size: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None,
        static_features: Optional[List[str]] = None,
        dropna: bool = True,
        keep_last_n: Optional[int] = None,
        weights: Optional[Sequence[float]] = None,
        metric: Union[str, Callable] = 'mape',
        input_size: Optional[int] = None,        
    ):
        """Initialize internal data structures to iteratively train the boosters. Use this before calling partial_fit.
        
        Parameters
        ----------
        data : pandas DataFrame
            Series data in long format.
        n_windows : int
            Number of windows to evaluate.
        window_size : int
            Number of test periods in each window.
        id_col : str (default='unique_id')
            Column that identifies each serie.
        time_col : str (default='ds')
            Column that identifies each timestep, its values can be timestamps or integers.
        target_col : str (default='y')
            Column that contains the target.
        step_size : int, optional (default=None)
            Step size between each cross validation window. If None it will be equal to `window_size`.
        params : dict, optional(default=None)
            Parameters to be passed to the LightGBM Boosters.       
        static_features : list of str, optional (default=None)
            Names of the features that are static and will be repeated when forecasting.
        dropna : bool (default=True)
            Drop rows with missing values produced by the transformations.
        keep_last_n : int, optional (default=None)
            Keep only these many records from each serie for the forecasting step. Can save time and memory if your features allow it.
        weights : sequence of float, optional (default=None)
            Weights to multiply the metric of each window. If None, all windows have the same weight.
        metric : str or callable, default='mape'
            Metric used to assess the performance of the models and perform early stopping.
        input_size : int, optional (default=None)
            Maximum training samples per serie in each window. If None, will use an expanding window.            
            
        Returns
        -------
        self : LightGBMCV
            CV object with internal data structures for partial_fit.
        """
        if weights is None:
            self.weights = np.full(n_windows, 1 / n_windows)        
        elif len(weights) != n_windows:
            raise ValueError('Must specify as many weights as the number of windows')
        else:
            self.weights = np.asarray(weights)
        if callable(metric):
            self.metric_fn = metric
            self.metric_name = metric.__name__
        else:
            if metric not in _metric2fn:
                raise ValueError(f'{metric} is not one of the implemented metrics: ({", ".join(_metric2fn.keys())})')
            self.metric_fn = _metric2fn[metric]
            self.metric_name = metric
        if np.issubdtype(data[time_col].dtype.type, np.integer):
            freq = 1
        else:
            freq = self.ts.freq
        self.items = []
        self.window_size = window_size
        self.id_col = id_col
        self.time_col = time_col
        self.target_col = target_col
        self.params = {} if params is None else params
        splits = backtest_splits(
            data,
            n_windows=n_windows,
            window_size=window_size,
            id_col=id_col,
            time_col=time_col,
            freq=freq,
            step_size=step_size,
            input_size=input_size,
        )
        for _, train, valid in splits:
            ts = copy.deepcopy(self.ts)
            prep = ts.fit_transform(train, id_col, time_col, target_col, static_features, dropna, keep_last_n)
            ds = lgb.Dataset(prep.drop(columns=[id_col, time_col, target_col]), prep[target_col]).construct()
            bst = lgb.Booster({**self.params, 'num_threads': self.bst_threads}, ds)
            bst.predict = partial(bst.predict, num_threads=self.bst_threads)
            self.items.append((ts, bst, valid))
        return self

    def _single_threaded_partial_fit(
        self,
        metric_values,
        num_iterations,
        before_predict_callback: Optional[Callable] = None,
        after_predict_callback: Optional[Callable] = None,
    ):  
        for j, (ts, bst, valid) in enumerate(self.items):
            preds = _update_and_predict(
                ts=ts,
                bst=bst,
                valid=valid,
                n=num_iterations,
                h=self.window_size,
                before_predict_callback=before_predict_callback,
                after_predict_callback=after_predict_callback,
            )
            metric_values[j] = self.metric_fn(
                preds[self.target_col], preds['Booster'], preds[self.id_col], preds[self.time_col]
            )

    def _multithreaded_partial_fit(
        self,
        metric_values,
        num_iterations,
        before_predict_callback: Optional[Callable] = None,
        after_predict_callback: Optional[Callable] = None,
    ):                           
        with ThreadPoolExecutor(self.num_threads) as executor:
            futures = []
            for ts, bst, valid in self.items:
                _update(bst, num_iterations)
                future = executor.submit(
                    _predict,
                    ts=ts,
                    bst=bst,
                    valid=valid,
                    h=self.window_size,
                    before_predict_callback=before_predict_callback,
                    after_predict_callback=after_predict_callback,
                )
                futures.append(future)
            cv_preds = [f.result() for f in futures]
        metric_values[:] = [
            self.metric_fn(
                preds[self.target_col], preds['Booster'], preds[self.id_col], preds[self.time_col]
            )
            for preds in cv_preds
        ]
        
    def partial_fit(
        self,
        num_iterations: int,
        before_predict_callback: Optional[Callable] = None,
        after_predict_callback: Optional[Callable] = None,
    ) -> float:
        """Train the boosters for some iterations.
        
        Parameters
        ----------
        num_iterations : int
            Number of boosting iterations to run
        before_predict_callback : callable, optional (default=None)
            Function to call on the features before computing the predictions.
                This function will take the input dataframe that will be passed to the model for predicting and should return a dataframe with the same structure.
                The series identifier is on the index.
        after_predict_callback : callable, optional (default=None)
            Function to call on the predictions before updating the targets.
                This function will take a pandas Series with the predictions and should return another one with the same structure.
                The series identifier is on the index.  
                    
        Returns
        -------
        metric_value : float
            Weighted metric after training for num_iterations.
        """
        metric_values = np.empty(len(self.items))
        if self.num_threads == 1:
            self._single_threaded_partial_fit(
                metric_values, num_iterations, before_predict_callback, after_predict_callback
            )
        else:
            self._multithreaded_partial_fit(
                metric_values, num_iterations, before_predict_callback, after_predict_callback
            )
        return metric_values @ self.weights
    
    def should_stop(self, hist, early_stopping_evals, early_stopping_pct) -> bool:
        if len(hist) < early_stopping_evals + 1:
            return False
        improvement_pct = 1 - hist[-1][1] / hist[-(early_stopping_evals + 1)][1]
        return improvement_pct < early_stopping_pct

    def find_best_iter(self, hist, early_stopping_evals) -> int:
        best_iter, best_score = hist[-1]
        for r, m in hist[-(early_stopping_evals + 1):-1]:
            if m < best_score:
                best_score = m
                best_iter = r
        return best_iter
   
    def fit(
        self,
        data: pd.DataFrame,
        n_windows: int,
        window_size: int,
        id_col: str = 'unique_id',
        time_col: str = 'ds',
        target_col: str = 'y',
        step_size: Optional[int] = None,
        num_iterations: int = 100,
        params: Optional[Dict[str, Any]] = None,
        static_features: Optional[List[str]] = None,
        dropna: bool = True,
        keep_last_n: Optional[int] = None,
        eval_every: int = 10,
        weights: Optional[Sequence[float]] = None,
        metric: Union[str, Callable] = 'mape',
        verbose_eval: bool = True,
        early_stopping_evals: int = 2,
        early_stopping_pct: float = 0.01,
        compute_cv_preds: bool = False,
        before_predict_callback: Optional[Callable] = None,
        after_predict_callback: Optional[Callable] = None,
        input_size: Optional[int] = None,
    ) -> List[CVResult]:
        """Train boosters simultaneously and assess their performance on the complete forecasting window.
        
        Parameters
        ----------
        data : pandas DataFrame
            Series data in long format.
        n_windows : int
            Number of windows to evaluate.
        window_size : int
            Number of test periods in each window.    
        id_col : str (default='unique_id')
            Column that identifies each serie.
        time_col : str (default='ds')
            Column that identifies each timestep, its values can be timestamps or integers.
        target_col : str (default='y')
            Column that contains the target.
        step_size : int, optional (default=None)
            Step size between each cross validation window. If None it will be equal to `window_size`.
        num_iterations : int (default=100)
            Maximum number of boosting iterations to run.
        params : dict, optional(default=None)
            Parameters to be passed to the LightGBM Boosters.            
        static_features : list of str, optional (default=None)
            Names of the features that are static and will be repeated when forecasting.
        dropna : bool (default=True)
            Drop rows with missing values produced by the transformations.
        keep_last_n : int, optional (default=None)
            Keep only these many records from each serie for the forecasting step. Can save time and memory if your features allow it.
        eval_every : int (default=10)
            Number of boosting iterations to train before evaluating on the whole forecast window.
        weights : sequence of float, optional (default=None)
            Weights to multiply the metric of each window. If None, all windows have the same weight.
        metric : str or callable, default='mape'
            Metric used to assess the performance of the models and perform early stopping.
        verbose_eval : bool
            Print the metrics of each evaluation.
        early_stopping_evals : int (default=2)
            Maximum number of evaluations to run without improvement.
        early_stopping_pct : float (default=0.01)
            Minimum percentage improvement in metric value in `early_stopping_evals` evaluations.
        compute_cv_preds : bool (default=True)
            Compute predictions for each window after finding the best iteration.        
        before_predict_callback : callable, optional (default=None)
            Function to call on the features before computing the predictions.
                This function will take the input dataframe that will be passed to the model for predicting and should return a dataframe with the same structure.
                The series identifier is on the index.
        after_predict_callback : callable, optional (default=None)
            Function to call on the predictions before updating the targets.
                This function will take a pandas Series with the predictions and should return another one with the same structure.
                The series identifier is on the index.
        input_size : int, optional (default=None)
            Maximum training samples per serie in each window. If None, will use an expanding window.                

        Returns
        -------
        cv_result : list of tuple.
            List of (boosting rounds, metric value) tuples.
        """
        self.setup(
            data=data,
            n_windows=n_windows,
            window_size=window_size,
            params=params,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
            input_size=input_size,
            step_size=step_size,
            static_features=static_features,
            dropna=dropna,
            keep_last_n=keep_last_n,
            weights=weights,
            metric=metric,
        )
        hist = []
        for i in range(0, num_iterations, eval_every):
            metric_value = self.partial_fit(eval_every, before_predict_callback, after_predict_callback)
            rounds = eval_every + i
            hist.append((rounds, metric_value))
            if verbose_eval:
                print(f'[{rounds:,d}] {self.metric_name}: {metric_value:,f}')                
            if self.should_stop(hist, early_stopping_evals, early_stopping_pct):
                print(f"Early stopping at round {rounds:,}")
                break
        self.best_iteration_ = self.find_best_iter(hist, early_stopping_evals)
        print(f'Using best iteration: {self.best_iteration_:,}')
        hist = hist[:self.best_iteration_ // eval_every]
        for _, bst, _ in self.items:
            bst.best_iteration = self.best_iteration_

        self.cv_models_ = {f'Booster{i}': item[1] for i, item in enumerate(self.items)}
        if compute_cv_preds:
            with ThreadPoolExecutor(self.num_threads) as executor:
                futures = []            
                for ts, bst, valid in self.items:
                    future = executor.submit(
                        _predict,
                        ts=ts,
                        bst=bst,
                        valid=valid,
                        h=self.window_size,
                        before_predict_callback=before_predict_callback,
                        after_predict_callback=after_predict_callback,
                    )
                    futures.append(future)            
                self.cv_preds_ = pd.concat([f.result().assign(window=i) for i, f in enumerate(futures)])
        self.ts._fit(data, id_col, time_col, target_col, static_features, keep_last_n)
        return hist

    def predict(
        self,
        horizon: int,
        dynamic_dfs: Optional[List[pd.DataFrame]] = None,
        before_predict_callback: Optional[Callable] = None,
        after_predict_callback: Optional[Callable] = None,
    ) -> pd.DataFrame:
        """Compute predictions with each of the trained boosters.
        
        Parameters
        ----------
        horizon : int
            Number of periods to predict.
        dynamic_dfs : list of pandas DataFrame, optional (default=None)
            Future values of the dynamic features, e.g. prices.
        before_predict_callback : callable, optional (default=None)
            Function to call on the features before computing the predictions.
                This function will take the input dataframe that will be passed to the model for predicting and should return a dataframe with the same structure.
                The series identifier is on the index.
        after_predict_callback : callable, optional (default=None)
            Function to call on the predictions before updating the targets.
                This function will take a pandas Series with the predictions and should return another one with the same structure.
                The series identifier is on the index. 
                    
        Returns
        -------
        result : pandas DataFrame
            Predictions for each serie and timestep, with one column per window.
        """        
        return self.ts.predict(
            self.cv_models_,
            horizon,
            dynamic_dfs,
            before_predict_callback,
            after_predict_callback,
        )
```

</details>

:::

<details>
<summary>Code</summary>

``` python
show_doc(LightGBMCV)
```

</details>

------------------------------------------------------------------------

### LightGBMCV {#lightgbmcv}

> ``` text
>  LightGBMCV (freq:Union[int,str,pandas._libs.tslibs.offsets.BaseOffset,Non
>              eType]=None, lags:Optional[Iterable[int]]=None, lag_transform
>              s:Optional[Dict[int,List[Union[Callable,Tuple[Callable,Any]]]
>              ]]=None,
>              date_features:Optional[Iterable[Union[str,Callable]]]=None,
>              differences:Optional[Iterable[int]]=None, num_threads:int=1, 
>              target_transforms:Optional[List[mlforecast.target_transforms.
>              BaseTargetTransform]]=None)
> ```

Create LightGBM CV object.

|                   | **Type** | **Default** | **Details**                                                                                                               |
|------|------------------|-------------------------|-------------------------|
| freq              | Union    | None        | Pandas offset alias, e.g. ‘D’, ‘W-THU’ or integer denoting the frequency of the series.                                   |
| lags              | Optional | None        | Lags of the target to use as features.                                                                                    |
| lag_transforms    | Optional | None        | Mapping of target lags to their transformations.                                                                          |
| date_features     | Optional | None        | Features computed from the dates. Can be pandas date attributes or functions that will take the dates as input.           |
| differences       | Optional | None        | Differences to take of the target before computing the features. These are restored at the forecasting step.              |
| num_threads       | int      | 1           | Number of threads to use when computing the features.                                                                     |
| target_transforms | Optional | None        | Transformations that will be applied to the target before computing the features and restored after the forecasting step. |

## Example {#example}

This shows an example with just 4 series of the M4 dataset. If you want
to run it yourself on all of them, you can refer to [this
notebook](https://www.kaggle.com/code/lemuz90/m4-competition-cv).

<details>
<summary>Code</summary>

``` python
import random

from datasetsforecast.m4 import M4, M4Info
from fastcore.test import test_eq, test_fail
from mlforecast.target_transforms import Differences
from nbdev import show_doc
from window_ops.ewm import ewm_mean
from window_ops.rolling import rolling_mean, seasonal_rolling_mean
```

</details>
<details>
<summary>Code</summary>

``` python
group = 'Hourly'
await M4.async_download('data', group=group)
df, *_ = M4.load(directory='data', group=group)
df['ds'] = df['ds'].astype('int')
ids = df['unique_id'].unique()
random.seed(0)
sample_ids = random.choices(ids, k=4)
sample_df = df[df['unique_id'].isin(sample_ids)]
sample_df
```

</details>
<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[0] }} />

|        | unique_id | ds   | y    |
|--------|-----------|------|------|
| 86796  | H196      | 1    | 11.8 |
| 86797  | H196      | 2    | 11.4 |
| 86798  | H196      | 3    | 11.1 |
| 86799  | H196      | 4    | 10.8 |
| 86800  | H196      | 5    | 10.6 |
| ...    | ...       | ...  | ...  |
| 325235 | H413      | 1004 | 99.0 |
| 325236 | H413      | 1005 | 88.0 |
| 325237 | H413      | 1006 | 47.0 |
| 325238 | H413      | 1007 | 41.0 |
| 325239 | H413      | 1008 | 34.0 |

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[1] }} />

<details>
<summary>Code</summary>

``` python
info = M4Info[group]
horizon = info.horizon
valid = sample_df.groupby('unique_id').tail(horizon)
train = sample_df.drop(valid.index)
train.shape, valid.shape
```

</details>

``` text
((3840, 3), (192, 3))
```

What LightGBMCV does is emulate [LightGBM’s cv
function](https://lightgbm.readthedocs.io/en/v3.3.2/pythonapi/lightgbm.cv.html#lightgbm.cv)
where several Boosters are trained simultaneously on different
partitions of the data, that is, one boosting iteration is performed on
all of them at a time. This allows to have an estimate of the error by
iteration, so if we combine this with early stopping we can find the
best iteration to train a final model using all the data or even use
these individual models’ predictions to compute an ensemble.

In order to have a good estimate of the forecasting performance of our
model we compute predictions for the whole test period and compute a
metric on that. Since this step can slow down training, there’s an
`eval_every` parameter that can be used to control this, that is, if
`eval_every=10` (the default) every 10 boosting iterations we’re going
to compute forecasts for the complete window and report the error.

We also have early stopping parameters:

-   `early_stopping_evals`: how many evaluations of the full window
    should we go without improving to stop training?
-   `early_stopping_pct`: what’s the minimum percentage improvement we
    want in these `early_stopping_evals` in order to keep training?

This makes the LightGBMCV class a good tool to quickly test different
configurations of the model. Consider the following example, where we’re
going to try to find out which features can improve the performance of
our model. We start just using lags.

<details>
<summary>Code</summary>

``` python
static_fit_config = dict(
    n_windows=2,
    window_size=horizon,
    params={'verbose': -1},
    compute_cv_preds=True,
)
cv = LightGBMCV(
    freq=1,
    lags=[24 * (i+1) for i in range(7)],  # one week of lags
)
```

</details>
<details>
<summary>Code</summary>

``` python
show_doc(LightGBMCV.fit)
```

</details>

------------------------------------------------------------------------

### LightGBMCV.fit {#lightgbmcv.fit}

> ``` text
>  LightGBMCV.fit (data:pandas.core.frame.DataFrame, n_windows:int,
>                  window_size:int, id_col:str='unique_id',
>                  time_col:str='ds', target_col:str='y',
>                  step_size:Optional[int]=None, num_iterations:int=100,
>                  params:Optional[Dict[str,Any]]=None,
>                  static_features:Optional[List[str]]=None,
>                  dropna:bool=True, keep_last_n:Optional[int]=None,
>                  eval_every:int=10,
>                  weights:Optional[Sequence[float]]=None,
>                  metric:Union[str,Callable]='mape',
>                  verbose_eval:bool=True, early_stopping_evals:int=2,
>                  early_stopping_pct:float=0.01,
>                  compute_cv_preds:bool=False,
>                  before_predict_callback:Optional[Callable]=None,
>                  after_predict_callback:Optional[Callable]=None,
>                  input_size:Optional[int]=None)
> ```

Train boosters simultaneously and assess their performance on the
complete forecasting window.

|                         | **Type**  | **Default** | **Details**                                                                                                                                                                                                                                                          |
|------|------------------|-------------------------|-------------------------|
| data                    | DataFrame |             | Series data in long format.                                                                                                                                                                                                                                          |
| n_windows               | int       |             | Number of windows to evaluate.                                                                                                                                                                                                                                       |
| window_size             | int       |             | Number of test periods in each window.                                                                                                                                                                                                                               |
| id_col                  | str       | unique_id   | Column that identifies each serie.                                                                                                                                                                                                                                   |
| time_col                | str       | ds          | Column that identifies each timestep, its values can be timestamps or integers.                                                                                                                                                                                      |
| target_col              | str       | y           | Column that contains the target.                                                                                                                                                                                                                                     |
| step_size               | Optional  | None        | Step size between each cross validation window. If None it will be equal to `window_size`.                                                                                                                                                                           |
| num_iterations          | int       | 100         | Maximum number of boosting iterations to run.                                                                                                                                                                                                                        |
| params                  | Optional  | None        | Parameters to be passed to the LightGBM Boosters.                                                                                                                                                                                                                    |
| static_features         | Optional  | None        | Names of the features that are static and will be repeated when forecasting.                                                                                                                                                                                         |
| dropna                  | bool      | True        | Drop rows with missing values produced by the transformations.                                                                                                                                                                                                       |
| keep_last_n             | Optional  | None        | Keep only these many records from each serie for the forecasting step. Can save time and memory if your features allow it.                                                                                                                                           |
| eval_every              | int       | 10          | Number of boosting iterations to train before evaluating on the whole forecast window.                                                                                                                                                                               |
| weights                 | Optional  | None        | Weights to multiply the metric of each window. If None, all windows have the same weight.                                                                                                                                                                            |
| metric                  | Union     | mape        | Metric used to assess the performance of the models and perform early stopping.                                                                                                                                                                                      |
| verbose_eval            | bool      | True        | Print the metrics of each evaluation.                                                                                                                                                                                                                                |
| early_stopping_evals    | int       | 2           | Maximum number of evaluations to run without improvement.                                                                                                                                                                                                            |
| early_stopping_pct      | float     | 0.01        | Minimum percentage improvement in metric value in `early_stopping_evals` evaluations.                                                                                                                                                                                |
| compute_cv_preds        | bool      | False       | Compute predictions for each window after finding the best iteration.                                                                                                                                                                                                |
| before_predict_callback | Optional  | None        | Function to call on the features before computing the predictions.<br> This function will take the input dataframe that will be passed to the model for predicting and should return a dataframe with the same structure.<br> The series identifier is on the index. |
| after_predict_callback  | Optional  | None        | Function to call on the predictions before updating the targets.<br> This function will take a pandas Series with the predictions and should return another one with the same structure.<br> The series identifier is on the index.                                  |
| input_size              | Optional  | None        | Maximum training samples per serie in each window. If None, will use an expanding window.                                                                                                                                                                            |
| **Returns**             | **List**  |             | **List of (boosting rounds, metric value) tuples.**                                                                                                                                                                                                                  |

<details>
<summary>Code</summary>

``` python
hist = cv.fit(train, **static_fit_config)
```

</details>

``` text
[LightGBM] [Info] Start training from score 51.745632
[10] mape: 0.590690
[20] mape: 0.251093
[30] mape: 0.143643
[40] mape: 0.109723
[50] mape: 0.102099
[60] mape: 0.099448
[70] mape: 0.098349
[80] mape: 0.098006
[90] mape: 0.098718
Early stopping at round 90
Using best iteration: 80
```

By setting `compute_cv_preds` we get the predictions from each model on
their corresponding validation fold.

<details>
<summary>Code</summary>

``` python
cv.cv_preds_
```

</details>
<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[2] }} />

|     | unique_id | ds  | y    | Booster   | window |
|-----|-----------|-----|------|-----------|--------|
| 0   | H196      | 865 | 15.5 | 15.522924 | 0      |
| 1   | H196      | 866 | 15.1 | 14.985832 | 0      |
| 2   | H196      | 867 | 14.8 | 14.667901 | 0      |
| 3   | H196      | 868 | 14.4 | 14.514592 | 0      |
| 4   | H196      | 869 | 14.2 | 14.035793 | 0      |
| ... | ...       | ... | ...  | ...       | ...    |
| 187 | H413      | 956 | 59.0 | 77.227905 | 1      |
| 188 | H413      | 957 | 58.0 | 80.589641 | 1      |
| 189 | H413      | 958 | 53.0 | 53.986834 | 1      |
| 190 | H413      | 959 | 38.0 | 36.749786 | 1      |
| 191 | H413      | 960 | 46.0 | 36.281225 | 1      |

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[3] }} />

The individual models we trained are saved, so calling `predict` returns
the predictions from every model trained.

<details>
<summary>Code</summary>

``` python
show_doc(LightGBMCV.predict)
```

</details>

------------------------------------------------------------------------

### LightGBMCV.predict {#lightgbmcv.predict}

> ``` text
>  LightGBMCV.predict (horizon:int,
>                      dynamic_dfs:Optional[List[pandas.core.frame.DataFrame
>                      ]]=None,
>                      before_predict_callback:Optional[Callable]=None,
>                      after_predict_callback:Optional[Callable]=None)
> ```

Compute predictions with each of the trained boosters.

|                         | **Type**      | **Default** | **Details**                                                                                                                                                                                                                                                          |
|------|------------------|-------------------------|-------------------------|
| horizon                 | int           |             | Number of periods to predict.                                                                                                                                                                                                                                        |
| dynamic_dfs             | Optional      | None        | Future values of the dynamic features, e.g. prices.                                                                                                                                                                                                                  |
| before_predict_callback | Optional      | None        | Function to call on the features before computing the predictions.<br> This function will take the input dataframe that will be passed to the model for predicting and should return a dataframe with the same structure.<br> The series identifier is on the index. |
| after_predict_callback  | Optional      | None        | Function to call on the predictions before updating the targets.<br> This function will take a pandas Series with the predictions and should return another one with the same structure.<br> The series identifier is on the index.                                  |
| **Returns**             | **DataFrame** |             | **Predictions for each serie and timestep, with one column per window.**                                                                                                                                                                                             |

<details>
<summary>Code</summary>

``` python
preds = cv.predict(horizon)
preds
```

</details>
<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[4] }} />

|     | unique_id | ds   | Booster0  | Booster1  |
|-----|-----------|------|-----------|-----------|
| 0   | H196      | 961  | 15.670252 | 15.848888 |
| 1   | H196      | 962  | 15.522924 | 15.697399 |
| 2   | H196      | 963  | 14.985832 | 15.166213 |
| 3   | H196      | 964  | 14.985832 | 14.723238 |
| 4   | H196      | 965  | 14.562152 | 14.451092 |
| ... | ...       | ...  | ...       | ...       |
| 187 | H413      | 1004 | 70.695242 | 65.917620 |
| 188 | H413      | 1005 | 66.216580 | 62.615788 |
| 189 | H413      | 1006 | 63.896573 | 67.848598 |
| 190 | H413      | 1007 | 46.922797 | 50.981950 |
| 191 | H413      | 1008 | 45.006541 | 42.752819 |

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[5] }} />

We can average these predictions and evaluate them.

<details>
<summary>Code</summary>

``` python
def evaluate_on_valid(preds):
    preds = preds.copy()
    preds['final_prediction'] = preds.drop(columns=['unique_id', 'ds']).mean(1)
    merged = preds.merge(valid, on=['unique_id', 'ds'])
    merged['abs_err'] = abs(merged['final_prediction'] - merged['y']) / merged['y']
    return merged.groupby('unique_id')['abs_err'].mean().mean()
```

</details>
<details>
<summary>Code</summary>

``` python
eval1 = evaluate_on_valid(preds)
eval1
```

</details>

``` text
0.11036194712311806
```

Now, since these series are hourly, maybe we can try to remove the daily
seasonality by taking the 168th (24 \* 7) difference, that is, substract
the value at the same hour from one week ago, thus our target will be
$z_t = y_{t} - y_{t-168}$. The features will be computed from this
target and when we predict they will be automatically re-applied.

<details>
<summary>Code</summary>

``` python
cv2 = LightGBMCV(
    freq=1,
    target_transforms=[Differences([24 * 7])],
    lags=[24 * (i+1) for i in range(7)],
)
hist2 = cv2.fit(train, **static_fit_config)
```

</details>

``` text
[LightGBM] [Info] Start training from score 0.519010
[10] mape: 0.089024
[20] mape: 0.090683
[30] mape: 0.092316
Early stopping at round 30
Using best iteration: 10
```

<details>
<summary>Code</summary>

``` python
assert hist2[-1][1] < hist[-1][1]
```

</details>

Nice! We achieve a better score in less iterations. Let’s see if this
improvement translates to the validation set as well.

<details>
<summary>Code</summary>

``` python
preds2 = cv2.predict(horizon)
eval2 = evaluate_on_valid(preds2)
eval2
```

</details>

``` text
0.08956665504570135
```

<details>
<summary>Code</summary>

``` python
assert eval2 < eval1
```

</details>

Great! Maybe we can try some lag transforms now. We’ll try the seasonal
rolling mean that averages the values “every season”, that is, if we set
`season_length=24` and `window_size=7` then we’ll average the value at
the same hour for every day of the week.

<details>
<summary>Code</summary>

``` python
cv3 = LightGBMCV(
    freq=1,
    target_transforms=[Differences([24 * 7])],
    lags=[24 * (i+1) for i in range(7)],
    lag_transforms={
        48: [(seasonal_rolling_mean, 24, 7)],
    },
)
hist3 = cv3.fit(train, **static_fit_config)
```

</details>

``` text
[LightGBM] [Info] Start training from score 0.273641
[10] mape: 0.086724
[20] mape: 0.088466
[30] mape: 0.090536
Early stopping at round 30
Using best iteration: 10
```

Seems like this is helping as well!

<details>
<summary>Code</summary>

``` python
assert hist3[-1][1] < hist2[-1][1]
```

</details>

Does this reflect on the validation set?

<details>
<summary>Code</summary>

``` python
preds3 = cv3.predict(horizon)
eval3 = evaluate_on_valid(preds3)
eval3
```

</details>

``` text
0.08961279023129345
```

Nice! mlforecast also supports date features, but in this case our time
column is made from integers so there aren’t many possibilites here. As
you can see this allows you to iterate faster and get better estimates
of the forecasting performance you can expect from your model.

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
test_eq(cv.find_best_iter([(0, 1), (1, 0.5)], 1), 1)
test_eq(cv.find_best_iter([(0, 1), (1, 0.5), (2, 0.6)], 1), 1)
test_eq(cv.find_best_iter([(0, 1), (1, 0.5), (2, 0.6), (3, 0.4)], 2), 3)
```

</details>

:::

If you’re doing hyperparameter tuning it’s useful to be able to run a
couple of iterations, assess the performance, and determine if this
particular configuration isn’t promising and should be discarded. For
example, [optuna](https://optuna.org/) has
[pruners](https://optuna.readthedocs.io/en/stable/reference/pruners.html)
that you can call with your current score and it decides if the trial
should be discarded. We’ll now show how to do that.

Since the CV requires a bit of setup, like the LightGBM datasets and the
internal features, we have this `setup` method.

<details>
<summary>Code</summary>

``` python
show_doc(LightGBMCV.setup)
```

</details>

------------------------------------------------------------------------

### LightGBMCV.setup {#lightgbmcv.setup}

> ``` text
>  LightGBMCV.setup (data:pandas.core.frame.DataFrame, n_windows:int,
>                    window_size:int, id_col:str='unique_id',
>                    time_col:str='ds', target_col:str='y',
>                    step_size:Optional[int]=None,
>                    params:Optional[Dict[str,Any]]=None,
>                    static_features:Optional[List[str]]=None,
>                    dropna:bool=True, keep_last_n:Optional[int]=None,
>                    weights:Optional[Sequence[float]]=None,
>                    metric:Union[str,Callable]='mape',
>                    input_size:Optional[int]=None)
> ```

Initialize internal data structures to iteratively train the boosters.
Use this before calling partial_fit.

|                 | **Type**       | **Default** | **Details**                                                                                                                |
|------|------------------|-------------------------|-------------------------|
| data            | DataFrame      |             | Series data in long format.                                                                                                |
| n_windows       | int            |             | Number of windows to evaluate.                                                                                             |
| window_size     | int            |             | Number of test periods in each window.                                                                                     |
| id_col          | str            | unique_id   | Column that identifies each serie.                                                                                         |
| time_col        | str            | ds          | Column that identifies each timestep, its values can be timestamps or integers.                                            |
| target_col      | str            | y           | Column that contains the target.                                                                                           |
| step_size       | Optional       | None        | Step size between each cross validation window. If None it will be equal to `window_size`.                                 |
| params          | Optional       | None        | Parameters to be passed to the LightGBM Boosters.                                                                          |
| static_features | Optional       | None        | Names of the features that are static and will be repeated when forecasting.                                               |
| dropna          | bool           | True        | Drop rows with missing values produced by the transformations.                                                             |
| keep_last_n     | Optional       | None        | Keep only these many records from each serie for the forecasting step. Can save time and memory if your features allow it. |
| weights         | Optional       | None        | Weights to multiply the metric of each window. If None, all windows have the same weight.                                  |
| metric          | Union          | mape        | Metric used to assess the performance of the models and perform early stopping.                                            |
| input_size      | Optional       | None        | Maximum training samples per serie in each window. If None, will use an expanding window.                                  |
| **Returns**     | **LightGBMCV** |             | **CV object with internal data structures for partial_fit.**                                                               |

<details>
<summary>Code</summary>

``` python
cv4 = LightGBMCV(
    freq=1,
    lags=[24 * (i+1) for i in range(7)],
)
cv4.setup(
    train,
    n_windows=2,
    window_size=horizon,
    params={'verbose': -1},
)
```

</details>

``` text
LightGBMCV(freq=1, lag_features=['lag24', 'lag48', 'lag72', 'lag96', 'lag120', 'lag144', 'lag168'], date_features=[], num_threads=1, bst_threads=8)
```

Once we have this we can call `partial_fit` to only train for some
iterations and return the score of the forecast window.

<details>
<summary>Code</summary>

``` python
show_doc(LightGBMCV.partial_fit)
```

</details>

------------------------------------------------------------------------

### LightGBMCV.partial_fit {#lightgbmcv.partial_fit}

> ``` text
>  LightGBMCV.partial_fit (num_iterations:int,
>                          before_predict_callback:Optional[Callable]=None,
>                          after_predict_callback:Optional[Callable]=None)
> ```

Train the boosters for some iterations.

|                         | **Type**  | **Default** | **Details**                                                                                                                                                                                                                                                          |
|------|------------------|-------------------------|-------------------------|
| num_iterations          | int       |             | Number of boosting iterations to run                                                                                                                                                                                                                                 |
| before_predict_callback | Optional  | None        | Function to call on the features before computing the predictions.<br> This function will take the input dataframe that will be passed to the model for predicting and should return a dataframe with the same structure.<br> The series identifier is on the index. |
| after_predict_callback  | Optional  | None        | Function to call on the predictions before updating the targets.<br> This function will take a pandas Series with the predictions and should return another one with the same structure.<br> The series identifier is on the index.                                  |
| **Returns**             | **float** |             | **Weighted metric after training for num_iterations.**                                                                                                                                                                                                               |

<details>
<summary>Code</summary>

``` python
score = cv4.partial_fit(10)
score
```

</details>

``` text
[LightGBM] [Info] Start training from score 51.745632
```

``` text
0.5906900462828166
```

This is equal to the first evaluation from our first example.

<details>
<summary>Code</summary>

``` python
assert hist[0][1] == score
```

</details>

We can now use this score to decide if this configuration is promising.
If we want to we can train some more iterations.

<details>
<summary>Code</summary>

``` python
score2 = cv4.partial_fit(20)
```

</details>

This is now equal to our third metric from the first example, since this
time we trained for 20 iterations.

<details>
<summary>Code</summary>

``` python
assert hist[2][1] == score2
```

</details>
<details>
<summary>Code</summary>

``` python
%%capture
#| hide
# test we don't need dynamic_dfs
from mlforecast.utils import generate_daily_series, generate_prices_for_series

def before_predict_callback(df):
    assert not df['price'].isnull().any()
    return df

dynamic_series = generate_daily_series(100, equal_ends=True, n_static_features=2, static_as_categorical=False)
dynamic_series = dynamic_series.rename(columns={'static_1': 'product_id'})
prices_catalog = generate_prices_for_series(dynamic_series)
series_with_prices = dynamic_series.merge(prices_catalog, how='left')
cv = LightGBMCV(freq='D', lags=[24])
_ = cv.fit(
    series_with_prices,
    n_windows=2,
    window_size=5,
    params={'verbosity': -1},
    static_features=['static_0', 'product_id'],
    verbose_eval=False,
    before_predict_callback=before_predict_callback,
)
```

</details>

### Using a custom metric {#using-a-custom-metric}

The built-in metrics are MAPE and RMSE, which are computed by serie and
then averaged across all series. If you want to do something different
or use a different metric entirely, you can define your own metric like
the following:

<details>
<summary>Code</summary>

``` python
def weighted_mape(
    y_true: pd.Series,
    y_pred: pd.Series,
    ids: pd.Series,
    dates: pd.Series,
):
    """Weighs the MAPE by the magnitude of the series values"""
    abs_pct_err = abs(y_true - y_pred) / abs(y_true)
    mape_by_serie = abs_pct_err.groupby(ids).mean()
    totals_per_serie = y_pred.groupby(ids).sum()
    series_weights = totals_per_serie / totals_per_serie.sum()
    return (mape_by_serie * series_weights).sum()
```

</details>
<details>
<summary>Code</summary>

``` python
_ = LightGBMCV(
    freq=1,
    lags=[24 * (i+1) for i in range(7)],
).fit(
    train,
    n_windows=2,
    window_size=horizon,
    params={'verbose': -1},
    metric=weighted_mape,
)
```

</details>

``` text
[LightGBM] [Info] Start training from score 51.745632
[10] weighted_mape: 0.480353
[20] weighted_mape: 0.218670
[30] weighted_mape: 0.161706
[40] weighted_mape: 0.149992
[50] weighted_mape: 0.149024
[60] weighted_mape: 0.148496
Early stopping at round 60
Using best iteration: 60
```

