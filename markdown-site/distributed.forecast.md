---
title: DistributedMLForecast
---

export const quartoRawHtml =
[`<div><strong>Dask DataFrame Structure:</strong></div>
<div>
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
</div>
<div>Dask Name: assign, 5 graph layers</div>`,`<div><strong>Dask DataFrame Structure:</strong></div>
<div>
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
</div>
<div>Dask Name: map, 17 graph layers</div>`,`<div>
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
<p>20185 rows × 3 columns</p>
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
<p>700 rows × 4 columns</p>
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
<p>1400 rows × 4 columns</p>
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
<p>2800 rows × 6 columns</p>
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
<p>1400 rows × 4 columns</p>
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
<p>2800 rows × 6 columns</p>
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

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
import warnings

from fastcore.test import test_warns, test_eq, test_ne
from nbdev import show_doc
from sklearn import set_config
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
warnings.simplefilter('ignore', FutureWarning)
set_config(display='text')
```

</details>

:::

> Distributed pipeline encapsulation

**This interface is only tested on Linux**

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

<details>
<summary>Code</summary>

``` python
import copy
from collections import namedtuple
from typing import Any, Callable, Iterable, List, Optional

import cloudpickle
try:
    import dask.dataframe as dd
    DASK_INSTALLED = True
except ModuleNotFoundError:
    DASK_INSTALLED = False
import fugue
import fugue.api as fa
import pandas as pd
try:
    from pyspark.ml.feature import VectorAssembler
    from pyspark.sql import DataFrame as SparkDataFrame
    SPARK_INSTALLED = True
except ModuleNotFoundError:
    SPARK_INSTALLED = False
try:
    from lightgbm_ray import RayDMatrix
    from ray.data import Dataset as RayDataset
    RAY_INSTALLED = True
except ModuleNotFoundError:
    RAY_INSTALLED = False
from sklearn.base import clone

from mlforecast.core import (
    DateFeature,
    Freq,
    LagTransforms,
    Lags,
    TimeSeries,
    _name_models,
)
from mlforecast.utils import single_split
from mlforecast.target_transforms import BaseTargetTransform
```

</details>

:::

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’ 6=‘i’}

<details>
<summary>Code</summary>

``` python
WindowInfo = namedtuple('WindowInfo', ['n_windows', 'window_size', 'step_size', 'i_window', 'input_size'])
```

</details>

:::

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

<details>
<summary>Code</summary>

``` python
class DistributedMLForecast:
    """Multi backend distributed pipeline"""
    
    def __init__(
        self,
        models,
        freq: Optional[Freq] = None,
        lags: Optional[Lags] = None,
        lag_transforms: Optional[LagTransforms] = None,
        date_features: Optional[Iterable[DateFeature]] = None,
        differences: Optional[Iterable[int]] = None,
        num_threads: int = 1,
        target_transforms: Optional[List[BaseTargetTransform]] = None,        
        engine = None,
        num_partitions: Optional[int] = None,        
    ):
        """Create distributed forecast object

        Parameters
        ----------
        models : regressor or list of regressors
            Models that will be trained and used to compute the forecasts.
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
        engine : fugue execution engine, optional (default=None)
            Dask Client, Spark Session, etc to use for the distributed computation.
            If None will infer depending on the input type.
        num_partitions: number of data partitions to use, optional (default=None)
            If None, the default partitions provided by the AnyDataFrame used
            by the `fit` and `cross_validation` methods will be used. If a Ray
            Dataset is provided and `num_partitions` is None, the partitioning
            will be done by the `id_col`.
        """        
        if not isinstance(models, dict) and not isinstance(models, list):
            models = [models]
        if isinstance(models, list):
            model_names = _name_models([m.__class__.__name__ for m in models])
            models_with_names = dict(zip(model_names, models))
        else:
            models_with_names = models
        self.models = models_with_names
        self._base_ts = TimeSeries(
            freq=freq,
            lags=lags,
            lag_transforms=lag_transforms,
            date_features=date_features,
            differences=differences,
            num_threads=num_threads,
            target_transforms=target_transforms,
        )
        self.engine = engine
        self.num_partitions = num_partitions
        
    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(models=[{", ".join(self.models.keys())}], '
            f"freq={self._base_ts.freq}, "
            f"lag_features={list(self._base_ts.transforms.keys())}, "
            f"date_features={self._base_ts.date_features}, "
            f"num_threads={self._base_ts.num_threads}, "
            f"engine={self.engine})"
        )

    @staticmethod
    def _preprocess_partition(
        part: pd.DataFrame,
        base_ts: TimeSeries,        
        id_col: str,
        time_col: str,
        target_col: str,
        static_features: Optional[List[str]] = None,
        dropna: bool = True,
        keep_last_n: Optional[int] = None,
        window_info: Optional[WindowInfo] = None,
        fit_ts_only: bool = False,
    ) -> List[List[Any]]:
        ts = copy.deepcopy(base_ts)
        if fit_ts_only:
            ts._fit(
                part,
                id_col=id_col,
                time_col=time_col,
                target_col=target_col,
                static_features=static_features,
                keep_last_n=keep_last_n,                
            )
            return [[cloudpickle.dumps(ts), cloudpickle.dumps(None), cloudpickle.dumps(None)]]        
        if window_info is None:
            train = part
            valid = None
        else:
            max_dates = part.groupby(id_col, observed=True)[time_col].transform('max')
            cutoffs, train_mask, valid_mask = single_split(
                part,
                i_window=window_info.i_window,
                n_windows=window_info.n_windows,
                window_size=window_info.window_size,
                id_col=id_col,
                time_col=time_col,
                freq=base_ts.freq,
                max_dates=max_dates,
                step_size=window_info.step_size,
                input_size=window_info.input_size,
            )
            train = part[train_mask]
            valid_keep_cols = part.columns
            if static_features is not None:
                valid_keep_cols.drop(static_features)
            valid = part.loc[valid_mask, valid_keep_cols].merge(cutoffs, on=id_col)
        transformed = ts.fit_transform(
            train,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
            static_features=static_features,
            dropna=dropna,
            keep_last_n=keep_last_n,
        )
        return [[cloudpickle.dumps(ts), cloudpickle.dumps(transformed), cloudpickle.dumps(valid)]]

    @staticmethod
    def _retrieve_df(items: List[List[Any]]) -> Iterable[pd.DataFrame]:
        for _, serialized_train, _ in items:
            yield cloudpickle.loads(serialized_train)
            
    def _preprocess_partitions(
        self,
        data: fugue.AnyDataFrame,
        id_col: str,
        time_col: str,
        target_col: str,
        static_features: Optional[List[str]] = None,
        dropna: bool = True,
        keep_last_n: Optional[int] = None,
        window_info: Optional[WindowInfo] = None,
        fit_ts_only: bool = False,
    ) -> List[Any]:
        if self.num_partitions:
            partition = dict(by=id_col, num=self.num_partitions, algo='coarse')
        elif RAY_INSTALLED and isinstance(data, RayDataset): # num partitions is None but data is a RayDataset
            # We need to add this because 
            # currently ray doesnt support partitioning a Dataset
            # based on a column.
            # If a Dataset is partitioned using `.repartition(num_partitions)`
            # we will have akward results.
            partition = dict(by=id_col)
        else:
            partition = None
        return fa.transform(
            data,
            DistributedMLForecast._preprocess_partition,
            params={
                'base_ts': self._base_ts,
                'id_col': id_col,
                'time_col': time_col,
                'target_col': target_col,
                'static_features': static_features,
                'dropna': dropna,
                'keep_last_n': keep_last_n,
                'window_info': window_info,
                'fit_ts_only': fit_ts_only,
            },
            schema='ts:binary,train:binary,valid:binary',
            engine=self.engine,
            as_fugue=True,
            partition=partition,
        )        

    def _preprocess(
        self,
        data: fugue.AnyDataFrame,
        id_col: str,
        time_col: str,
        target_col: str,
        static_features: Optional[List[str]] = None,
        dropna: bool = True,
        keep_last_n: Optional[int] = None,
        window_info: Optional[WindowInfo] = None,
    ) -> fugue.AnyDataFrame:
        self.id_col = id_col
        self.time_col = time_col
        self.target_col = target_col
        self.static_features = static_features
        self.dropna = dropna
        self.keep_last_n = keep_last_n
        self.partition_results = self._preprocess_partitions(
            data=data,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
            static_features=static_features,
            dropna=dropna,
            keep_last_n=keep_last_n,
            window_info=window_info,
        )
        base_schema = str(fa.get_schema(data))
        features_schema = ','.join(f'{feat}:double' for feat in self._base_ts.features)
        res = fa.transform(
            self.partition_results,
            DistributedMLForecast._retrieve_df,
            schema=f'{base_schema},{features_schema}',
            engine=self.engine,
        )
        return fa.get_native_as_df(res)
    
    def preprocess(
        self,
        data: fugue.AnyDataFrame,
        id_col: str = 'unique_id',
        time_col: str = 'ds',
        target_col: str = 'y',
        static_features: Optional[List[str]] = None,
        dropna: bool = True,
        keep_last_n: Optional[int] = None,
    ) -> fugue.AnyDataFrame:
        """Add the features to `data`.

        Parameters
        ----------
        data : dask or spark DataFrame.
            Series data in long format.
        id_col : str (default='unique_id')
            Column that identifies each serie.
        time_col : str (default='ds')
            Column that identifies each timestep, its values can be timestamps or integers.
        target_col : str (default='y')
            Column that contains the target.
        static_features : list of str, optional (default=None)
            Names of the features that are static and will be repeated when forecasting.
        dropna : bool (default=True)
            Drop rows with missing values produced by the transformations.
        keep_last_n : int, optional (default=None)
            Keep only these many records from each serie for the forecasting step. Can save time and memory if your features allow it.

        Returns
        -------
        result : same type as input
            data with added features.
        """        
        return self._preprocess(
            data,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
            static_features=static_features,
            dropna=dropna,
            keep_last_n=keep_last_n,
        )
    
    def _fit(
        self,
        data: fugue.AnyDataFrame,
        id_col: str,
        time_col: str,
        target_col: str,
        static_features: Optional[List[str]] = None,
        dropna: bool = True,
        keep_last_n: Optional[int] = None,
        window_info: Optional[WindowInfo] = None,
    ) -> 'DistributedMLForecast':
        prep = self._preprocess(
            data,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
            static_features=static_features,
            dropna=dropna,
            keep_last_n=keep_last_n,
            window_info=window_info,
        )
        features = [x for x in fa.get_column_names(prep) if x not in {id_col, time_col, target_col}]
        self.models_ = {}
        if SPARK_INSTALLED and isinstance(data, SparkDataFrame):
            featurizer = VectorAssembler(inputCols=features, outputCol="features")
            train_data = featurizer.transform(prep)[target_col, "features"]
            for name, model in self.models.items():
                trained_model = model._pre_fit(target_col).fit(train_data)
                self.models_[name] = model.extract_local_model(trained_model)
        elif DASK_INSTALLED and isinstance(data, dd.DataFrame):
            X, y = prep[features], prep[target_col]
            for name, model in self.models.items():
                trained_model = clone(model).fit(X, y)
                self.models_[name] = trained_model.model_
        elif RAY_INSTALLED and isinstance(data, RayDataset):
            X = RayDMatrix(
                prep.select_columns(cols=features + [target_col]),
                label=target_col,
            )
            for name, model in self.models.items():
                trained_model = clone(model).fit(X, y=None)
                self.models_[name] = trained_model.model_
        else:
            raise NotImplementedError('Only spark, dask, and ray engines are supported.')
        return self
    
    def fit(
        self,
        data: fugue.AnyDataFrame,
        id_col: str = 'unique_id',
        time_col: str = 'ds',
        target_col: str = 'y',
        static_features: Optional[List[str]] = None,
        dropna: bool = True,
        keep_last_n: Optional[int] = None,        
    ) -> 'DistributedMLForecast':
        """Apply the feature engineering and train the models.

        Parameters
        ----------
        data : dask or spark DataFrame
            Series data in long format.
        id_col : str (default='unique_id')
            Column that identifies each serie.
        time_col : str (default='ds')
            Column that identifies each timestep, its values can be timestamps or integers.
        target_col : str (default='y')
            Column that contains the target.
        static_features : list of str, optional (default=None)
            Names of the features that are static and will be repeated when forecasting.
        dropna : bool (default=True)
            Drop rows with missing values produced by the transformations.
        keep_last_n : int, optional (default=None)
            Keep only these many records from each serie for the forecasting step. Can save time and memory if your features allow it.

        Returns
        -------
        self : DistributedMLForecast
            Forecast object with series values and trained models.
        """        
        return self._fit(
            data,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
            static_features=static_features,
            dropna=dropna,
            keep_last_n=keep_last_n,
        )

    @staticmethod
    def _predict(
        items: List[List[Any]],
        models,
        horizon,
        dynamic_dfs=None,
        before_predict_callback=None,
        after_predict_callback=None,
    ) -> Iterable[pd.DataFrame]:
        for serialized_ts, _, serialized_valid in items:
            valid = cloudpickle.loads(serialized_valid)
            ts = cloudpickle.loads(serialized_ts)
            if valid is not None:
                dynamic_features = valid.columns.drop(
                    [ts.id_col, ts.time_col, ts.target_col]
                )
                if not dynamic_features.empty:
                    dynamic_dfs = [valid.drop(columns=ts.target_col)]
            res = ts.predict(
                models=models,
                horizon=horizon,
                dynamic_dfs=dynamic_dfs,
                before_predict_callback=before_predict_callback,
                after_predict_callback=after_predict_callback,
            )
            if valid is not None:
                res = res.merge(valid, how='left')
            yield res
            
    def _get_predict_schema(self) -> str:
        model_names = self.models.keys()
        models_schema = ','.join(f'{model_name}:double' for model_name in model_names)
        schema = f'{self.id_col}:string,{self.time_col}:datetime,' + models_schema
        return schema

    def predict(
        self,
        horizon: int,
        dynamic_dfs: Optional[List[pd.DataFrame]] = None,
        before_predict_callback: Optional[Callable] = None,
        after_predict_callback: Optional[Callable] = None,
        new_data: Optional[fugue.AnyDataFrame] = None,
    ) -> fugue.AnyDataFrame:
        """Compute the predictions for the next `horizon` steps.

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
        new_data : dask or spark DataFrame, optional (default=None)
            Series data of new observations for which forecasts are to be generated.
                This dataframe should have the same structure as the one used to fit the model, including any features and time series data.
                If `new_data` is not None, the method will generate forecasts for the new observations.                

        Returns
        -------
        result : dask, spark or ray DataFrame
            Predictions for each serie and timestep, with one column per model.
        """        
        if new_data is not None:
            partition_results = self._preprocess_partitions(
                data=new_data,
                id_col=self.id_col,
                time_col=self.time_col,
                target_col=self.target_col,
                static_features=self.static_features,
                dropna=self.dropna,
                keep_last_n=self.keep_last_n,
                fit_ts_only=True,
            )
        else:
            partition_results = self.partition_results
        schema = self._get_predict_schema()
        res = fa.transform(
            partition_results,
            DistributedMLForecast._predict,
            params={
                'models': self.models_,
                'horizon': horizon,
                'dynamic_dfs': dynamic_dfs,
                'before_predict_callback': before_predict_callback,
                'after_predict_callback': after_predict_callback,
            },
            schema=schema,
            engine=self.engine,
        )
        return fa.get_native_as_df(res)

    def cross_validation(
        self,
        data: fugue.AnyDataFrame,
        n_windows: int,
        window_size: int,
        id_col: str = 'unique_id',
        time_col: str = 'ds',
        target_col: str = 'y',
        step_size: Optional[int] = None,
        static_features: Optional[List[str]] = None,
        dropna: bool = True,
        keep_last_n: Optional[int] = None,
        refit: bool = True,
        before_predict_callback: Optional[Callable] = None,
        after_predict_callback: Optional[Callable] = None,
        input_size: Optional[int] = None,        
    ) -> fugue.AnyDataFrame:
        """Perform time series cross validation.
        Creates `n_windows` splits where each window has `window_size` test periods,
        trains the models, computes the predictions and merges the actuals.

        Parameters
        ----------
        data : dask, spark or ray DataFrame
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
        static_features : list of str, optional (default=None)
            Names of the features that are static and will be repeated when forecasting.
        dropna : bool (default=True)
            Drop rows with missing values produced by the transformations.
        keep_last_n : int, optional (default=None)
            Keep only these many records from each serie for the forecasting step. Can save time and memory if your features allow it.
        refit : bool (default=True)
            Retrain model for each cross validation window.
            If False, the models are trained at the beginning and then used to predict each window.            
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
        result : dask, spark or ray DataFrame
            Predictions for each window with the series id, timestamp, target value and predictions from each model.
        """            
        self.cv_models_ = []
        results = []
        for i in range(n_windows):
            window_info = WindowInfo(n_windows, window_size, step_size, i, input_size)            
            if refit or i == 0:
                self._fit(
                    data,
                    id_col=id_col,
                    time_col=time_col,
                    target_col=target_col,
                    static_features=static_features,
                    dropna=dropna,
                    keep_last_n=keep_last_n,
                    window_info=window_info,
                )
                self.cv_models_.append(self.models_)
                partition_results = self.partition_results
            elif not refit:
                partition_results = self._preprocess_partitions(
                    data=data,
                    id_col=id_col,
                    time_col=time_col,
                    target_col=target_col,
                    static_features=static_features,
                    dropna=dropna,
                    keep_last_n=keep_last_n,
                    window_info=window_info,
                )
            schema = self._get_predict_schema() + f',cutoff:datetime,{self.target_col}:double'
            preds = fa.transform(
                partition_results,
                DistributedMLForecast._predict,
                params={
                    'models': self.models_,
                    'horizon': window_size,
                    'before_predict_callback': before_predict_callback,
                    'after_predict_callback': after_predict_callback,
                },
                schema=schema,
                engine=self.engine,
            )
            results.append(fa.get_native_as_df(preds))
        if len(results) == 1:
            return results[0]
        if len(results) == 2:
            return fa.union(results[0], results[1])
        return fa.union(results[0], results[1], results[2:])
```

</details>

:::

<details>
<summary>Code</summary>

``` python
show_doc(DistributedMLForecast)
```

</details>

------------------------------------------------------------------------

### DistributedMLForecast {#distributedmlforecast}

> ``` text
>  DistributedMLForecast (models,
>                         freq:Union[int,str,pandas._libs.tslibs.offsets.Bas
>                         eOffset,NoneType]=None,
>                         lags:Optional[Iterable[int]]=None, lag_transforms:
>                         Optional[Dict[int,List[Union[Callable,Tuple[Callab
>                         le,Any]]]]]=None, date_features:Optional[Iterable[
>                         Union[str,Callable]]]=None,
>                         differences:Optional[Iterable[int]]=None,
>                         num_threads:int=1, target_transforms:Optional[List
>                         [mlforecast.target_transforms.BaseTargetTransform]
>                         ]=None, engine=None,
>                         num_partitions:Optional[int]=None)
> ```

Multi backend distributed pipeline

The `DistributedMLForecast` class is a high level abstraction that
encapsulates all the steps in the pipeline (preprocessing, fitting the
model and computing predictions) and applies them in a distributed way.

The different things that you need to use `DistributedMLForecast` (as
opposed to `MLForecast`) are:

1.  You need to set up a cluster. We currently support dask and spark
    (ray is on the roadmap).
2.  Your data needs to be a distributed collection. We currently support
    dask and spark dataframes.
3.  You need to use a model that implements distributed training in your
    framework of choice, e.g. SynapseML for LightGBM in spark.

<details>
<summary>Code</summary>

``` python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dask.distributed import Client
from window_ops.expanding import expanding_mean
from window_ops.rolling import rolling_mean

from mlforecast.target_transforms import Differences
from mlforecast.utils import backtest_splits, generate_daily_series, generate_prices_for_series
from mlforecast.distributed.models.dask.lgb import DaskLGBMForecast
from mlforecast.distributed.models.dask.xgb import DaskXGBForecast
```

</details>

## Dask {#dask}

### Client setup {#client-setup}

<details>
<summary>Code</summary>

``` python
client = Client(n_workers=2, threads_per_worker=1)
```

</details>

Here we define a client that connects to a
`dask.distributed.LocalCluster`, however it could be any other kind of
cluster.

### Data setup {#data-setup}

For dask, the data must be a `dask.dataframe.DataFrame`. You need to
make sure that each time serie is only in one partition and it is
recommended that you have as many partitions as you have workers. If you
have more partitions than workers make sure to set `num_threads=1` to
avoid having nested parallelism.

The required input format is the same as for `MLForecast`, except that
it’s a `dask.dataframe.DataFrame` instead of a `pandas.Dataframe`.

<details>
<summary>Code</summary>

``` python
series = generate_daily_series(100, n_static_features=2, equal_ends=True, static_as_categorical=False)
npartitions = 10
partitioned_series = dd.from_pandas(series.set_index('unique_id'), npartitions=npartitions)  # make sure we split by the id_col
partitioned_series = partitioned_series.map_partitions(lambda df: df.reset_index())
partitioned_series['unique_id'] = partitioned_series['unique_id'].astype(str)  # can't handle categoricals atm
partitioned_series
```

</details>
<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[0] }} />

|                | unique_id | ds               | y       | static_0 | static_1 |
|----------------|-----------|------------------|---------|----------|----------|
| npartitions=10 |           |                  |         |          |          |
| id_00          | object    | datetime64\[ns\] | float64 | int64    | int64    |
| id_10          | ...       | ...              | ...     | ...      | ...      |
| ...            | ...       | ...              | ...     | ...      | ...      |
| id_89          | ...       | ...              | ...     | ...      | ...      |
| id_99          | ...       | ...              | ...     | ...      | ...      |

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[1] }} />

### Models {#models}

In order to perform distributed forecasting, we need to use a model that
is able to train in a distributed way using `dask`. The current
implementations are in `DaskLGBMForecast` and `DaskXGBForecast` which
are just wrappers around the native implementations.

<details>
<summary>Code</summary>

``` python
models = [DaskXGBForecast(random_state=0), DaskLGBMForecast(random_state=0)]
```

</details>

### Training {#training}

Once we have our models we instantiate a `DistributedMLForecast` object
defining our features.

<details>
<summary>Code</summary>

``` python
fcst = DistributedMLForecast(
    models=models,
    freq='D',
    lags=[7],
    lag_transforms={
        1: [expanding_mean],
        7: [(rolling_mean, 14)]
    },
    date_features=['dayofweek', 'month'],
    num_threads=1,
    engine=client,
)
fcst
```

</details>

``` text
DistributedMLForecast(models=[DaskXGBForecast, DaskLGBMForecast], freq=<Day>, lag_features=['lag7', 'expanding_mean_lag1', 'rolling_mean_lag7_window_size14'], date_features=['dayofweek', 'month'], num_threads=1, engine=<Client: 'tcp://127.0.0.1:42893' processes=2 threads=2, memory=15.50 GiB>)
```

Here where we say that:

-   Our series have daily frequency.
-   We want to use lag 7 as a feature
-   We want the lag transformations to be:
    -   expanding mean of the lag 1
    -   rolling mean of the lag 7 over a window of size 14
-   We want to use dayofweek and month as date features.
-   We want to perform the preprocessing and the forecasting steps using
    1 thread, because we have 10 partitions and 2 workers.

From this point we have two options:

1.  Compute the features and fit our models.
2.  Compute the features and get them back as a dataframe to do some
    custom splitting or adding additional features, then training the
    models.

### 1. Using all the data {#using-all-the-data}

<details>
<summary>Code</summary>

``` python
show_doc(DistributedMLForecast.fit, title_level=2)
```

</details>

------------------------------------------------------------------------

## DistributedMLForecast.fit {#distributedmlforecast.fit}

> ``` text
>  DistributedMLForecast.fit (data:~AnyDataFrame, id_col:str='unique_id',
>                             time_col:str='ds', target_col:str='y',
>                             static_features:Optional[List[str]]=None,
>                             dropna:bool=True,
>                             keep_last_n:Optional[int]=None)
> ```

Apply the feature engineering and train the models.

|                 | **Type**                  | **Default** | **Details**                                                                                                                |
|------|------------------|-------------------------|-------------------------|
| data            | AnyDataFrame              |             | Series data in long format.                                                                                                |
| id_col          | str                       | unique_id   | Column that identifies each serie.                                                                                         |
| time_col        | str                       | ds          | Column that identifies each timestep, its values can be timestamps or integers.                                            |
| target_col      | str                       | y           | Column that contains the target.                                                                                           |
| static_features | Optional                  | None        | Names of the features that are static and will be repeated when forecasting.                                               |
| dropna          | bool                      | True        | Drop rows with missing values produced by the transformations.                                                             |
| keep_last_n     | Optional                  | None        | Keep only these many records from each serie for the forecasting step. Can save time and memory if your features allow it. |
| **Returns**     | **DistributedMLForecast** |             | **Forecast object with series values and trained models.**                                                                 |

Calling `fit` on our data computes the features independently for each
partition and performs distributed training.

<details>
<summary>Code</summary>

``` python
fcst.fit(partitioned_series)
```

</details>

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
# function to test the partition_results data
# has the right size
def test_partition_results_size(fcst_object, expected_n_partitions):
    test_eq(
        fa.get_num_partitions(fcst_object.partition_results),
        expected_n_partitions,
    )
    test_eq(
        fa.count(fcst_object.partition_results),
        expected_n_partitions,
    )
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
test_partition_results_size(fcst, npartitions)
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
# test num_partitions works properly
num_partitions_test = 4
test_dd = dd.from_pandas(series, npartitions=num_partitions_test) # In this case we dont have to specify the column
test_dd['unique_id'] = test_dd['unique_id'].astype(str)
fcst_np = DistributedMLForecast(
    models=models,
    freq='D',
    lags=[7],
    lag_transforms={
        1: [expanding_mean],
        7: [(rolling_mean, 14)]
    },
    date_features=['dayofweek', 'month'],
    num_threads=1,
    engine=client,
    num_partitions=num_partitions_test
)
fcst_np.fit(test_dd)
test_partition_results_size(fcst_np, num_partitions_test)
preds_np = fcst_np.predict(7).compute().sort_values(['unique_id', 'ds']).reset_index(drop=True)
preds = fcst.predict(7).compute().sort_values(['unique_id', 'ds']).reset_index(drop=True)
pd.testing.assert_frame_equal(
    preds[['unique_id', 'ds']], 
    preds_np[['unique_id', 'ds']], 
)
```

</details>

:::

### Forecasting {#forecasting}

<details>
<summary>Code</summary>

``` python
show_doc(DistributedMLForecast.predict, title_level=2)
```

</details>

------------------------------------------------------------------------

## DistributedMLForecast.predict {#distributedmlforecast.predict}

> ``` text
>  DistributedMLForecast.predict (horizon:int,
>                                 dynamic_dfs:Optional[List[pandas.core.fram
>                                 e.DataFrame]]=None, before_predict_callbac
>                                 k:Optional[Callable]=None, after_predict_c
>                                 allback:Optional[Callable]=None,
>                                 new_data:Optional[~AnyDataFrame]=None)
> ```

Compute the predictions for the next `horizon` steps.

|                         | **Type**         | **Default** | **Details**                                                                                                                                                                                                                                                                                           |
|------|------------------|-------------------------|-------------------------|
| horizon                 | int              |             | Number of periods to predict.                                                                                                                                                                                                                                                                         |
| dynamic_dfs             | Optional         | None        | Future values of the dynamic features, e.g. prices.                                                                                                                                                                                                                                                   |
| before_predict_callback | Optional         | None        | Function to call on the features before computing the predictions.<br> This function will take the input dataframe that will be passed to the model for predicting and should return a dataframe with the same structure.<br> The series identifier is on the index.                                  |
| after_predict_callback  | Optional         | None        | Function to call on the predictions before updating the targets.<br> This function will take a pandas Series with the predictions and should return another one with the same structure.<br> The series identifier is on the index.                                                                   |
| new_data                | Optional         | None        | Series data of new observations for which forecasts are to be generated.<br> This dataframe should have the same structure as the one used to fit the model, including any features and time series data.<br> If `new_data` is not None, the method will generate forecasts for the new observations. |
| **Returns**             | **AnyDataFrame** |             | **Predictions for each serie and timestep, with one column per model.**                                                                                                                                                                                                                               |

Once we have our fitted models we can compute the predictions for the
next 7 timesteps.

<details>
<summary>Code</summary>

``` python
preds = fcst.predict(7)
preds
```

</details>
<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[2] }} />

|                | unique_id | ds               | DaskXGBForecast | DaskLGBMForecast |
|----------------|-----------|------------------|-----------------|------------------|
| npartitions=10 |           |                  |                 |                  |
| id_00          | object    | datetime64\[ns\] | float64         | float64          |
| id_10          | ...       | ...              | ...             | ...              |
| ...            | ...       | ...              | ...             | ...              |
| id_89          | ...       | ...              | ...             | ...              |
| id_99          | ...       | ...              | ...             | ...              |

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[3] }} />

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
preds = preds.compute()
preds2 = fcst.predict(7).compute()
preds3 = fcst.predict(7, new_data=partitioned_series).compute()
pd.testing.assert_frame_equal(preds, preds2)
pd.testing.assert_frame_equal(preds, preds3)
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
non_std_series = partitioned_series.copy()
non_std_series['ds'] = non_std_series.map_partitions(lambda part: part.groupby('unique_id').cumcount())
non_std_series = non_std_series.rename(columns={'ds': 'time', 'y': 'value', 'unique_id': 'some_id'})
flow_params = dict(
    models=[DaskXGBForecast(random_state=0)],
    lags=[7],
    lag_transforms={
        1: [expanding_mean],
        7: [(rolling_mean, 14)]
    },
    num_threads=1,
)
fcst = DistributedMLForecast(freq='D', **flow_params)
fcst.fit(partitioned_series)
preds = fcst.predict(7).compute()
fcst2 = DistributedMLForecast(**flow_params)
fcst2.preprocess(non_std_series, id_col='some_id', time_col='time', target_col='value')
fcst2.models_ = fcst.models_  # distributed training can end up with different fits
non_std_preds = fcst2.predict(7).compute()
pd.testing.assert_frame_equal(
    preds.drop(columns='ds'),
    non_std_preds.drop(columns='time').rename(columns={'some_id': 'unique_id'})
)
```

</details>

:::

### 2. Preprocess and train {#preprocess-and-train}

If we only want to perform the preprocessing step we call `preprocess`
with our data.

<details>
<summary>Code</summary>

``` python
show_doc(DistributedMLForecast.preprocess, title_level=2)
```

</details>

------------------------------------------------------------------------

## DistributedMLForecast.preprocess {#distributedmlforecast.preprocess}

> ``` text
>  DistributedMLForecast.preprocess (data:~AnyDataFrame,
>                                    id_col:str='unique_id',
>                                    time_col:str='ds', target_col:str='y', 
>                                    static_features:Optional[List[str]]=Non
>                                    e, dropna:bool=True,
>                                    keep_last_n:Optional[int]=None)
> ```

Add the features to `data`.

|                 | **Type**         | **Default** | **Details**                                                                                                                |
|------|------------------|-------------------------|-------------------------|
| data            | AnyDataFrame     |             | Series data in long format.                                                                                                |
| id_col          | str              | unique_id   | Column that identifies each serie.                                                                                         |
| time_col        | str              | ds          | Column that identifies each timestep, its values can be timestamps or integers.                                            |
| target_col      | str              | y           | Column that contains the target.                                                                                           |
| static_features | Optional         | None        | Names of the features that are static and will be repeated when forecasting.                                               |
| dropna          | bool             | True        | Drop rows with missing values produced by the transformations.                                                             |
| keep_last_n     | Optional         | None        | Keep only these many records from each serie for the forecasting step. Can save time and memory if your features allow it. |
| **Returns**     | **AnyDataFrame** |             | **data with added features.**                                                                                              |

<details>
<summary>Code</summary>

``` python
features_ddf = fcst.preprocess(partitioned_series)
features_ddf.head()
```

</details>
<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[4] }} />

|     | unique_id | ds         | y         | static_0 | static_1 | lag7      | expanding_mean_lag1 | rolling_mean_lag7_window_size14 |
|-----|-----------|------------|-----------|----------|----------|-----------|---------------------|---------------------------------|
| 20  | id_00     | 2000-10-25 | 49.766844 | 79       | 45       | 50.694639 | 25.001367           | 26.320060                       |
| 21  | id_00     | 2000-10-26 | 3.918347  | 79       | 45       | 3.887780  | 26.180675           | 26.313387                       |
| 22  | id_00     | 2000-10-27 | 9.437778  | 79       | 45       | 11.512774 | 25.168751           | 26.398056                       |
| 23  | id_00     | 2000-10-28 | 17.923574 | 79       | 45       | 18.038498 | 24.484796           | 26.425272                       |
| 24  | id_00     | 2000-10-29 | 26.754645 | 79       | 45       | 24.222859 | 24.211411           | 26.305563                       |

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[5] }} />

This is useful if we want to inspect the data the model will be trained.
If we do this we must manually train our models and add a local version
of them to the `models_` attribute.

<details>
<summary>Code</summary>

``` python
X, y = features_ddf.drop(columns=['unique_id', 'ds', 'y']), features_ddf['y']
model = DaskXGBForecast(random_state=0).fit(X, y)
fcst.models_ = {'DaskXGBForecast': model.model_}
fcst.predict(7)
```

</details>

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
fcst.models_ = fcst2.models_
preds2 = fcst.predict(7).compute()
pd.testing.assert_frame_equal(preds, preds2)
```

</details>

:::

### Dynamic features {#dynamic-features}

By default the predict method repeats the static features and updates
the transformations and the date features. If you have dynamic features
like prices or a calendar with holidays you can pass them as a list to
the `dynamic_dfs` argument of `DistributedMLForecast.predict`, which
will call `pd.DataFrame.merge` on each of them in order.

Here’s an example:

Suppose that we have a `product_id` column and we have a catalog for
prices based on that `product_id` and the date.

<details>
<summary>Code</summary>

``` python
dynamic_series = series.rename(columns={'static_1': 'product_id'})
prices_catalog = generate_prices_for_series(dynamic_series)
prices_catalog
```

</details>
<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[6] }} />

|       | ds         | product_id | price    |
|-------|------------|------------|----------|
| 0     | 2000-06-09 | 1          | 0.548814 |
| 1     | 2000-06-10 | 1          | 0.715189 |
| 2     | 2000-06-11 | 1          | 0.602763 |
| 3     | 2000-06-12 | 1          | 0.544883 |
| 4     | 2000-06-13 | 1          | 0.423655 |
| ...   | ...        | ...        | ...      |
| 20180 | 2001-05-17 | 99         | 0.223520 |
| 20181 | 2001-05-18 | 99         | 0.446104 |
| 20182 | 2001-05-19 | 99         | 0.044783 |
| 20183 | 2001-05-20 | 99         | 0.483216 |
| 20184 | 2001-05-21 | 99         | 0.799660 |

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[7] }} />

And you have already merged these prices into your series dataframe.

<details>
<summary>Code</summary>

``` python
dynamic_series = partitioned_series.rename(columns={'static_1': 'product_id'})
dynamic_series = dynamic_series
series_with_prices = dynamic_series.merge(prices_catalog, how='left')
series_with_prices.head()
```

</details>
<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[8] }} />

|     | unique_id | ds         | y         | static_0 | product_id | price    |
|-----|-----------|------------|-----------|----------|------------|----------|
| 0   | id_00     | 2000-10-05 | 3.981198  | 79       | 45         | 0.570826 |
| 1   | id_00     | 2000-10-06 | 10.327401 | 79       | 45         | 0.260562 |
| 2   | id_00     | 2000-10-07 | 17.657474 | 79       | 45         | 0.274048 |
| 3   | id_00     | 2000-10-08 | 25.898790 | 79       | 45         | 0.433878 |
| 4   | id_00     | 2000-10-09 | 34.494040 | 79       | 45         | 0.653738 |

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[9] }} />

This dataframe will be passed to `DistributedMLForecast.fit` (or
`DistributedMLForecast.preprocess`), however since the price is dynamic
we have to tell that method that only `static_0` and `product_id` are
static and we’ll have to update `price` in every timestep, which
basically involves merging the updated features with the prices catalog.

<details>
<summary>Code</summary>

``` python
fcst = DistributedMLForecast(
    models,
    freq='D',
    lags=[7],
    lag_transforms={
        1: [expanding_mean],
        7: [(rolling_mean, 14)]
    },
    date_features=['dayofweek', 'month'],
    num_threads=1,
)
series_with_prices = series_with_prices
fcst.fit(
    series_with_prices,
    static_features=['static_0', 'product_id'],
)
```

</details>

So in order to update the price in each timestep we just call
`DistributedMLForecast.predict` with our forecast horizon and pass the
prices catalog as a dynamic dataframe.

<details>
<summary>Code</summary>

``` python
preds = fcst.predict(7, dynamic_dfs=[prices_catalog])
preds.compute()
```

</details>
<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[10] }} />

|     | unique_id | ds         | DaskXGBForecast | DaskLGBMForecast |
|-----|-----------|------------|-----------------|------------------|
| 0   | id_00     | 2001-05-15 | 42.253372       | 43.385181        |
| 1   | id_00     | 2001-05-16 | 50.520786       | 49.826045        |
| 2   | id_00     | 2001-05-17 | 1.802876        | 1.923489         |
| 3   | id_00     | 2001-05-18 | 10.081938       | 10.237613        |
| 4   | id_00     | 2001-05-19 | 18.507072       | 18.577109        |
| ... | ...       | ...        | ...             | ...              |
| 72  | id_99     | 2001-05-17 | 44.015308       | 44.185574        |
| 73  | id_99     | 2001-05-18 | 2.281136        | 1.992937         |
| 74  | id_99     | 2001-05-19 | 9.186671        | 9.123272         |
| 75  | id_99     | 2001-05-20 | 15.712516       | 15.263331        |
| 76  | id_99     | 2001-05-21 | 22.769850       | 22.825026        |

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[11] }} />

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
# test we can compute cross validation with
# exougenous variables without adding extra information
# later a more robust test is performed
cv_with_ex = fcst.cross_validation(
    series_with_prices,
    window_size=7,
    n_windows=2,
    static_features=['static_0', 'product_id'],
).compute()
```

</details>

:::

### Custom predictions {#custom-predictions}

If you want to do something like scaling the predictions you can define
a function and pass it to `DistributedMLForecast.predict` as described
in <a href="/forecast.html#custom-predictions">Custom predictions</a>.

#### Cross validation {#cross-validation}

Refer to `MLForecast.cross_validation`.

<details>
<summary>Code</summary>

``` python
show_doc(DistributedMLForecast.cross_validation, title_level=2)
```

</details>

------------------------------------------------------------------------

## DistributedMLForecast.cross_validation {#distributedmlforecast.cross_validation}

> ``` text
>  DistributedMLForecast.cross_validation (data:~AnyDataFrame,
>                                          n_windows:int, window_size:int,
>                                          id_col:str='unique_id',
>                                          time_col:str='ds',
>                                          target_col:str='y',
>                                          step_size:Optional[int]=None, sta
>                                          tic_features:Optional[List[str]]=
>                                          None, dropna:bool=True,
>                                          keep_last_n:Optional[int]=None,
>                                          refit:bool=True, before_predict_c
>                                          allback:Optional[Callable]=None, 
>                                          after_predict_callback:Optional[C
>                                          allable]=None,
>                                          input_size:Optional[int]=None)
> ```

Perform time series cross validation. Creates `n_windows` splits where
each window has `window_size` test periods, trains the models, computes
the predictions and merges the actuals.

|                         | **Type**         | **Default** | **Details**                                                                                                                                                                                                                                                          |
|------|------------------|-------------------------|-------------------------|
| data                    | AnyDataFrame     |             | Series data in long format.                                                                                                                                                                                                                                          |
| n_windows               | int              |             | Number of windows to evaluate.                                                                                                                                                                                                                                       |
| window_size             | int              |             | Number of test periods in each window.                                                                                                                                                                                                                               |
| id_col                  | str              | unique_id   | Column that identifies each serie.                                                                                                                                                                                                                                   |
| time_col                | str              | ds          | Column that identifies each timestep, its values can be timestamps or integers.                                                                                                                                                                                      |
| target_col              | str              | y           | Column that contains the target.                                                                                                                                                                                                                                     |
| step_size               | Optional         | None        | Step size between each cross validation window. If None it will be equal to `window_size`.                                                                                                                                                                           |
| static_features         | Optional         | None        | Names of the features that are static and will be repeated when forecasting.                                                                                                                                                                                         |
| dropna                  | bool             | True        | Drop rows with missing values produced by the transformations.                                                                                                                                                                                                       |
| keep_last_n             | Optional         | None        | Keep only these many records from each serie for the forecasting step. Can save time and memory if your features allow it.                                                                                                                                           |
| refit                   | bool             | True        | Retrain model for each cross validation window.<br>If False, the models are trained at the beginning and then used to predict each window.                                                                                                                           |
| before_predict_callback | Optional         | None        | Function to call on the features before computing the predictions.<br> This function will take the input dataframe that will be passed to the model for predicting and should return a dataframe with the same structure.<br> The series identifier is on the index. |
| after_predict_callback  | Optional         | None        | Function to call on the predictions before updating the targets.<br> This function will take a pandas Series with the predictions and should return another one with the same structure.<br> The series identifier is on the index.                                  |
| input_size              | Optional         | None        | Maximum training samples per serie in each window. If None, will use an expanding window.                                                                                                                                                                            |
| **Returns**             | **AnyDataFrame** |             | **Predictions for each window with the series id, timestamp, target value and predictions from each model.**                                                                                                                                                         |

<details>
<summary>Code</summary>

``` python
fcst = DistributedMLForecast(
    models=[DaskLGBMForecast(), DaskXGBForecast()],
    freq='D',
    lags=[7],
    lag_transforms={
        1: [expanding_mean],
        7: [(rolling_mean, 14)]
    },
    date_features=['dayofweek', 'month'],
    num_threads=1,
)
```

</details>
<details>
<summary>Code</summary>

``` python
n_windows = 2
window_size = 14

cv_results = fcst.cross_validation(
    partitioned_series,
    n_windows,
    window_size,
)
cv_results
```

</details>

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
# input_size
input_size = 100
reduced_train = fcst._preprocess(
    partitioned_series,
    id_col='unique_id',
    time_col='ds',
    target_col='y',
    dropna=False,
    window_info=WindowInfo(
        n_windows=1,
        window_size=10,
        step_size=None,
        i_window=0,
        input_size=input_size,
    ),
)
assert reduced_train.groupby('unique_id').size().compute().max() == input_size
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
cv_results_no_refit = fcst.cross_validation(
    partitioned_series,
    n_windows,
    window_size,
    refit=False
)
cv_results_df = cv_results.compute()
cv_results_no_refit_df = cv_results_no_refit.compute()
# test we recover the same "metadata"
models = ['DaskXGBForecast', 'DaskLGBMForecast']
test_eq(
    cv_results_no_refit_df.drop(columns=models),
    cv_results_df.drop(columns=models)
)
```

</details>

:::

We can aggregate these by date to get a rough estimate of how our model
is doing.

<details>
<summary>Code</summary>

``` python
agg_results = cv_results_df.drop(columns='cutoff').groupby('ds').mean()
agg_results.head()
```

</details>
<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[12] }} />

|            | DaskLGBMForecast | DaskXGBForecast | y         |
|------------|------------------|-----------------|-----------|
| ds         |                  |                 |           |
| 2001-04-17 | 16.195230        | 16.168709       | 16.123231 |
| 2001-04-18 | 15.145318        | 15.135734       | 15.213920 |
| 2001-04-19 | 17.149119        | 17.087150       | 16.985699 |
| 2001-04-20 | 18.002781        | 18.045092       | 18.068340 |
| 2001-04-21 | 18.136612        | 18.142144       | 18.200609 |

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[13] }} />

We can also compute the error for each model.

<details>
<summary>Code</summary>

``` python
def mse_from_dask_dataframe(ddf):
    mses = {}
    for model_name in ddf.columns.drop(['unique_id', 'ds', 'y', 'cutoff']):
        mses[model_name] = (ddf['y'] - ddf[model_name]).pow(2).mean()
    return client.gather(client.compute(mses))

{k: round(v, 2) for k, v in mse_from_dask_dataframe(cv_results).items()}
```

</details>

``` text
{'DaskLGBMForecast': 0.92, 'DaskXGBForecast': 0.87}
```

<details>
<summary>Code</summary>

``` python
client.close()
```

</details>

## Spark {#spark}

### Session setup {#session-setup}

<details>
<summary>Code</summary>

``` python
from pyspark.sql import SparkSession
```

</details>
<details>
<summary>Code</summary>

``` python
spark = (
    SparkSession.builder.appName("MyApp")
    .config("spark.jars.packages", "com.microsoft.azure:synapseml_2.12:0.10.2")
    .config("spark.jars.repositories", "https://mmlspark.azureedge.net/maven")
    .getOrCreate()
)
```

</details>

### Data setup {#data-setup-1}

For spark, the data must be a `pyspark DataFrame`. You need to make sure
that each time serie is only in one partition (which you can do using
`repartitionByRange`, for example) and it is recommended that you have
as many partitions as you have workers. If you have more partitions than
workers make sure to set `num_threads=1` to avoid having nested
parallelism.

The required input format is the same as for `MLForecast`, i.e. it
should have at least an id column, a time column and a target column.

<details>
<summary>Code</summary>

``` python
numPartitions = 4
series = generate_daily_series(100, n_static_features=2, equal_ends=True, static_as_categorical=False)
spark_series = spark.createDataFrame(series).repartitionByRange(numPartitions, 'unique_id')
```

</details>

### Models {#models-1}

In order to perform distributed forecasting, we need to use a model that
is able to train in a distributed way using `spark`. The current
implementations are in `SparkLGBMForecast` and `SparkXGBForecast` which
are just wrappers around the native implementations.

<details>
<summary>Code</summary>

``` python
from mlforecast.distributed.models.spark.lgb import SparkLGBMForecast

models = [SparkLGBMForecast()]
try:
    from xgboost.spark import SparkXGBRegressor
    from mlforecast.distributed.models.spark.xgb import SparkXGBForecast
    models.append(SparkXGBForecast())
except ModuleNotFoundError:  # py < 38
    pass
```

</details>

### Training {#training-1}

<details>
<summary>Code</summary>

``` python
fcst = DistributedMLForecast(
    models,
    freq='D',
    lags=[1],
    lag_transforms={
        1: [expanding_mean]
    },
    date_features=['dayofweek'],
)
fcst.fit(
    spark_series,
    static_features=['static_0', 'static_1'],
)
```

</details>

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
test_partition_results_size(fcst, numPartitions)
```

</details>

``` text
                                                                                
```

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
# test num_partitions works properly
test_spark_df = spark.createDataFrame(series)
num_partitions_test = 10
fcst_np = DistributedMLForecast(
    models=models,
    freq='D',
    lags=[7],
    lag_transforms={
        1: [expanding_mean],
        7: [(rolling_mean, 14)]
    },
    date_features=['dayofweek', 'month'],
    num_threads=1,
    num_partitions=num_partitions_test,
)
fcst_np.fit(test_spark_df)
test_partition_results_size(fcst_np, num_partitions_test)
preds_np = fcst_np.predict(7).toPandas().sort_values(['unique_id', 'ds']).reset_index(drop=True)
preds = fcst.predict(7).toPandas().sort_values(['unique_id', 'ds']).reset_index(drop=True)
pd.testing.assert_frame_equal(
    preds[['unique_id', 'ds']], 
    preds_np[['unique_id', 'ds']], 
)
```

</details>

:::

### Forecasting {#forecasting-1}

<details>
<summary>Code</summary>

``` python
preds = fcst.predict(14)
```

</details>
<details>
<summary>Code</summary>

``` python
preds.toPandas()
```

</details>

``` text
                                                                                
```

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[14] }} />

|      | unique_id | ds         | SparkLGBMForecast | SparkXGBForecast |
|------|-----------|------------|-------------------|------------------|
| 0    | id_00     | 2001-05-15 | 42.213984         | 42.305004        |
| 1    | id_00     | 2001-05-16 | 49.718021         | 50.262386        |
| 2    | id_00     | 2001-05-17 | 1.306248          | 1.912686         |
| 3    | id_00     | 2001-05-18 | 10.060104         | 10.240939        |
| 4    | id_00     | 2001-05-19 | 18.070785         | 18.265749        |
| ...  | ...       | ...        | ...               | ...              |
| 1395 | id_99     | 2001-05-24 | 43.426901         | 43.780163        |
| 1396 | id_99     | 2001-05-25 | 1.361680          | 2.097803         |
| 1397 | id_99     | 2001-05-26 | 8.787283          | 8.593580         |
| 1398 | id_99     | 2001-05-27 | 15.551965         | 15.622238        |
| 1399 | id_99     | 2001-05-28 | 22.518518         | 22.943216        |

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[15] }} />

### Cross validation {#cross-validation-1}

<details>
<summary>Code</summary>

``` python
cv_res = fcst.cross_validation(
    spark_series,
    n_windows=2,
    window_size=14,
).toPandas()
```

</details>
<details>
<summary>Code</summary>

``` python
cv_res
```

</details>
<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[16] }} />

|      | unique_id | ds         | SparkLGBMForecast | SparkXGBForecast | cutoff     | y         |
|------|-----------|------------|-------------------|------------------|------------|-----------|
| 0    | id_17     | 2001-04-30 | 31.467849         | 31.676336        | 2001-04-16 | 30.832464 |
| 1    | id_07     | 2001-04-17 | 1.015429          | 1.039312         | 2001-04-16 | 1.034871  |
| 2    | id_06     | 2001-04-29 | 21.133919         | 1.368022         | 2001-04-16 | 0.944155  |
| 3    | id_11     | 2001-04-17 | 57.069013         | 57.591526        | 2001-04-16 | 57.406090 |
| 4    | id_12     | 2001-04-27 | 7.965585          | 7.741258         | 2001-04-16 | 8.498222  |
| ...  | ...       | ...        | ...               | ...              | ...        | ...       |
| 2795 | id_96     | 2001-05-12 | 9.069598          | 8.925149         | 2001-04-30 | 7.983343  |
| 2796 | id_84     | 2001-05-04 | 10.474623         | 9.959846         | 2001-04-30 | 10.683266 |
| 2797 | id_87     | 2001-05-07 | 2.162316          | 2.065432         | 2001-04-30 | 1.277810  |
| 2798 | id_80     | 2001-05-11 | 22.679552         | 20.547785        | 2001-04-30 | 19.823192 |
| 2799 | id_90     | 2001-05-08 | 40.225448         | 40.293419        | 2001-04-30 | 39.215204 |

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[17] }} />

<details>
<summary>Code</summary>

``` python
spark.stop()
```

</details>

## Ray {#ray}

### Session setup {#session-setup-1}

<details>
<summary>Code</summary>

``` python
import ray
from ray.cluster_utils import Cluster
```

</details>
<details>
<summary>Code</summary>

``` python
ray_cluster = Cluster(
    initialize_head=True,
    head_node_args={"num_cpus": 2}
)
ray.init(address=ray_cluster.address, ignore_reinit_error=True)
# add mock node to simulate a cluster
mock_node = ray_cluster.add_node(num_cpus=2)
```

</details>

### Data setup {#data-setup-2}

For ray, the data must be a `ray DataFrame`. It is recommended that you
have as many partitions as you have workers. If you have more partitions
than workers make sure to set `num_threads=1` to avoid having nested
parallelism.

The required input format is the same as for `MLForecast`, i.e. it
should have at least an id column, a time column and a target column.

<details>
<summary>Code</summary>

``` python
series = generate_daily_series(100, n_static_features=2, equal_ends=True, static_as_categorical=False)
# we need noncategory unique_id
series['unique_id'] = series['unique_id'].astype(str)
ray_series = ray.data.from_pandas(series)
```

</details>

### Models {#models-2}

The ray integration allows to include `lightgbm` (`RayLGBMRegressor`),
and `xgboost` (`RayXGBRegressor`).

<details>
<summary>Code</summary>

``` python
from mlforecast.distributed.models.ray.lgb import RayLGBMForecast
from mlforecast.distributed.models.ray.xgb import RayXGBForecast

models = [
    RayLGBMForecast(),
    RayXGBForecast(),
]
```

</details>

### Training {#training-2}

To control the number of partitions to use using Ray, we have to include
`num_partitions` to `DistributedMLForecast`.

<details>
<summary>Code</summary>

``` python
num_partitions = 4
```

</details>
<details>
<summary>Code</summary>

``` python
%%capture
fcst = DistributedMLForecast(
    models,
    freq='D',
    lags=[1],
    lag_transforms={
        1: [expanding_mean]
    },
    date_features=['dayofweek'],
    num_partitions=num_partitions, # Use num_partitions to reduce overhead
)
fcst.fit(
    ray_series,
    static_features=['static_0', 'static_1'],
)
```

</details>

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
test_partition_results_size(fcst, num_partitions)
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
# test num_partitions works properly
# In this case we test that the default behavior 
# for ray datasets works as expected
fcst_np = DistributedMLForecast(
    models=models,
    freq='D',
    lags=[7],
    lag_transforms={
        1: [expanding_mean],
        7: [(rolling_mean, 14)]
    },
    date_features=['dayofweek', 'month'],
    num_threads=1,
)
fcst_np.fit(ray_series)
# we dont use test_partition_results_size
# since the number of objects is different 
# from the number of partitions
test_eq(fa.count(fcst_np.partition_results), 100) # number of series
preds_np = fcst_np.predict(7).to_pandas().sort_values(['unique_id', 'ds']).reset_index(drop=True)
preds = fcst.predict(7).to_pandas().sort_values(['unique_id', 'ds']).reset_index(drop=True)
pd.testing.assert_frame_equal(
    preds[['unique_id', 'ds']], 
    preds_np[['unique_id', 'ds']], 
)
```

</details>

:::

### Forecasting {#forecasting-2}

<details>
<summary>Code</summary>

``` python
preds = fcst.predict(14)
```

</details>
<details>
<summary>Code</summary>

``` python
preds.to_pandas()
```

</details>
<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[18] }} />

|      | unique_id | ds         | RayLGBMForecast | RayXGBForecast |
|------|-----------|------------|-----------------|----------------|
| 0    | id_00     | 2001-05-15 | 42.213984       | 41.992321      |
| 1    | id_00     | 2001-05-16 | 49.718021       | 50.999878      |
| 2    | id_00     | 2001-05-17 | 1.306248        | 1.712625       |
| 3    | id_00     | 2001-05-18 | 10.060104       | 10.157331      |
| 4    | id_00     | 2001-05-19 | 18.070785       | 18.163649      |
| ...  | ...       | ...        | ...             | ...            |
| 1395 | id_99     | 2001-05-24 | 43.426901       | 42.060478      |
| 1396 | id_99     | 2001-05-25 | 1.361680        | 2.587303       |
| 1397 | id_99     | 2001-05-26 | 8.787283        | 8.652343       |
| 1398 | id_99     | 2001-05-27 | 15.551965       | 15.278493      |
| 1399 | id_99     | 2001-05-28 | 22.518518       | 22.898369      |

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[19] }} />

### Cross validation {#cross-validation-2}

<details>
<summary>Code</summary>

``` python
%%capture
cv_res = fcst.cross_validation(
    ray_series,
    n_windows=2,
    window_size=14,
).to_pandas()
```

</details>
<details>
<summary>Code</summary>

``` python
cv_res
```

</details>
<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[20] }} />

|      | unique_id | ds         | RayLGBMForecast | RayXGBForecast | cutoff     | y         |
|------|-----------|------------|-----------------|----------------|------------|-----------|
| 0    | id_00     | 2001-04-17 | 41.395948       | 41.968201      | 2001-04-16 | 40.499332 |
| 1    | id_00     | 2001-04-18 | 50.004670       | 50.191704      | 2001-04-16 | 50.888323 |
| 2    | id_00     | 2001-04-19 | 1.821105        | 1.978645       | 2001-04-16 | 0.121812  |
| 3    | id_00     | 2001-04-20 | 10.266459       | 10.211697      | 2001-04-16 | 10.987977 |
| 4    | id_00     | 2001-04-21 | 18.285400       | 17.944368      | 2001-04-16 | 16.370385 |
| ...  | ...       | ...        | ...             | ...            | ...        | ...       |
| 2795 | id_94     | 2001-05-04 | 8.131996        | 14.582675      | 2001-04-30 | 0.940743  |
| 2796 | id_00     | 2001-05-06 | 26.576981       | 25.874252      | 2001-04-30 | 26.049574 |
| 2797 | id_86     | 2001-05-02 | 8.402695        | 8.112425       | 2001-04-30 | 7.766106  |
| 2798 | id_50     | 2001-05-04 | 10.135376       | 10.109168      | 2001-04-30 | 12.211941 |
| 2799 | id_00     | 2001-05-03 | 1.819555        | 2.102209       | 2001-04-30 | 3.937318  |

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[21] }} />

<details>
<summary>Code</summary>

``` python
ray.shutdown()
```

</details>

