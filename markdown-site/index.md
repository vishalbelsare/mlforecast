---
title: 'Nixtla &nbsp; [![Tweet](https://img.shields.io/twitter/url/http/shields.io.svg?style=social)](https://twitter.com/intent/tweet?text=Statistical%20Forecasting%20Algorithms%20by%20Nixtla%20&url=https://github.com/Nixtla/statsforecast&via=nixtlainc&hashtags=StatisticalModels,TimeSeries,Forecasting) &nbsp;[![Slack](https://img.shields.io/badge/Slack-4A154B?&logo=slack&logoColor=white)](https://join.slack.com/t/nixtlacommunity/shared_invite/zt-1pmhan9j5-F54XR20edHk0UtYAPcW4KQ)'
---

export const quartoRawHtml =
[`<center>`,`</center>`,`<h1 align="center">`,`</h1>`,`<h3 align="center">`,`</h3>`,`<div>
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
<p>280 rows × 5 columns</p>
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
import os
os.chdir('..')
```

</details>

:::

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[0] }} />

<img src="https://raw.githubusercontent.com/Nixtla/neuralforecast/main/nbs/imgs_indx/logo_mid.png">
<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[1] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[2] }} />

Machine Learning 🤖 Forecast
<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[3] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[4] }} />

Scalable machine learning for time series forecasting
<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[5] }} />

[![CI](https://github.com/Nixtla/mlforecast/actions/workflows/ci.yaml/badge.svg)](https://github.com/Nixtla/mlforecast/actions/workflows/ci.yaml)
[![Python](https://img.shields.io/pypi/pyversions/mlforecast.png)](https://pypi.org/project/mlforecast/)
[![PyPi](https://img.shields.io/pypi/v/mlforecast?color=blue.png)](https://pypi.org/project/mlforecast/)
[![conda-forge](https://img.shields.io/conda/vn/conda-forge/mlforecast?color=blue.png)](https://anaconda.org/conda-forge/mlforecast)
[![License](https://img.shields.io/github/license/Nixtla/mlforecast.png)](https://github.com/Nixtla/mlforecast/blob/main/LICENSE)

**mlforecast** is a framework to perform time series forecasting using
machine learning models, with the option to scale to massive amounts of
data using remote clusters.

## Install {#install}

### PyPI {#pypi}

`pip install mlforecast`

If you want to perform distributed training, you can instead use
`pip install "mlforecast[distributed]"`, which will also install
[dask](https://dask.org/). Note that you’ll also need to install either
[LightGBM](https://github.com/microsoft/LightGBM/tree/master/python-package)
or
[XGBoost](https://xgboost.readthedocs.io/en/latest/install.html#python).

### conda-forge {#conda-forge}

`conda install -c conda-forge mlforecast`

Note that this installation comes with the required dependencies for the
local interface. If you want to perform distributed training, you must
install dask (`conda install -c conda-forge dask`) and either
[LightGBM](https://github.com/microsoft/LightGBM/tree/master/python-package)
or
[XGBoost](https://xgboost.readthedocs.io/en/latest/install.html#python).

## Quick Start {#quick-start}

**Minimal Example**

``` python
import lightgbm as lgb

from mlforecast import MLForecast
from sklearn.linear_model import LinearRegression

mlf = MLForecast(
    models = [LinearRegression(), lgb.LGBMRegressor()],
    lags=[1, 12],
    freq = 'M'
)
mlf.fit(df)
mlf.predict(12)
```

**Get Started with this [quick
guide](https://nixtla.github.io/mlforecast/docs/quick_start_local.html).**

**Follow this [end-to-end
walkthrough](https://nixtla.github.io/mlforecast/docs/end_to_end_walkthrough.html)
for best practices.**

## Why? {#why}

Current Python alternatives for machine learning models are slow,
inaccurate and don’t scale well. So we created a library that can be
used to forecast in production environments. `MLForecast` includes
efficient feature engineering to train any machine learning model (with
`fit` and `predict` methods such as
[`sklearn`](https://scikit-learn.org/stable/)) to fit millions of time
series.

## Features {#features}

-   Fastest implementations of feature engineering for time series
    forecasting in Python.
-   Out-of-the-box compatibility with Spark, Dask, and Ray.
-   Probabilistic Forecasting with Conformal Prediction.
-   Support for exogenous variables and static covariates.
-   Familiar `sklearn` syntax: `.fit` and `.predict`.

Missing something? Please open an issue or write us in
[![Slack](https://img.shields.io/badge/Slack-4A154B?&logo=slack&logoColor=white.png)](https://join.slack.com/t/nixtlaworkspace/shared_invite/zt-135dssye9-fWTzMpv2WBthq8NK0Yvu6A)

## Examples and Guides {#examples-and-guides}

📚 [End to End
Walkthrough](https://nixtla.github.io/mlforecast/docs/end_to_end_walkthrough.html):
model training, evaluation and selection for multiple time series.

🔎 [Probabilistic
Forecasting](https://nixtla.github.io/mlforecast/docs/prediction_intervals.html):
use Conformal Prediction to produce prediciton intervals.

👩‍🔬 [Cross
Validation](https://nixtla.github.io/mlforecast/docs/cross_validation.html):
robust model’s performance evaluation.

🔌 [Predict Demand
Peaks](https://nixtla.github.io/mlforecast/docs/electricity_peak_forecasting.html):
electricity load forecasting for detecting daily peaks and reducing
electric bills.

📈 [Transfer
Learning](https://nixtla.github.io/mlforecast/docs/transfer_learning.html):
pretrain a model using a set of time series and then predict another one
using that pretrained model.

🌡️ [Distributed
Training](https://nixtla.github.io/mlforecast/docs/quick_start_distributed.html):
use a Dask cluster to train models at scale.

## How to use {#how-to-use}

The following provides a very basic overview, for a more detailed
description see the
[documentation](https://nixtla.github.io/mlforecast/).

### Data setup {#data-setup}

Store your time series in a pandas dataframe in long format, that is,
each row represents an observation for a specific serie and timestamp.

<details>
<summary>Code</summary>

``` python
from mlforecast.utils import generate_daily_series

series = generate_daily_series(
    n_series=20,
    max_length=100,
    n_static_features=1,
    static_as_categorical=False,
    with_trend=True
)
series.head()
```

</details>
<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[6] }} />

|     | unique_id | ds         | y         | static_0 |
|-----|-----------|------------|-----------|----------|
| 0   | id_00     | 2000-01-01 | 1.751917  | 72       |
| 1   | id_00     | 2000-01-02 | 9.196715  | 72       |
| 2   | id_00     | 2000-01-03 | 18.577788 | 72       |
| 3   | id_00     | 2000-01-04 | 24.520646 | 72       |
| 4   | id_00     | 2000-01-05 | 33.418028 | 72       |

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[7] }} />

### Models {#models}

Next define your models. If you want to use the local interface this can
be any regressor that follows the scikit-learn API. For distributed
training there are `LGBMForecast` and `XGBForecast`.

<details>
<summary>Code</summary>

``` python
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

models = [
    lgb.LGBMRegressor(),
    xgb.XGBRegressor(),
    RandomForestRegressor(random_state=0),
]
```

</details>

### Forecast object {#forecast-object}

Now instantiate a `MLForecast` object with the models and the features
that you want to use. The features can be lags, transformations on the
lags and date features. The lag transformations are defined as
[numba](http://numba.pydata.org/) *jitted* functions that transform an
array, if they have additional arguments you can either supply a tuple
(`transform_func`, `arg1`, `arg2`, …) or define new functions fixing the
arguments. You can also define differences to apply to the series before
fitting that will be restored when predicting.

<details>
<summary>Code</summary>

``` python
from mlforecast import MLForecast
from mlforecast.target_transforms import Differences
from numba import njit
from window_ops.expanding import expanding_mean
from window_ops.rolling import rolling_mean


@njit
def rolling_mean_28(x):
    return rolling_mean(x, window_size=28)


fcst = MLForecast(
    models=models,
    freq='D',
    lags=[7, 14],
    lag_transforms={
        1: [expanding_mean],
        7: [rolling_mean_28]
    },
    date_features=['dayofweek'],
    target_transforms=[Differences([1])],
)
```

</details>

### Training {#training}

To compute the features and train the models call `fit` on your
`Forecast` object.

<details>
<summary>Code</summary>

``` python
fcst.fit(series)
```

</details>

``` text
MLForecast(models=[LGBMRegressor, XGBRegressor, RandomForestRegressor], freq=<Day>, lag_features=['lag7', 'lag14', 'expanding_mean_lag1', 'rolling_mean_28_lag7'], date_features=['dayofweek'], num_threads=1)
```

### Predicting {#predicting}

To get the forecasts for the next `n` days call `predict(n)` on the
forecast object. This will automatically handle the updates required by
the features using a recursive strategy.

<details>
<summary>Code</summary>

``` python
predictions = fcst.predict(14)
predictions
```

</details>
<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[8] }} />

|     | unique_id | ds         | LGBMRegressor | XGBRegressor | RandomForestRegressor |
|-----|-----------|------------|---------------|--------------|-----------------------|
| 0   | id_00     | 2000-04-04 | 69.082830     | 67.761337    | 68.226556             |
| 1   | id_00     | 2000-04-05 | 75.706024     | 74.588699    | 75.484774             |
| 2   | id_00     | 2000-04-06 | 82.222473     | 81.058289    | 82.853684             |
| 3   | id_00     | 2000-04-07 | 89.577638     | 88.735947    | 90.351212             |
| 4   | id_00     | 2000-04-08 | 44.149095     | 44.981384    | 46.291173             |
| ... | ...       | ...        | ...           | ...          | ...                   |
| 275 | id_19     | 2000-03-23 | 30.151270     | 31.814825    | 32.592799             |
| 276 | id_19     | 2000-03-24 | 31.418104     | 32.653374    | 33.563294             |
| 277 | id_19     | 2000-03-25 | 32.843567     | 33.586033    | 34.530912             |
| 278 | id_19     | 2000-03-26 | 34.127210     | 34.541473    | 35.507559             |
| 279 | id_19     | 2000-03-27 | 34.329202     | 35.450943    | 36.425001             |

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[9] }} />

### Visualize results {#visualize-results}

<details>
<summary>Code</summary>

``` python
import matplotlib.pyplot as plt
import pandas as pd

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 6), gridspec_kw=dict(hspace=0.3))
for i, (uid, axi) in enumerate(zip(series['unique_id'].unique(), ax.flat)):
    fltr = lambda df: df['unique_id'].eq(uid)
    pd.concat([series.loc[fltr, ['ds', 'y']], predictions.loc[fltr]]).set_index('ds').plot(ax=axi)
    axi.set(title=uid, xlabel=None)
    if i % 2 == 0:
        axi.legend().remove()
    else:
        axi.legend(bbox_to_anchor=(1.01, 1.0))
fig.savefig('figs/index.png', bbox_inches='tight')
plt.close()
```

</details>

![](https://raw.githubusercontent.com/Nixtla/mlforecast/main/figs/index.png)

## Sample notebooks {#sample-notebooks}

-   [m5](https://www.kaggle.com/code/lemuz90/m5-mlforecast-eval)
-   [m4](https://www.kaggle.com/code/lemuz90/m4-competition)
-   [m4-cv](https://www.kaggle.com/code/lemuz90/m4-competition-cv)

## How to contribute {#how-to-contribute}

See
[CONTRIBUTING.md](https://github.com/Nixtla/mlforecast/blob/main/CONTRIBUTING.md).

