---
title: Transfer Learning
---

export const quartoRawHtml =
[`        <script type="text/javascript">
        window.PlotlyConfig = {MathJaxConfig: 'local'};
        if (window.MathJax && window.MathJax.Hub && window.MathJax.Hub.Config) {window.MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}
        if (typeof require !== 'undefined') {
        require.undef("plotly");
        define('plotly', function(require, exports, module) {
            /**
* plotly.js v2.18.0
* Copyright 2012-2023, Plotly, Inc.
* All rights reserved.
* Licensed under the MIT license
*/
/*! For license information please see plotly.min.js.LICENSE.txt */
        });
        require(['plotly'], function(Plotly) {
            window._Plotly = Plotly;
        });
        }
        </script>
        `,`<div>
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
</div>`];

Transfer learning refers to the process of pre-training a flexible model
on a large dataset and using it later on other data with little to no
training. It is one of the most outstanding 🚀 achievements in Machine
Learning and has many practical applications.

For time series forecasting, the technique allows you to get
lightning-fast predictions ⚡ bypassing the tradeoff between accuracy
and speed (more than 30 times faster than our already fast
[AutoARIMA](https://github.com/Nixtla/statsforecast) for a similar
accuracy).

This notebook shows how to generate a pre-trained model to forecast new
time series never seen by the model.

Table of Contents

-   Installing MLForecast
-   Load M3 Monthly Data
-   Instantiate NeuralForecast core, Fit, and save
-   Use the pre-trained model to predict on AirPassengers
-   Evaluate Results

You can run these experiments with Google Colab.

<a href="https://colab.research.google.com/github/Nixtla/mlforecast/blob/main/nbs/docs/transfer_learning.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Installing Libraries {#installing-libraries}

<details>
<summary>Code</summary>

``` python
%%capture
!pip install mlforecast statsforecast datasetsforecast
```

</details>
<details>
<summary>Code</summary>

``` python
import lightgbm as lgb
import numpy as np
import pandas as pd
from datasetsforecast.m3 import M3
from datasetsforecast.losses import mae
from mlforecast import MLForecast
from statsforecast import StatsForecast as sf
from statsforecast.utils import AirPassengersDF
```

</details>

``` text
```

## Load M3 Data {#load-m3-data}

The `M3` class will automatically download the complete M3 dataset and
process it.

It return three Dataframes: `Y_df` contains the values for the target
variables, `X_df` contains exogenous calendar features and `S_df`
contains static features for each time-series. For this example we will
only use `Y_df`.

If you want to use your own data just replace `Y_df`. Be sure to use a
long format and have a simmilar structure than our data set.

<details>
<summary>Code</summary>

``` python
Y_df_M3, _, _ = M3.load(directory='./', group='Monthly')
```

</details>

In this tutorial we are only using `1_000` series to speed up
computations. Remove the filter to use the whole dataset.

<details>
<summary>Code</summary>

``` python
sf.plot(Y_df_M3)
```

</details>
<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[0] }} />

![](transfer_learning_files/figure-markdown_strict/cell-5-output-2.png)

## Model Training {#model-training}

Using the `MLForecast.fit` method you can train a set of models to your
dataset. You can modify the hyperparameters of the model to get a better
accuracy, in this case we will use the default hyperparameters of
`lgb.LGBMRegressor`.

<details>
<summary>Code</summary>

``` python
models = [lgb.LGBMRegressor()]
```

</details>

The `MLForecast` object has the following parameters:

-   `models`: a list of sklearn-like (`fit` and `predict`) models.
-   `freq`: a string indicating the frequency of the data. See [panda’s
    available
    frequencies.](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases)
-   `differences`: Differences to take of the target before computing
    the features. These are restored at the forecasting step.
-   `lags`: Lags of the target to use as features.

In this example, we are only using `differences` and `lags` to produce
features. See [the full
documentation](https://nixtla.github.io/mlforecast/forecast.html) to see
all available features.

Any settings are passed into the constructor. Then you call its `fit`
method and pass in the historical data frame `Y_df_M3`.

<details>
<summary>Code</summary>

``` python
fcst = MLForecast(
    models=models, 
    lags=range(1, 13),
    freq='M',
    differences=[1, 12]
)
fcst.fit(Y_df_M3, id_col='unique_id', time_col='ds', target_col='y');
```

</details>

## Transfer M3 to AirPassengers {#transfer-m3-to-airpassengers}

Now we can tranfr the trained model to forecast `AirPassengers` with the
`MLForecast.predict` method, we just have to pass the new dataframe to
the `new_data` argument.

<details>
<summary>Code</summary>

``` python
# We define the train df. 
Y_df = AirPassengersDF.copy()

Y_train_df = Y_df[Y_df.ds<='1959-12-31'] # 132 train
Y_test_df = Y_df[Y_df.ds>'1959-12-31']   # 12 test
```

</details>
<details>
<summary>Code</summary>

``` python
Y_hat_df = fcst.predict(horizon=12, new_data=Y_train_df)
Y_hat_df.head()
```

</details>
<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[1] }} />

|     | unique_id | ds         | LGBMRegressor |
|-----|-----------|------------|---------------|
| 0   | 1.0       | 1960-01-31 | 422.740096    |
| 1   | 1.0       | 1960-02-29 | 399.480193    |
| 2   | 1.0       | 1960-03-31 | 458.220289    |
| 3   | 1.0       | 1960-04-30 | 442.960385    |
| 4   | 1.0       | 1960-05-31 | 461.700482    |

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[2] }} />

<details>
<summary>Code</summary>

``` python
Y_hat_df = Y_test_df.merge(Y_hat_df, how='left', on=['unique_id', 'ds'])
```

</details>
<details>
<summary>Code</summary>

``` python
sf.plot(Y_df, Y_hat_df)
```

</details>

![](transfer_learning_files/figure-markdown_strict/cell-11-output-1.png)

## Evaluate Results {#evaluate-results}

We evaluate the forecasts of the pre-trained model with the Mean
Absolute Error (`mae`).

$$
\qquad MAE = \frac{1}{Horizon} \sum_{\tau} |y_{\tau} - \hat{y}_{\tau}|\qquad
$$

<details>
<summary>Code</summary>

``` python
y_true = Y_test_df.y.values
y_hat = Y_hat_df['LGBMRegressor'].values
```

</details>
<details>
<summary>Code</summary>

``` python
print('LGBMRegressor     MAE: %0.3f' % mae(y_hat, y_true))
print('ETS               MAE: 16.222')
print('AutoARIMA         MAE: 18.551')
```

</details>

``` text
LGBMRegressor     MAE: 13.560
ETS               MAE: 16.222
AutoARIMA         MAE: 18.551
```
