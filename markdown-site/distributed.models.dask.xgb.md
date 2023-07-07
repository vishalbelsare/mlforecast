---
title: DaskXGBForecast
---

> dask XGBoost forecaster

Wrapper of `xgboost.dask.DaskXGBRegressor` that adds a `model_` property
that contains the fitted model and is sent to the workers in the
forecasting step.

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

<details>
<summary>Code</summary>

``` python
import xgboost as xgb
```

</details>

:::

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

<details>
<summary>Code</summary>

``` python
class DaskXGBForecast(xgb.dask.DaskXGBRegressor):
    @property
    def model_(self):
        model_str = self.get_booster().save_raw('ubj')
        local_model = xgb.XGBRegressor()
        local_model.load_model(model_str)
        return local_model
```

</details>

:::
