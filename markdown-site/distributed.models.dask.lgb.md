---
title: DaskLGBMForecast
---

> dask LightGBM forecaster

Wrapper of `lightgbm.dask.DaskLGBMRegressor` that adds a `model_`
property that contains the fitted booster and is sent to the workers to
in the forecasting step.

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

<details>
<summary>Code</summary>

``` python
import warnings

import lightgbm as lgb
```

</details>

:::

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

<details>
<summary>Code</summary>

``` python
class DaskLGBMForecast(lgb.dask.DaskLGBMRegressor):
    if lgb.__version__ < "3.3.0":
        warnings.warn(
            "It is recommended to install LightGBM version >= 3.3.0, since "
            "the current LightGBM version might be affected by https://github.com/microsoft/LightGBM/issues/4026, "
            "which was fixed in 3.3.0"
        )

    @property
    def model_(self):
        return self.to_local()
```

</details>

:::

