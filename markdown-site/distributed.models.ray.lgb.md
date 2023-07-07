---
title: RayLGBMForecast
---

> ray LightGBM forecaster

Wrapper of `lightgbm.ray.RayLGBMRegressor` that adds a `model_` property
that contains the fitted booster and is sent to the workers to in the
forecasting step.

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

<details>
<summary>Code</summary>

``` python
import lightgbm as lgb
from lightgbm_ray import RayLGBMRegressor
```

</details>

:::

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

<details>
<summary>Code</summary>

``` python
class RayLGBMForecast(RayLGBMRegressor):
    @property
    def model_(self):
        return self._lgb_ray_to_local(lgb.LGBMRegressor)
```

</details>

:::

