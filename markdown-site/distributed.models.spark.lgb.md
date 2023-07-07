---
title: SparkLGBMForecast
---

> spark LightGBM forecaster

Wrapper of `synapse.ml.lightgbm.LightGBMRegressor` that adds an
`extract_local_model` method to get a local version of the trained model
and broadcast it to the workers.

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

<details>
<summary>Code</summary>

``` python
import lightgbm as lgb
try:
    from synapse.ml.lightgbm import LightGBMRegressor
except ModuleNotFoundError:
    import os
    
    if os.getenv('QUARTO_PREVIEW', '0') == '1' or os.getenv('IN_TEST', '0') == '1':
        LightGBMRegressor = object
    else:
        raise
```

</details>

:::

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

<details>
<summary>Code</summary>

``` python
class SparkLGBMForecast(LightGBMRegressor):
    def _pre_fit(self, target_col):
        return self.setLabelCol(target_col)
        
    def extract_local_model(self, trained_model):
        model_str = trained_model.getNativeModel()
        local_model = lgb.Booster(model_str=model_str)
        return local_model
```

</details>

:::

