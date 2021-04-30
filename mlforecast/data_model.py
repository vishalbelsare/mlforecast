# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/data_model.ipynb (unless otherwise specified).

__all__ = ['DataFreq', 'DataFormat', 'Data', 'Backtest', 'Features', 'Forecast', 'ModelConfig', 'Local', 'Cluster',
           'DistributedModelName', 'DistributedModelConfig', 'Distributed', 'Config', 'DateFeatures', 'Transforms']

# Cell
from enum import Enum
from typing import Dict, List, Optional, Union
try:
    from typing import Literal  # python>=3.8
except ImportError:
    from typing_extensions import Literal

import window_ops.rolling
import window_ops.expanding
import window_ops.ewm
from pydantic import BaseModel, root_validator

from .core import date_features_dtypes

# Internal Cell
_available_tfms = {}
for module_name in ('rolling', 'expanding', 'ewm'):
    module = getattr(window_ops, module_name)
    for tfm in module.__all__:
        _available_tfms[tfm] = getattr(module, tfm)

# Cell
DateFeatures = Literal[tuple(date_features_dtypes.keys())]

Transforms = Literal[tuple(_available_tfms.keys())]

class DataFreq(str, Enum):
    B = 'B'
    C = 'C'
    D = 'D'
    W = 'W'
    M = 'M'
    SM = 'SM'
    BM = 'BM'
    CBM = 'CBM'
    MS = 'MS'
    SMS = 'SMS'
    BMS = 'BMS'
    CBMS = 'CBMS'
    Q = 'Q'
    BQ = 'BQ'
    QS = 'QS'
    BQS = 'BQS'
    A = 'A'
    Y = 'Y'
    BA = 'BA'
    BY = 'BY'
    AS = 'AS'
    YS = 'YS'
    BAS = 'BAS'
    BYS = 'BYS'
    BH = 'BH'
    H = 'H'
    T = 'T'
    S = 'S'
    L = 'L'
    U = 'U'
    N = 'N'
    W_MON = 'W-MON'
    W_TUE = 'W-TUE'
    W_WED = 'W-WED'
    W_THU = 'W-THU'
    W_FRI = 'W-FRI'
    W_SAT = 'W-SAT'
    Q_JAN = 'Q-JAN'
    Q_FEB = 'Q-FEB'
    Q_MAR = 'Q-MAR'
    Q_APR = 'Q-APR'
    Q_MAY = 'Q-MAY'
    Q_JUN = 'Q-JUN'
    Q_JUL = 'Q-JUL'
    Q_AUG = 'Q-AUG'
    Q_SEP = 'Q-SEP'
    Q_OCT = 'Q-OCT'
    Q_NOV = 'Q-NOV'
    A_JAN = 'A-JAN'
    A_FEB = 'A-FEB'
    A_MAR = 'A-MAR'
    A_APR = 'A-APR'
    A_MAY = 'A-MAY'
    A_JUN = 'A-JUN'
    A_JUL = 'A-JUL'
    A_AUG = 'A-AUG'
    A_SEP = 'A-SEP'
    A_OCT = 'A-OCT'
    A_NOV = 'A-NOV'

class DataFormat(str, Enum):
    csv = 'csv'
    parquet = 'parquet'

class Data(BaseModel):
    prefix: str
    input: str
    output: str
    format: DataFormat

class Backtest(BaseModel):
    n_windows: int
    window_size: int

class Features(BaseModel):
    freq: DataFreq
    lags: Optional[List[int]]
    lag_transforms: Optional[Dict[int, List[Union[Transforms, Dict[Transforms, Dict]]]]]
    date_features: Optional[List[Literal[DateFeatures]]]
    static_features: Optional[List[str]]
    num_threads: Optional[int]

class Forecast(BaseModel):
    horizon: int

class ModelConfig(BaseModel):
    name: str
    params: Optional[Dict]

class Local(BaseModel):
    model: ModelConfig

class Cluster(BaseModel):
    class_name: str
    class_kwargs: Dict

class DistributedModelName(str, Enum):
    XGBoost = 'XGBForecast'
    LightGBM = 'LGBMForecast'

class DistributedModelConfig(BaseModel):
    name: DistributedModelName
    params: Optional[Dict]

class Distributed(BaseModel):
    model: DistributedModelConfig
    cluster: Cluster

class Config(BaseModel):
    data: Data
    features: Features
    backtest: Optional[Backtest]
    forecast: Optional[Forecast]
    local: Optional[Local]
    distributed: Optional[Distributed]

    @root_validator
    def check_local_or_distributed(cls, values):
        local = values.get('local')
        distributed = values.get('distributed')
        if local and distributed:
            raise ValueError('Must specify either local or distributed, not both.')
        if not local and not distributed:
            raise ValueError('Must specify either local or distributed.')
        return values