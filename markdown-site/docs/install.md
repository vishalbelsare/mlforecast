---
title: Install
---

> Instructions to install the package from different sources.

## Released versions {#released-versions}

### PyPI {#pypi}

#### Latest release {#latest-release}

To install the latest release of mlforecast from
[PyPI](https://pypi.org/project/mlforecast/) you just have to run the
following in a terminal:

`pip install mlforecast`

#### Specific version {#specific-version}

If you want a specific version you can include a filter, for example:

-   `pip install "mlforecast==0.3.0"` to install the 0.3.0 version
-   `pip install "mlforecast<0.4.0"` to install any version prior to
    0.4.0

#### Distributed training {#distributed-training}

If you want to perform distributed training you have to include the
[dask](https://www.dask.org/) extra:

`pip install "mlforecast[dask]"`

and also either
[LightGBM](https://github.com/microsoft/LightGBM/tree/master/python-package)
or
[XGBoost](https://xgboost.readthedocs.io/en/latest/install.html#python).

### Conda {#conda}

#### Latest release {#latest-release-1}

The mlforecast package is also published to
[conda-forge](https://anaconda.org/conda-forge/mlforecast), which you
can install by running the following in a terminal:

`conda install -c conda-forge mlforecast`

Note that this happens about a day later after it is published to PyPI,
so you may have to wait to get the latest release.

#### Specific version {#specific-version-1}

If you want a specific version you can include a filter, for example:

-   `conda install -c conda-forge "mlforecast==0.3.0"` to install the
    0.3.0 version
-   `conda install -c conda-forge "mlforecast<0.4.0"` to install any
    version prior to 0.4.0

#### Distributed training {#distributed-training-1}

If you want to perform distributed training you also have to install
[dask](https://www.dask.org/):

`conda install -c conda-forge dask`

and also either
[LightGBM](https://github.com/microsoft/LightGBM/tree/master/python-package)
or
[XGBoost](https://xgboost.readthedocs.io/en/latest/install.html#python).

## Development version {#development-version}

If you want to try out a new feature that hasn’t made it into a release
yet you have the following options:

-   Install from github:
    `pip install git+https://github.com/Nixtla/mlforecast`
-   Clone and install:
    -   `git clone https://github.com/Nixtla/mlforecast`
    -   `pip install mlforecast`

which will install the version from the current main branch.

