# Robust Regression via Elliptical Distributions

robust-regression-elliptical contains a set of solvers for estimating the second order statistics of an unknown multivariate elliptical distribution.

## Reproduction of results
### CAMSAP
#### Synthetic data

With default parameters:
`python plot_camsap_synthetic.py`

Custom:
`python plot_camsap_synthetic.py -d normal -x 7 -y 3 -N 50 60 70 -M 50`

#### Floods - dense regressors

With default parameters:
`python plot_camsap_floods_dense.py`

Custom:
`python plot_camsap_floods_dense.py -N 300 500 -M 10`

#### Floods - structured regressors

With default parameters:
`python plot_camsap_floods_structured.py`

Custom:
`python plot_camsap_floods_structured.py -N 100 200 -M 10`

#### Stocks

With default parameters:
`python plot_camsap_stocks.py`

Custom:
`python plot_camsap_stocks.py -N 100 200 -s 2 -d 4`

### NeurIPS
#### Synthetic data

With default parameters:
`python plot_neurips_synthetic.py`

Custom:
`python plot_neurips_synthetic.py -p 10 -N 50 60 70 -M 50 -b 0.5`

#### Floods

With default parameters:
`python plot_neurips_floods.py`

Custom:
`python plot_neurips_floods.py -N 100 200 300 -M 10`

#### Stocks

With default parameters:
`python plot_neurips_stocks.py`

Custom:
`python plot_neurips_stocks.py -N 100 200 300 -d 75 -s 10`

## Code examples

`plots.plot_variables_vs_N()` accepts a list of estimators or regressors.

#### Creating an estimator

To create an estimator, create an instance of any of the classes in `estimators.joint` or `estimators.conditional`.
For example, to create an MM newton joint estimator with Tyler loss of dimension 10:

```python
import estimators
import losses

dx = 3
dy = 7
dimension = dx + dy
estimator = estimators.joint.general_loss.MMNewtonJointEstimator(
    loss=losses.tyler(dimension))
```

#### Creating a regressor

To create a regressor, create an instance of any of the classes in `regressors.joint`, `regressors.conditional` or `regressors.common`.
For example, to create a joint regressor based on the previously defined joint estimator:

```python
import regressors

regressor = regressors.joint.JointRegressor(dx, dy, estimator)
```

## Results
The following are plots generated by code in this repository.

#### Synthetic data

Normally distributed:

![CAMSAP synthetic results, normally distributed](https://github.cs.huji.ac.il/nofar-noy/robust-regression-elliptical/raw/master/images/results/camsap_synthetic_normal.png)

T-distributed:

![CAMSAP synthetic results, T-distributed](https://github.cs.huji.ac.il/nofar-noy/robust-regression-elliptical/raw/master/images/results/camsap_synthetic_t.png)

#### Floods

Dense regressors:

![CAMSAP floods results, dense regressors](https://github.cs.huji.ac.il/nofar-noy/robust-regression-elliptical/raw/master/images/results/camsap_dense.png)

Structured regressors:

![CAMSAP floods results, dense regressors](https://github.cs.huji.ac.il/nofar-noy/robust-regression-elliptical/raw/master/images/results/camsap_dense_time_space.png)

#### Stocks

Mean squared error:

![CAMSAP floods results, dense regressors](https://github.cs.huji.ac.il/nofar-noy/robust-regression-elliptical/raw/master/images/results/camsap_stocks_mse.png)

Runtime:

![CAMSAP floods results, dense regressors](https://github.cs.huji.ac.il/nofar-noy/robust-regression-elliptical/raw/master/images/results/camsap_stocks_runtime.png)
