# FBMP2
-- Fast Bayesian Matching Pursuit with improvements

1. [Documentation](#documentation)
2. [Summary](#summary)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Usage / Fixed activity features](#fixing_features) 

<a name="documentation"></a>
##Documentaion


* Model and Algorithm: [./notes/FBMP2-development-docs.pdf](notes/FBMP2-development-docs.pdf)
* Function documentation: To build doc-string documentation type `make docs` at the root of the project

<a name="summary"></a>
## Summary

Fast Bayesian Matching Pursuit (FBMP) is an algorithm published [in 2008 by Schniter et al.](http://www2.ece.ohio-state.edu/~schniter/FBMP/pubs.html), designed to solve the **sparse linear regression** problem. In this problem, there are fewer samples than features, which renders the common strategies (such as OLS, Pursuit) for fitting a linear model useless. The main challenge of the sparse problem is to select which features are "active", i.e. have non-zero coefficients.

The key idea of FBMP is to rely on the **Bayesian model evidence** as the primary metric for comparing different sets of features. This metric can be efficiently calculated on a path of models, where consecutive models differ by the activation of exactly one feature. Averaging the predictions of different plausible models yields a robust estimate of the linear coefficients.The main difficulty that FBMP faces is the optimization of **three hyperparameters** ($\lambda$, $\sigma_x$, $\sigma$), which have significant effect on the set of active features in the final result. The expectation maximization scheme presented in their paper allows one to find the optimal values of these hyperparameters, but since these are point estimates, this strategy **results in overfitting** in terms of the set of active features.

We designed and implemented an improved version of FBMP, FBMP2. The main differences are

* Requiring that the input features and targets are normalized, which allows us to define a proper prior for $\sigma_x$ and average over it.
*  Assuming a proper prior for $\lambda$ and average over it.
*   Repeating the model discovery process for a range of $\alpha = \sigma / \sigma_x$ values, and estimating its empirical prior, which we then use to average over the models.
* Recording the immediate environment of the discovered path of models, which requires no extra computational time, only space.
* Avoiding computing the mean and covariance of linear coefficients, since they are irrelevant in determining which features are active.
* Instead of repeating the model search multiple times, we carry out a band search, which avoids re-computing the evidence of already discovered models.
* We perform this band search in a way that ensures that all features are investigated.

The FBMP2 package, command line tool, and tests are made available under [GPL 3.0 license](https://www.gnu.org/licenses/gpl-3.0.en.html), see [LICENSE](./LICENSE) file.

<a name="installation"></a>
## Installation

Install with pip:

```
git clone https://github.com/totient-bio/FBMP2.git
cd fbmp2
pip install -e .
```
<a name="usage"></a>
## Usage
Given the input feature matrix `X` (where rows represent samples and columns represent features) and target vector `y` (where each entry is the target value of the corresponding sample),

```python
import numpy as np
X = np.array([[0, 1, 1, 1, 1, 1],
              [0, 1, 0, 0, 0, 0],
              [0, 1, 1, 0, 0, 0],
              [0, 1, 0, 1, 0, 0]])
y = np.array([-3.40383193, -2.07696795, -3.33594256, -2.13322689])
```
we initialize the `FeatureSelection` model,

```python
from fbmp2.feature_selection import FeatureSelection
model = FeatureSelection(X, y)
```
and compute the model likelihoods for a grid of `alpha` values, which is best chosen to cover the range of [0.01 .. 1],

```python
alpha_grid = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3.0)
model.compute_likelihoods(alpha_grid, bandwidth=10, adaptive=True)
```
where increasing `bandwidth` (default is 1) increases runtime linearly, but improves estimation robustness, and setting `adaptive` to True (default is False) improves robustness further at a cost of and increase in runtime approximately by a factor of the numbe of column of X (i.e. the number of features $N$). Since this operation is done in parallel processes for each alpha value, it is optimal to set the length of `alpha_grid` to a multiple of the number of available CPUs.

Finally we compute the posterior,

```python
posterior = model.compute_posterior(full=True)
```
where we implicitly used the default prior for $\lambda$ (i.e. a Beta prior with mean of $1 / N$, and 95th percentile of $5 / N$). The optinal argument `full` (default is False) makes the resulting dictionary `posterior` contain the posterior log probability for all discovered models.

It contains the following keys

```python
posterior.keys()
# dict_keys(['alpha', 'logQ', 'log_pn_0', 'log_pN1', 'x_mmse', 'full_log_psy_df', 'x_fix_mmse'])
```
* `logQ` labels the log posterior of the $\alpha$ parameter. Its normalized posterior distribution is shown below, which indicates that the most likely alpha value is 0.01. (We can confirm that the model converged, by checking that this posterior of alpha is decreasing towards both lower and upper edges of the alpha grid.)

```python
list(zip(posterior['alpha'], np.exp(posterior['logQ'])))
# [(0.001, 0.003271351950311004),
#  (0.003, 0.12777468139000286),
#  (0.01, 0.3944004391926269),
#  (0.03, 0.2537491603503756),
#  (0.1, 0.19997628160956213),
#  (0.3, 0.017488863169074238),
#  (1.0, 0.002893091985557341),
#  (3.0, 0.00044613035249010214)]
```

* `log_pn_0` labels the log posterior probability of each feature being **inactive**. The posterior chance of each feature being active is shown below, indicating that the model is confident that feature 2 is active. (For features 0 and 1, the posterior is 1/6, same as the prior probability 1/N, since N = 6. This is because columns 0 and 1 of `X` are uniform across all samples, i.e. the samples do not demonstrate any contrast for these features.

```python
list(enumerate(1.0 - np.exp(posterior['log_pn_0'])))
# [(0, 0.16666666666666663),
#  (1, 0.16666666666666663),
#  (2, 0.9998897429855923),
#  (3, 0.6851104188025272),
#  (4, 0.01260062516516347),
#  (5, 0.01260062516516347)]
```

* `x_mmse` labels the posterior means of the linear coefficients associated with each feature. For features that are unform across all samples, features 0 and 1, this cannot be estimated, and NaN is returned. This result shows that feature 2, which has a high posterior chance of being active, has a coefficient of -1.26.

```python
list(enumerate(posterior['x_mmse']))
# [(0, nan),
#  (1, nan),
#  (2, -1.264612985882163),
#  (3, -0.060211650435812836),
#  (4, -6.008085100872249e-05),
#  (5, -6.008085100872249e-05)]
```

* `log_pN1` labels the log posterior of the number of active features. The normalized distribution is computed below, showing that the model is confident that at least one feature is active, and probably two, but not very confident about the exact number. (The model predicts that there is zero chance that 5 or more features are active. This is because the statistical model works only for models where the number of active features is not more than the number of samples minus two. The prior distribution of the two uniform features, feature 0 and feature 1, are added to this result, resulting in the maximum number of active features in the posterior being 4.)

```python
list(enumerate(np.exp(posterior['log_pN1'])))
# [(0, 7.252971178965746e-05),
#  (1, 0.20354828033904243),
#  (2, 0.5432418114012745),
#  (3, 0.13331071144006112),
#  (4, 0.062194134241905445),
#  (5, 0.0),
#  (6, 0.0)]
```

* `full_log_psy_df` labels the log posterior table, a pandas DataFrame, that lists all discovered models and their log probability, given the data. Here each column `s_{n}` contains the binary indicators showing if feature `n` is active in a given model (`NaN`s stand for "0 or 1", for features that are homogeneous in the available samples). Column `log_psy` is the log probability. We also computed two additional columns: `N1`, the number of active features, and `psy` the normalize posterior probability of the models. The winning model, where features 2 and 3 are active has 68.5% posterior probability, the next best model, where only feature 2 is active is at 29.0%, and the third best, where feature 2 and 4 are active only has 1.3% chance. Feature 2 is active in the four best models, that with total probability 99.9889743%.

```python
df = posterior['full_log_psy_df'].copy()
df.insert(6, 'N1', np.nansum(df[[f's_{n}' for n in range(6)]].values, axis=1).astype(int))
df['psy'] = np.exp(df['log_psy'])
print(df.sort_values(by='psy', ascending=False))
#     s_0  s_1  s_2  s_3  s_4  s_5  N1    log_psy           psy
# 5   NaN  NaN    1    1    0    0   2  -0.378178  6.851088e-01
# 1   NaN  NaN    1    0    0    0   1  -1.239301  2.895866e-01
# 6   NaN  NaN    1    0    1    0   2  -4.374281  1.259719e-02
# 7   NaN  NaN    1    0    0    1   2  -4.374281  1.259719e-02
# 0   NaN  NaN    0    0    0    0   0  -9.178867  1.031974e-04
# 3   NaN  NaN    0    0    1    0   1 -12.952326  2.370698e-06
# 4   NaN  NaN    0    0    0    1   1 -12.952326  2.370698e-06
# 2   NaN  NaN    0    1    0    0   1 -13.927596  8.939676e-07
# 9   NaN  NaN    0    0    1    1   2 -14.168477  7.026009e-07
# 8   NaN  NaN    0    1    0    1   2 -14.834896  3.608165e-07
# 10  NaN  NaN    0    1    1    0   2 -14.834896  3.608165e-07
```

(Note: In this simple example, FeatureSelection discovered all 11 states that have no more than 2 active features. For bigger X and y inputs, the discovery is incomplete, but it tends to find the states with highest posteriors. The reliability of this discovery is improved with increasing `bandwidth` and setting `adaptive` to True when calling `model.compute_likelihoods()`.)

* `x_fix_mmse` is used only if the `X_fix` optional argument if supplied to `FeatureSelection` (See [Fixed activity features](#fixing_features)). It labels the estimated coefficients for these features. Since we did not have `X_fix`, it is empty.

```python
posterior['x_fix_mmse']
# array([], dtype=float64)
```

<a name="fixing_features"></a>
### Fixed activity features
FeatureSelection can be called with the `X_fix` optional argument, which will instruct the model to treat the columns of `X_fix` as new features, which are always active.

```python
X_fix = np.array([
    [0],
    [1],
    [1],
    [0]
])
model = FeatureSelection(X, y, X_fix)
model.compute_likelihoods(alpha_grid, bandwidth=10, adaptive=True)
posterior = model.compute_posterior(full=True)
```

Now, `x_fix_mmse` contains the coefficient associated with the active feature:

```python
posterior['x_fix_mmse']
# array([0.06207283])
```

The resulting posterior is different in two aspects: 1) The maximal complexity `N1` of the models is reduced by the number of columns in `X_fix` to avoid overfitting. 2) Posterior values are different, because the features in `X_fix` can take care of explaining certain aspects of the that for which, previously, we needed features from `X`.

```python
df = posterior['full_log_psy_df'].copy()
df.insert(6, 'N1', np.nansum(df[[f's_{n}' for n in range(6)]].values, axis=1).astype(int))
df['psy'] = np.exp(df['log_psy'])
print(df.sort_values(by='psy', ascending=False))
#    s_0  s_1  s_2  s_3  s_4  s_5  N1    log_psy           psy
# 1  NaN  NaN    1    0    0    0   1  -0.000001  9.999986e-01
# 0  NaN  NaN    0    0    0    0   0 -13.607338  1.231425e-06
# 3  NaN  NaN    0    0    1    0   1 -16.456200  7.131209e-08
# 4  NaN  NaN    0    0    0    1   1 -16.456200  7.131209e-08
# 2  NaN  NaN    0    1    0    0   1 -16.805016  5.031233e-08

```