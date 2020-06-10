##############################################################################
# fbmp2 is a python3 package that implements an improved version of
# Fast Bayesian Matching Pursuit, a variable selection algorithm for
# linear regression
#
# Copyright (C) 2020  Totient, Inc.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License v3.0
# along with this program.
# If not, see <https://www.gnu.org/licenses/gpl-3.0.en.html>.
#
# Developer:
# Peter Komar (peter.komar@totient.bio)
##############################################################################

import fbmp2.fbmp2 as fbmp2
from warnings import warn
import numpy as np


class FeatureSelection:
    """Realizes the feature selection algorithm of FBMP2.

    Args:
        X (numpy.ndarray): [samples]x[features] feature matrix
        y (numpy.ndarray): [samples] target vector
        X_fix (numpy.ndarray): [samples]x[fixed_features] feature matrix
            (features that are fixed to be always on)
        normalize_X_scales (bool): if True scales of each column of X
            feature matrix is centered and normalized by its empirical stdev,
            if False: they are only centered [default: True]

    Attributes:
        X (numpy.ndarray)
        y (numpy.ndarray)
        X_fix (numpy.ndarray)
        X_norm (numpy.ndarray): feature matrix with columns normalized,
            and uniform columns removed
        y_norm (numpy.ndarray): normalized target vector
        X_fix_norm (numpy.ndarray): normalized fixed feature matrix
    """

    def __init__(self, X, y, X_fix=None, normalize_X_scales=True):
        if X_fix is None:
            X_fix = np.zeros([X.shape[0], 0])
        self._check_input(X, y, X_fix)
        self.X = X
        self.y = y
        self.X_fix = X_fix

        self._normalization_results = None
        self._normalize(normalize_X_scales)
        self.X_norm = self._normalization_results['A_norm']
        self.y_norm = self._normalization_results['y_norm']
        self.X_fix_norm = self._normalization_results['A_fix_norm']

        self._sweep_alpha_results = None

    @staticmethod
    def _check_input(X, y, X_fix):
        if not isinstance(X, np.ndarray):
            raise ValueError('X must be a 2-d numpy array (matrix)')
        if len(X.shape) != 2:
            raise ValueError('X must be a 2-d numpy array (matrix)')
        if not np.issubdtype(X.dtype, np.number):
            raise ValueError('dtype of X must be subtype of numpy.number')
        if np.isnan(X).any() or np.isinf(X).any():
            raise ValueError('all elements of X must be finite, non-NaN')

        if not isinstance(X_fix, np.ndarray):
            raise ValueError('X_fix must be a 2-d numpy array (matrix)')
        if len(X_fix.shape) != 2:
            raise ValueError('X_fix must be a 2-d numpy array (matrix)')
        if not np.issubdtype(X_fix.dtype, np.number):
            raise ValueError('dtype of X_fix must be subtype of '
                             'numpy.number')
        if np.isnan(X_fix).any() or np.isinf(X_fix).any():
            raise ValueError('all elements of X_fix must be finite, '
                             'non-NaN')

        if not isinstance(y, np.ndarray):
            raise ValueError('y must be a 1-d numpy array (vector)')
        if len(y.shape) != 1:
            raise ValueError('y must be a 1-d numpy array (vector)')
        if not np.issubdtype(y.dtype, np.number):
            raise ValueError('dtype of y must be subtype of numpy.number')
        if np.isnan(y).any() or np.isinf(y).any():
            raise ValueError('all elements of y must be finite, non-NaN')

        if y.shape[0] != X.shape[0]:
            raise ValueError(f'length of y {y.shape} is inconsistent '
                             f'with the number of rows of X {X.shape}')

        if y.shape[0] != X_fix.shape[0]:
            raise ValueError(f'length of y {y.shape} is inconsistent '
                             f'with the number of rows of '
                             f'X_fix {X_fix.shape}')
        if y.shape[0] < 3 + X_fix.shape[1]:
            raise ValueError(f'length of y {y.shape}, must be at least 3 '
                             f'+ number of fixed features '
                             f'({3 + X_fix.shape[1]})')

    def _normalize(self, normalize_X_scales):
        self._normalization_results = fbmp2.normalize_input(self.X,
                                                            self.y,
                                                            self.X_fix,
                                                            normalize_X_scales)

    def compute_likelihoods(self,
                            alpha_grid=(0.001, 0.01, 0.1, 1),
                            a=0,
                            b=0,
                            max_active=None,
                            bandwidth=1,
                            adaptive=False,
                            algorithm='N1-space',
                            processes=None):
        """Computes the likelihood for a large set of models.

        This is the main function of FBMP2, it takes about 5 us per node to
        run. The number of considered nodes is B * F * min(F, max_active)
        for non-adaptive runs, and about 0.5 * F times more for adaptive runs,
        where F is the number of features and B is the bandwidth.

        Running this function updates the internal variable
        `_sweep_alpha_results`, which enables calling
        :func:`compute_posterior` to obtain the final results.

        Args:
            alpha_grid (sequence of :obj:`float`): alpha values to consider
                (alpha = sigma / sigma_coeffs,
                where sigma is the uncorrelated noise strength, and
                sigma_coeffs is the stdev of the non-zero features,
                all understood to correspond to
                the normalized inputs, X_norm and y_norm)
            a (float): "a" parameter of the inverse-gamma prior of sigma_x^2
                (must be >= 0)
            b (float): "b" parameter of the inverse-gamma prior of sigma_x^2
                (must be >= 0)
            max_active (int): maximum number of active features to consider
                (must be >= 0, optional,
                default: [number of samples] - 2 - [number of fixed features])
            bandwidth (int): width of the band search algorithm, higher the
                better, but runtime increases approximately linearly with
                bandwidth (must be > 0, optional)
            adaptive (bool): If True, the band search is performed in a way
                that ensures that the set of discovered models include both
                active and inactive states of every feature at least
                `bandwidth` times. Setting it to True typically increases
                runtime by a factor of 0.5*[features], but improves
                reliability of the result more than one would get by setting
                `bandwidth` to 0.5*[features].
            algorithm (str): Either "M-space" or "N1-space", indicating
                whether the updates should be computed in sample("M") space
                or active feature ("N1") space
            processes (int): Max number of parallel computational processes
                (optional, default: number of CPUs)
        """
        M, N_fix = self.X_fix_norm.shape
        M, N = self.X_norm.shape
        if max_active is None:
            max_active = M - 2 - N_fix
        if max_active > M - 2 - N_fix:
            warn(f'Despite max_active being set to {max_active}, '
                 f' only models up to {M} - 2 - {N_fix} = {M-2 -N_fix} '
                 f'active features will '
                 f'be considered, because there is only {M} samples, '
                 f'and there are {N_fix} fixed features')
            max_active = M - 2 - N_fix

        if algorithm not in {'M-space', 'N1-space'}:
            raise ValueError('`algorithm` must one of '
                             '{"M-space", "N1-space"}')

        if processes is None:
            processes = fbmp2.cpu_count()

        # Run FBMP2 on normalized inputs starting from s = (0,0,0, ... 0)
        s_start = [0] * N
        sweep_alpha_results = fbmp2.sweep_alpha(alpha_grid,
                                                self.X_norm,
                                                self.y_norm,
                                                self.X_fix_norm,
                                                s_start,
                                                max_active,
                                                bandwidth,
                                                adaptive,
                                                a,
                                                b,
                                                algorithm,
                                                max_processes=processes)
        self._sweep_alpha_results = sweep_alpha_results

    def compute_posterior(self, lambda_ev=None, lambda_pseudocount=None,
                          estimate_coeffs=True, full=False):
        """Compute posteriors based on a Beta prior for lambda.

        This must be called after :func:`compute_likelihoods`.

        Features that are uniform in the original input feature matrix X,
        get assigned trivial posterior: 0.0 chance of being active.

        Args:
            lambda_ev (float): expectation value of a Beta prior of
                a prior feature activity probability (must be 0..1, optional,
                default is 1 / [number of features])
            lambda_pseudocount (float): pseudo-count of Beta prior of
                a prior feature activity probability (must be > 0, optional,
                default is such that lambda is < 5.0 / [number of features]
                with a priori chance of 95%)
            estimate_coeffs (bool): if True, the linear coefficients are
                also estimated
            full (bool): if True, posterior probabilities for all discovered
                models are returned in a form of a pandas DataFrame.

        Returns:
            dict: dict containing

                "alpha": vector of considered alpha values

                "logQ": emirical log-prior of alpha

                "log_pn_0": log-posterior probabilities of each feature
                    being *inactive* (large negative values indicate that the
                    corresponding feature is most likely *active*, the
                    probability of which can be computed by 1 - exp(log_pn_0))

                "log_pN1": vector of [features] + 1 log-probabilities, where
                    log_pN1[k] is the log-probaility of exactly k feature
                    being active.

                "x_mmse": vector of [features] linear coefficients
                    corresponding to the columns of the A features matrix

                "x_fix_mmse": vector of [fixed_features] linear coefficients
                    correspondign to the columns of the A_fix feature matrix

                "full_log_psy_df": pandas DataFrame, where each record is one
                    of the discovered model, columns "s_{idx}" stand for the
                    binary feature activation pattern of the model, and
                    column "log_psy" is the posterior log-probability of the
                    model. (None, if `full` optional argument is False)

        """
        N_full = self.X.shape[1]
        N = self.X_norm.shape[1]

        if lambda_ev is None:
            lambda_ev = 1.0 / N_full
        if lambda_pseudocount is None:
            # setting pseudo count to 0.2254 * N,
            # while expectation value is 1.0 / N, ensures
            # that the 95th percentile of the beta distribution is 5.0 / N
            # with very high accuracy, for N larger than 10.
            lambda_pseudocount = 0.2254 * N_full

        if self._sweep_alpha_results is None:
            raise RuntimeError('Likelihoods not computed. '
                               'Try running self.compute_likelihoods() first.')
        if lambda_ev <= 0 or lambda_ev >= 1:
            raise ValueError('lambda_ev must be >0 and <1')
        if lambda_pseudocount <= 0:
            raise ValueError('lambda_pseudocount must be positive')

        log_ps = fbmp2.calculate_s_logprior(N,
                                            lambda_pseudocount,
                                            lambda_ev)

        posteriors = fbmp2.calculate_posteriors(self._sweep_alpha_results,
                                                log_ps, self.X_norm,
                                                self.y_norm, self.X_fix_norm,
                                                estimate_coeffs, full)

        unnormed_posteriors = \
            fbmp2.unnormalize_posteriors(posteriors,
                                         self._normalization_results,
                                         lambda_pseudocount,
                                         lambda_ev)

        return unnormed_posteriors
