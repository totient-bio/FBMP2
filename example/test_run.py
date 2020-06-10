#!/usr/bin/env python3

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

import sys
sys.path.append('../fbmp2/')
from fbmp2.fbmp2 import sweep_alpha, calculate_posteriors, calculate_s_logprior
import numpy as np
from scipy.stats import bernoulli
from itertools import product
from pickle import dump
from timeit import default_timer as timer


def generate_data(M, N, active_features, sigma_x, sigma):
    A = bernoulli.rvs(0.1, size=(M, N))  # feature matrix

    # pick active features
    T_true = np.random.choice(N, size=active_features, replace=False)
    s_true = np.zeros(N, dtype=int)
    s_true[T_true] = 1

    # pick linear coefficients
    x_true = np.zeros(N, dtype=float)
    x_true[T_true] = sigma_x * np.random.randn(active_features)

    # generate observations
    y = A.dot(x_true) + sigma * np.random.randn(M)

    return A, y, s_true, x_true


def normalize_data(A, y):
    M, N = A.shape
    feature_is_uniform = np.array(
        [np.min(A[:, n]) == np.max(A[:, n]) for n in range(N)])
    A_useful = A[:, ~feature_is_uniform]
    A_shifted = A_useful - np.einsum('i,j->ij', np.ones(A_useful.shape[0]),
                                     np.mean(A_useful, axis=0))
    A_norm = A_shifted / np.einsum('i,j->ij', np.ones(A_shifted.shape[0]),
                                   np.std(A_shifted, axis=0))
    y_norm = (y - np.mean(y)) / np.std(y)

    return A_norm, y_norm, feature_is_uniform


def main():
    N_full = 50  # number of features
    M = 20  # samples

    replicates = range(10)

    active_features_list = [0, 1, 2, 3]
    sigma_x_list = [1.0]
    sigma_list = [0.01, 0.1, 1, 10]

    run_id = 0

    with open('runs.tsv', 'w') as fout:
        header = ['run_id', 'N', 'M', 'replicate_id', 'active_features', 'sigma_x', 'sigma', 'data_file', 'result_file']
        fout.write('\t'.join(header) + '\n')

    total_runs = len(list(product(replicates, active_features_list, sigma_x_list, sigma_list)))
    for replicate_id, active_features, sigma_x, sigma in \
            product(replicates, active_features_list, sigma_x_list, sigma_list):
        print(f'Run {run_id+1} / {total_runs}', end=',  ')

        try:
            data_file = f'simulated_data/data_{run_id}.pkl'
            result_file = f'results/result_{run_id}.pkl'

            A, y, s_true, x_true = generate_data(M, N_full, active_features, sigma_x, sigma)
            A_norm, y_norm, feature_is_uniform = normalize_data(A, y)
            M, N = A_norm.shape

            with open('runs.tsv', 'a') as fout:
                active_features = np.sum(s_true[~feature_is_uniform])
                record = [run_id, N, M, replicate_id, active_features,
                          sigma_x, sigma, data_file, result_file]
                fout.write('\t'.join(list(map(str, record))) + '\n')
            with open(data_file, 'wb') as fout:
                dump({'s_true': s_true[~feature_is_uniform],
                      'x_true': x_true[~feature_is_uniform],
                      'A_norm': A_norm,
                      'y_norm': y_norm},
                     fout)

            time_start = timer()

            s_start = (0,) * N
            N1_max = M - 1
            kappa = 2
            pi = 1.0 / N

            log_ps = calculate_s_logprior(N, kappa, pi)

            alpha_grid = np.logspace(np.log10(0.001), np.log10(10), 8)

            results = sweep_alpha(alpha_grid, A_norm, y_norm, s_start, N1_max,
                                  bandwidth=10, adaptive=True)

            posteriors = calculate_posteriors(results, log_ps, full=False)

            with open(result_file, 'wb') as fout:
                dump(posteriors, fout)

            print(f'{timer() - time_start} seconds')

        except:
            print('failed')

        run_id += 1





if __name__ == '__main__':
    main()
