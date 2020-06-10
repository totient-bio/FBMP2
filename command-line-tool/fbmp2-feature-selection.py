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

"""fbmp2-feature-selection.py

Command line tools for running FBMP2 feature selection with default behavior.

Feature selection is run for all target vectors independently, and the results
are reported in separate columns in the output TSV files.

Usage:

    python3 fbmp2-feature-selection.py \
        --xy_data  <xy_data.tsv(.gz)>  \
        --column_types  <column_types.tsv(.gz)>  \
        --alpha_grid <alpha_grid> \
        [--a <a>] \
        [--b <b>] \
        [--normalize_x_scales <normalize_x_scales>] \
        [--lambda_pseudocount <lambda_prior_pseudocount>] \
        [--lambda_mean <lambda_prior_mean>] \
        [--bandwidth <bandwidth>] \
        [--adaptive <adaptive>] \
        [--algorithm] <algorithm> \
        [--max_active] <max_active> \
        [--out_prefix <out_prefix>]


Parameters:

    <xy_data.tsv(.gz)>      : input TSV where columns contain features (X) and targets (Y)
    <column_types.tsv(.gz)> : TSV(.GZ) file with column ("column", "type")
                              describing the type of each column in
                              training_data, i.e. "index", "feature",
                              "fixed_feature", "target", or "ignore".
    <alpha_grid> : list of alpha (noise / std_coeffs) values to consider
    <a>: "a" parameter of inverse-gamma prior of sigma_x^2
    <b>: "b" parameter of inverse-gamma prior of sigma_x^2
    <normalize_x_scales>: "True" of "False"
    <lambda_pseudocount>: pseudocount of Beta prior for lambda
    <lambda_mean>: mean of Beta prior for lambda
    <bandwidth>  : bandwidth of FBMP2's band search
    <adaptive>   : "True" or "False", if "True" FBMP2's adaptive search algorithm is used
    <algorithm>  : "M-space" or "N1-space"
                   M-space algorithm performs recursion in sample space,
                   and it usuall slower. Preferred only if
                   max_active > 0.5 * [# of samples].
                   N1-space algorithm performs the recursion in the space of
                   active features. It is much faster at the beginning, when
                   the [# of active features] << [# of samples], but starts to
                   slow down around [# of active features] = 0.5 * [# of samples]
    <max_active> : Maximal number of active features to consider
    <out_prefix> : prefix to add to the output files.

Output files:

    <out_prefix>_post_alpha.tsv: posterior distribution of the alpha parameter
    <out_prefix>_post_feautures.tsv: posterior probability of each "feature"
        being active
    <out_prefix>_post_number_of_actives.tsv: posterior distribution of
        number of active features (not counting "fixed_features")
    <out_prefix>_post_coeffs.tsv: posterior mean of the linear coefficients of
        "features" and "fixed_features"
"""

import sys
import argparse
import numpy as np
import pandas as pd

from fbmp2.feature_selection import FeatureSelection

def preprocess(data_file, column_types_file):

    # Load column metadata
    df_columns = pd.read_csv(column_types_file, sep='\t')
    assert list(df_columns.columns) == ['column', 'type']
    assert set(df_columns['type']).issubset({'index',
                                             'feature', 'fixed_feature',
                                             'target', 'ignore'}),  \
        "only 'index', 'feature', 'fixed_feature', 'target', and 'ignore' " \
        "are allowed column types"
    index_cols = list(df_columns[df_columns['type']=='index']['column'])
    feature_cols = list(df_columns[df_columns['type']=='feature']['column'])
    fixed_feature_cols = list(df_columns[df_columns['type']
                                         == 'fixed_feature']['column'])
    target_cols = list(df_columns[df_columns['type']=='target']['column'])
    ignore_cols = list(df_columns[df_columns['type']=='ignore']['column'])
    all_cols = index_cols + feature_cols + fixed_feature_cols + \
               target_cols + ignore_cols
    assert len(feature_cols) > 0, 'No features found'
    assert len(target_cols) > 0, 'No targets found'
    assert len(set(all_cols)) == len(all_cols),  \
        f'Duplicate columns in column_type TSV input'

    # Load data
    df_XY = pd.read_csv(data_file, sep='\t')
    assert set(df_XY.columns) == set(all_cols),  \
        f'Unspecified columns: {set(df_XY.columns) - set(all_cols)},  \
          Missing columns: {set(all_cols) - set(df_XY.columns)}'
    if len(index_cols) > 0:
        df_XY.set_index(index_cols, inplace=True)
    df_XY = df_XY[feature_cols + fixed_feature_cols + target_cols]

    # drop incomplete rows
    complete_rows = df_XY.notnull().all(1)
    if np.sum(complete_rows) < len(df_XY):
        incomplete_row_idxs = np.where(~complete_rows)[0]
        print('Rows ignored due to missing one or more entries: ',
              file=sys.stderr)
        print('\n'.join(map(str, incomplete_row_idxs)), file=sys.stderr)
    df_XY = df_XY[complete_rows]


    return df_XY, feature_cols, fixed_feature_cols, target_cols


def run_fbmp2_feature_selection(X, y, X_fix, alpha_grid, a, b,
                                normalize_X_scale,
                                lambda_pseudocount, lambda_ev,
                                bandwidth,
                                adaptive, max_active):
    model = FeatureSelection(X, y, X_fix, normalize_X_scale)
    model.compute_likelihoods(alpha_grid, a, b,
                              bandwidth=bandwidth,
                              adaptive=adaptive,
                              max_active=max_active)
    posterior = model.compute_posterior(lambda_ev=lambda_ev,
                                        lambda_pseudocount=lambda_pseudocount,
                                        estimate_coeffs=True,
                                        full=False)

    return posterior


def main(args):
    df_XY, feature_cols, fixed_feature_cols, target_cols = \
        preprocess(args.xy_data, args.column_types)
    alpha_grid = [float(alpha) for alpha in args.alpha_grid.split(',')]
    lambda_pseudocount = args.lambda_pseudocount
    if lambda_pseudocount is not None:
        lambda_pseudocount = float(lambda_pseudocount)
    lambda_ev = args.lambda_mean
    if lambda_ev is not None:
        lambda_ev = float(lambda_ev)
    bandwidth = int(args.bandwidth)
    adaptive = True if args.adaptive in {"true", "TRUE", "True"} else False
    max_active = args.max_active
    if max_active is not None:
        max_active = int(max_active)
    a = args.a
    if a < 0.0:
        raise ValueError('command line parameter `a` must be non-negative')
    b = args.b
    if b < 0.0:
        raise ValueError('command line parameter `b` must be non-negative')
    normalize_x_scales = args.normalize_x_scales
    if normalize_x_scales not in {'True', 'False'}:
        raise ValueError('command line parameter `normalize_x_scales` '
                         'must be "True" or "False".')
    normalize_x_scales = (normalize_x_scales == 'True')

    df_palpha = pd.DataFrame({'alpha': alpha_grid})
    df_ps1 = pd.DataFrame({'feature': feature_cols})
    df_pN1 = pd.DataFrame({'active_features': np.arange(len(feature_cols) + 1)})
    df_coeff = pd.DataFrame({'feature': feature_cols + fixed_feature_cols})
    X = df_XY[feature_cols].values
    X_fix = df_XY[fixed_feature_cols].values  # can be an empty numpy array

    for target in target_cols:
        print(f'Running on {target} ... ', end='', file=sys.stderr)
        y = df_XY[target].values
        posterior = run_fbmp2_feature_selection(X, y, X_fix,
                                                alpha_grid,
                                                a, b,
                                                normalize_x_scales,
                                                lambda_pseudocount,
                                                lambda_ev,
                                                bandwidth,
                                                adaptive,
                                                max_active)
        df_palpha[target] = np.exp(posterior['logQ'])
        df_ps1[target] = 1.0 - np.exp(posterior['log_pn_0'])
        df_pN1[target] = np.exp(posterior['log_pN1'])
        df_coeff[target] = np.concatenate([posterior['x_mmse'],
                                           posterior['x_fix_mmse']], axis=0)
        print('Done', file=sys.stderr)

    options = {'sep': '\t', 'index': False}
    df_palpha.to_csv(args.out_prefix + '_post_alpha.tsv', **options)
    df_ps1.to_csv(args.out_prefix + '_post_features.tsv', **options)
    df_pN1.to_csv(args.out_prefix + '_post_number_of_actives.tsv', **options)
    df_coeff.to_csv(args.out_prefix + '_post_coeffs.tsv', **options)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Command line tools for running FBMP2 feature selection "
                    "with default behavior.")
    parser.add_argument('--xy_data', help='TSV(.GZ) of features and targets',
                        required=True)
    parser.add_argument('--column_types', help='TSV(.GZ) column metadata',
                        required=True)
    parser.add_argument('--alpha_grid',
                        help='comma-separated list of alpha values',
                        required=True)
    parser.add_argument('--a',
                        help='"a" parameter of inverse-gamma prior '
                             'for sigma_x^2 [float, default: 0]',
                        required=False, type=float, default=0.0)
    parser.add_argument('--b',
                        help='"b" parameter of inverse-gamma prior '
                             'for sigma_x^2 [float, default: 0]',
                        required=False, type=float, default=0.0)
    parser.add_argument('--normalize_x_scales',
                        help='If "True", scales of features '
                             'will be normalized with their stdevs. '
                             'If "False", they will only be centered. '
                             '(For binary features False is more appropriate.)'
                             '["True" of "False", default: "True"]',
                        required=False, default='True')
    parser.add_argument('--lambda_pseudocount',
                        help='pseudocount of Beta prior for lambda '
                             '[default: 0.2254 * [# of features]]',
                        required=False, default=None)
    parser.add_argument('--lambda_mean',
                        help='mean of Beta prior for lambda '
                             '[default: 1 / [# of features]]',
                        required=False, default=None)
    parser.add_argument('--out_prefix',
                        help='prefix to add to the output file names',
                        required=True)
    parser.add_argument('--bandwidth',
                        help="bandwidth of FBMP2's band search",
                        required=False, default="10")
    parser.add_argument('--adaptive',
                        help='"True" or "False", if "True" the adaptive search algorithm is used',
                        required=False, default="True")
    parser.add_argument('--max_active',
                        help='Maximal number of active features to consider',
                        required=False, default=None)
    parser.add_argument('--algorithm',
                        help='Maximal number of active features to consider',
                        required=False, default="N1-space")

    args = parser.parse_args()

    main(args)
