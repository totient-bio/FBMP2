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

import os

try:
    from fbmp2 import fbmp2 as fbmp2
except ImportError:
    import sys

    sys.path.append(
        os.path.dirname(
            os.path.dirname(os.path.realpath(__file__))))
    from fbmp2 import fbmp2 as fbmp2

import numpy as np
# from scipy.special import comb
from itertools import product


def is_approx(A, B, rel_error=1e-5, abs_error_floor=1e-7):
    A = np.array(A)
    B = np.array(B)
    if A.shape == B.shape and A.size == 0:
        return True
    if not (np.isinf(A) == np.isinf(B)).all():
        return False
    if not (A[np.isinf(A)] == B[np.isinf(B)]).all():
        return False
    A = A[~np.isinf(A)]
    B = B[~np.isinf(B)]
    diff = np.abs(A - B)
    size = np.max(np.abs([A, B]), axis=0)
    return (diff <= rel_error * size + abs_error_floor).all()


def generate_test_data(M, N, N_fix, sigma_x, sigma, seed=None):
    if seed is not None:
        np.random.seed(seed)

    # feature matrix
    A = np.zeros([M, N])
    for n in range(N):
        A[np.random.choice(M, size=np.random.choice(np.arange(1, M, 1)),
                           replace=False), n] = 1
    A_fix = np.zeros([M, N_fix])
    for n_fix in range(N_fix):
        A_fix[np.random.choice(M, size=np.random.choice(np.arange(1, M, 1)),
                               replace=False), n_fix] = 1

    # true x coefficients
    active_features = np.random.choice(N, size=np.random.choice(N + 1),
                                       replace=False)
    s_true = np.zeros(N, dtype=bool)
    s_true[active_features] = True
    x_true = np.zeros(N)
    x_true[s_true] = sigma_x * np.random.randn(len(active_features))

    x_fix_true = sigma_x * np.random.randn(N_fix)

    # target
    y = A.dot(x_true) + A_fix.dot(x_fix_true) + sigma * np.random.randn(M)

    return A, A_fix, s_true, x_true, x_fix_true, y


def compute_GHlogLC(A, y, A_fix, s, alpha, a, b):
    M, N = A.shape
    D = np.diag(s)
    phi = A.dot(D).dot(A.T) + A_fix.dot(A_fix.T) + alpha ** 2 * np.eye(M)
    phi_inv = np.linalg.inv(phi)

    G = np.linalg.slogdet(phi)[1]
    H = y.T.dot(phi_inv).dot(y)
    logL = -0.5 * G - (0.5 * M + a) * np.log(b + 0.5 * H)
    C = phi_inv.dot(A)
    return G, H, C, logL


def compute_all(A, y, A_fix, alpha, a, b):
    M, N = A.shape
    s_list = list(product(*[(0, 1) for n in range(N)]))
    G = {}
    H = {}
    C = {}
    logL = {}

    for s in s_list:
        Gs, Hs, Cs, logLs = compute_GHlogLC(A, y, A_fix, s, alpha, a, b)
        G[s] = Gs
        H[s] = Hs
        C[s] = Cs
        logL[s] = logLs
    return s_list, G, H, C, logL


def compute_CN1(A, A_fix, alpha, s_list, s_order_list):
    M, N_fix = A_fix.shape
    CN1 = {}
    for s, s_order in zip(s_list, s_order_list):
        A_active = np.concatenate([A_fix, A[:, s_order]], axis=1)
        AAs = A_active.T.dot(A)

        psi = A_active.T.dot(A_active) + alpha**2 * np.eye(N_fix + sum(s))
        CN1s = np.linalg.inv(psi).dot(AAs)
        CN1[s] = CN1s
    return CN1


def neighbors(s):
    s_neighbors = []
    for n, sn in enumerate(s):
        s_new = list(s)
        s_new[n] = 1 - sn
        s_neighbors.append(tuple(s_new))
    return s_neighbors


# Here we compute all variables for a simple test case
seed = 123456789
M, N = 7, 10
N_fix = 2
A, A_fix, s_true, x_true, x_fix_true, y = \
    generate_test_data(M, N, N_fix, 1.0, 0.1, seed=seed)
alpha_grid = [0.1, 1.0]
a = 0
b = 0
G_list = []
H_list = []
C_list = []
logL_list = []

for alpha in alpha_grid:
    s_list, Galpha, Halpha, Calpha, logLalpha = \
        compute_all(A, y, A_fix, alpha, a, b)
    G_list.append(Galpha)
    H_list.append(Halpha)
    C_list.append(Calpha)
    logL_list.append(logLalpha)

# The following variables are needed only for testing NodeN1 and ExtensionN1
Ayfix = A_fix.T.dot(y)
AAdiag = np.diag(A.T.dot(A))
Ay = A.T.dot(y)
s_order_list = []
AAs = {}
Ays = {}
for s in s_list:
    s_order = np.argwhere(np.array(s) == 1)[:, 0]
    np.random.shuffle(s_order)
    s_order_list.append(list(s_order))
    A_active = np.concatenate([A_fix, A[:, s_order]], axis=1)
    AAs[s] = A_active.T.dot(A)
    Ays[s] = np.concatenate([Ayfix, Ay[s_order]], axis=0)
CN1_list = []
for alpha in alpha_grid:
    CN1alpha = compute_CN1(A, A_fix, alpha, s_list, s_order_list)
    CN1_list.append(CN1alpha)


def test_NodeM_discover_neighbors():
    bandwidth = 5
    for alpha_idx, alpha in enumerate(alpha_grid):
        G = G_list[alpha_idx]
        H = H_list[alpha_idx]
        C = C_list[alpha_idx]
        logL = logL_list[alpha_idx]
        for s_idx, s in enumerate(s_list[::-1]):
            node = fbmp2.NodeM(np.array(s),
                               G[s],
                               H[s],
                               C[s])
            logL_quest, new_forward_extensions = \
                node.discover_neighbors(A, y, bandwidth, a, b)

            for s_new, logLquestn in zip(neighbors(s), logL_quest):
                assert is_approx(logLquestn, logL[s_new])

            forward_idxs = []
            forward_s = []
            for n, s_new in enumerate(neighbors(s)):
                if s_new[n] == 1:
                    forward_idxs.append(n)
                    forward_s.append(s_new)
            n_logL_pairs = sorted([(n, logL[s_new])
                                   for (n, s_new) in
                                   zip(forward_idxs, forward_s)],
                                  key=lambda t: t[1], reverse=True)
            selected_n_logL_pairs = n_logL_pairs[:bandwidth]

            assert set([ext.n for ext in new_forward_extensions]) == \
                set([nlogL[0] for nlogL in selected_n_logL_pairs])


def test_ExtensionM_get_target_node():
    for alpha_idx, alpha in enumerate(alpha_grid):
        G = G_list[alpha_idx]
        H = H_list[alpha_idx]
        C = C_list[alpha_idx]
        logL = logL_list[alpha_idx]
        for s_idx, s in enumerate(s_list):
            for n in range(N):
                s_new = list(s)
                s_new[n] = 1 - s_new[n]
                s_new = tuple(s_new)
                beta_n = 1.0 / (1 + (-1) ** s[n] * A[:, n].dot(C[s][:, n]))
                extension = fbmp2.ExtensionM(np.array(s),
                                             n,
                                             C[s],
                                             beta_n,
                                             G[s_new],
                                             H[s_new],
                                             logL[s_new])
                new_node = extension.get_target_node(A)
                assert (new_node.s == np.array(s_new)).all()
                assert is_approx(new_node.G, G[s_new])
                assert is_approx(new_node.H, H[s_new])
                assert is_approx(new_node.C, C[s_new])


def test_NodeN1_discover_neighbors():
    bandwidth = 5
    for alpha_idx, alpha in enumerate(alpha_grid):
        G = G_list[alpha_idx]
        H = H_list[alpha_idx]
        CN1 = CN1_list[alpha_idx]
        logL = logL_list[alpha_idx]
        for s_idx, s in enumerate(s_list):
            node = fbmp2.NodeN1(np.array(s),
                                G[s],
                                H[s],
                                CN1[s],
                                AAs[s],
                                Ays[s])
            logL_quest, new_forward_extensions = \
                node.discover_neighbors(A, AAdiag, Ay, alpha**2,
                                        bandwidth, a, b)

            for s_new, logLquestn in zip(neighbors(s), logL_quest):
                assert is_approx(logLquestn, logL[s_new])

            forward_idxs = []
            forward_s = []
            for n, s_new in enumerate(neighbors(s)):
                if s_new[n] == 1:
                    forward_idxs.append(n)
                    forward_s.append(s_new)
            n_logL_pairs = sorted([(n, logL[s_new])
                                   for (n, s_new) in
                                   zip(forward_idxs, forward_s)],
                                  key=lambda t: t[1], reverse=True)
            selected_n_logL_pairs = n_logL_pairs[:bandwidth]

            assert set([ext.n for ext in new_forward_extensions]) == \
                set([nlogL[0] for nlogL in selected_n_logL_pairs])


def test_ExtensionN1_get_target_node():
    for alpha_idx, alpha in enumerate(alpha_grid):
        G = G_list[alpha_idx]
        H = H_list[alpha_idx]
        CN1 = CN1_list[alpha_idx]
        C = C_list[alpha_idx]
        logL = logL_list[alpha_idx]
        for s_idx, s in enumerate(s_list):
            for n in range(N):
                # skipping backward extensions
                if s[n] == 1:
                    continue
                s_new = list(s)
                s_new[n] = 1 - s_new[n]
                s_new = tuple(s_new)
                beta_n = 1.0 / (1 + (-1) ** s[n] * A[:, n].dot(C[s][:, n]))
                extension = fbmp2.ExtensionN1(np.array(s),
                                              n,
                                              CN1[s],
                                              AAs[s],
                                              Ays[s],
                                              beta_n / alpha**2,
                                              G[s_new],
                                              H[s_new],
                                              logL[s_new])
                new_node = extension.get_target_node(A, y)
                assert (new_node.s == np.array(s_new)).all()
                assert is_approx(new_node.G, G[s_new])
                assert is_approx(new_node.H, H[s_new])
                assert new_node.C.shape == CN1[s_new].shape


def test_adaptive_band_search_in_Mspace():
    s_start = [0] * N
    N1_max = N
    bandwidth = 10
    adaptive = True
    for alpha_idx, alpha in enumerate(alpha_grid):
        fbmp2.band_search(alpha,
                          A,
                          y,
                          A_fix,
                          s_start,
                          N1_max,
                          bandwidth,
                          adaptive,
                          a,
                          b,
                          'M-space')
        # results['s'] = np.frombuffer(b''.join(results['s']),
        #                              dtype=np.uint8) \
        #     .reshape([len(results['s']), N])
        # results['N1'] = np.sum(results['s'], axis=1)
        # for N1 in range(0, N + 1, 1):
        #     sel = (results['N1'] == N1)
        #     if np.sum(sel) < comb(N, N1):
        #         s = results['s'][sel, :]
        #         for act in (0, 1):
        #             for n in range(N):
        #                 assert np.sum(s[:, n] == act) >= bandwidth


def test_intialize_NodeN1():
    for alpha_idx, alpha in enumerate(alpha_grid):
        G = G_list[alpha_idx]
        H = H_list[alpha_idx]
        for s_start in s_list:
            node = fbmp2.initialize_nodeN1(alpha, A, y, A_fix,
                                           np.array(s_start))
            assert is_approx(node.G, G[s_start])
            assert is_approx(node.H, H[s_start])

        CN1 = CN1_list[alpha_idx]
        s_start = (0,) * N
        node = fbmp2.initialize_nodeN1(alpha, A, y, A_fix, np.array(s_start))
        assert is_approx(node.C, CN1[s_start])


def test_adaptive_band_search_in_N1space():
    s_start = [0] * N
    N1_max = N
    bandwidth = 10
    adaptive = True
    for alpha_idx, alpha in enumerate(alpha_grid):
        fbmp2.band_search(alpha,
                          A,
                          y,
                          A_fix,
                          s_start,
                          N1_max,
                          bandwidth,
                          adaptive,
                          a,
                          b,
                          'N1-space')
        # results['s'] = np.frombuffer(b''.join(results['s']),
        #                              dtype=np.uint8) \
        #     .reshape([len(results['s']), N])
        # results['N1'] = np.sum(results['s'], axis=1)
        # for N1 in range(0, N + 1, 1):
        #     sel = (results['N1'] == N1)
        #     if np.sum(sel) < comb(N, N1):
        #         s = results['s'][sel, :]
        #         for act in (0, 1):
        #             for n in range(N):
        #                 assert np.sum(s[:, n] == act) >= bandwidth


def test_sweep_alpha():
    s_start = [0] * N
    N1_max = N
    bandwidth = 10
    adaptive = True

    for algorithm in ('M-space', 'N1-space'):
        results = []
        for alpha_idx, alpha in enumerate(alpha_grid):
            results.append(fbmp2.band_search(alpha,
                                             A,
                                             y,
                                             A_fix,
                                             s_start,
                                             N1_max,
                                             bandwidth,
                                             adaptive,
                                             a,
                                             b,
                                             algorithm)
                           )

        results_quest = fbmp2.sweep_alpha(alpha_grid, A, y, A_fix,
                                          s_start, N1_max, bandwidth, adaptive,
                                          a, b, algorithm,
                                          max_processes=None)

        results.sort(key=lambda res: res['alpha'])
        for alpha_idx, alpha in enumerate(alpha_grid):
            assert results_quest[alpha_idx]['alpha'] == \
                results[alpha_idx]['alpha']
            assert (results_quest[alpha_idx]['s'] == results[alpha_idx]['s'])
            assert is_approx(results_quest[alpha_idx]['logL'],
                             results[alpha_idx]['logL'])


def test_calculate_posteriors():
    kappa = 2
    pi = 1.0 / N
    log_ps = fbmp2.calculate_s_logprior(N, kappa, pi)

    s_start = [0] * N
    N1_max = M - 1
    bandwidth = 10
    adaptive = True
    sweep_alpha_results = fbmp2.sweep_alpha(alpha_grid, A, y, A_fix,
                                            s_start, N1_max, bandwidth,
                                            adaptive, a, b,
                                            'N1-space',
                                            max_processes=None)

    Q = np.zeros(len(sweep_alpha_results))
    for alpha_idx, res in enumerate(sweep_alpha_results):
        res['s'] = np.frombuffer(b''.join(res['s']), dtype=np.uint8) \
            .reshape([len(res['s']), N])
        res['logL'] = np.array(res['logL'])
        res['N1'] = np.sum(res['s'], axis=1).astype(int)
        L = np.exp(res['logL'] - np.max(res['logL'])) * \
            np.exp(np.max(res['logL']))
        p = np.exp(log_ps[res['N1']] - np.max(log_ps)) * \
            np.exp(np.max(log_ps))
        pL = p * L
        res['pL'] = pL
        Q[alpha_idx] = np.sum(pL)
    Q = Q / np.sum(Q)

    p_sn0 = np.zeros([len(sweep_alpha_results), N])
    p_sn1 = np.zeros([len(sweep_alpha_results), N])
    p_N1 = np.zeros([len(sweep_alpha_results), N + 1])
    for alpha_idx, res in enumerate(sweep_alpha_results):
        pL = res['pL']
        Q_alpha = Q[alpha_idx]
        for n in range(N):
            p_sn0[alpha_idx, n] = Q_alpha * np.sum(pL[res['s'][:, n] == 0])
            p_sn1[alpha_idx, n] = Q_alpha * np.sum(pL[res['s'][:, n] == 1])

        for K in range(np.max(res['N1']) + 1):
            p_N1[alpha_idx, K] = Q_alpha * np.sum(pL[res['N1'] == K])

    pn_0 = np.sum(p_sn0, axis=0)
    pn_1 = np.sum(p_sn1, axis=0)
    pn_0 = pn_0 / (pn_0 + pn_1)

    pN1 = np.sum(p_N1, axis=0)
    pN1 = pN1 / np.sum(pN1)

    post_quest = fbmp2.calculate_posteriors(sweep_alpha_results, log_ps,
                                            A, y, A_fix,
                                            True, True)

    assert is_approx(alpha_grid, post_quest['alpha'])
    assert is_approx(Q, np.exp(post_quest['logQ']))
    assert is_approx(pn_0, np.exp(post_quest['log_pn_0']))
    assert is_approx(pN1, np.exp(post_quest['log_pN1']))

    selected_s, selected_x, selected_x_fix = \
        fbmp2.estimate_coefficients(sweep_alpha_results,
                                    post_quest['logQ'],
                                    log_ps,
                                    A, y, A_fix,
                                    test=True)
    best_alpha = np.exp(np.sum(np.log(post_quest['alpha']) *
                               np.exp(post_quest['logQ'])))
    for s, x, x_fix in zip(selected_s, selected_x, selected_x_fix):
        A_active = np.concatenate([A_fix, A.dot(np.diag(s))], axis=1)
        expected = A_active.T.dot(
            np.linalg.inv(
                A_active.dot(A_active.T) + best_alpha**2 * np.eye(M)
            )
        ).dot(y)
        received = np.concatenate([x_fix, x], axis=0)
        assert is_approx(expected, received)


def test_normalize_input():
    A_nonuniform = np.arange(12).reshape((4, 3))
    uniform_col = np.ones((4, 1))
    A = np.concatenate((-1 * uniform_col,
                        A_nonuniform[:, 0:1],
                        -2 * uniform_col,
                        A_nonuniform[:, 1:3],
                        -3 * uniform_col), axis=1)
    A_fix_nonuniform = np.arange(4).reshape((4, 1))
    A_fix = np.concatenate((-1 * uniform_col,
                            A_fix_nonuniform[:, 0:1],
                            -2 * uniform_col), axis=1)
    y = np.arange(4)

    results = fbmp2.normalize_input(A, y, A_fix,
                                    normalize_feature_scales=True)

    assert results['A_norm'].shape == (4, 3)
    assert is_approx(np.mean(results['A_norm'], axis=0), np.zeros(3))
    assert is_approx(np.std(results['A_norm'], axis=0), np.ones(3))

    assert results['A_fix_norm'].shape == (4, 1)
    assert is_approx(np.mean(results['A_fix_norm'], axis=0), np.zeros(1))
    assert is_approx(np.std(results['A_fix_norm'], axis=0), np.ones(1))

    assert is_approx(np.mean(results['y_norm']), np.array(0.0))
    assert is_approx(np.std(results['y_norm']), np.array(1.0))

    assert (results['uniform_features'] ==
            np.array([True, False, True, False, False, True])).all()
    assert (results['fix_uniform_features'] ==
            np.array([True, False, True])).all()


def test_normalize_input_centering_only():
    A_nonuniform = np.arange(12).reshape((4, 3))
    uniform_col = np.ones((4, 1))
    A = np.concatenate((-1 * uniform_col,
                        A_nonuniform[:, 0:1],
                        -2 * uniform_col,
                        A_nonuniform[:, 1:3],
                        -3 * uniform_col), axis=1)
    A_fix_nonuniform = np.arange(4).reshape((4, 1))
    A_fix = np.concatenate((-1 * uniform_col,
                            A_fix_nonuniform[:, 0:1],
                            -2 * uniform_col), axis=1)
    y = np.arange(4)

    results = fbmp2.normalize_input(A, y, A_fix,
                                    normalize_feature_scales=False)

    assert results['A_norm'].shape == (4, 3)
    assert is_approx(np.mean(results['A_norm'], axis=0), np.zeros(3))

    assert results['A_fix_norm'].shape == (4, 1)
    assert is_approx(np.mean(results['A_fix_norm'], axis=0), np.zeros(1))

    assert is_approx(np.mean(results['y_norm']), np.array(0.0))
    assert is_approx(np.std(results['y_norm']), np.array(1.0))

    assert (results['uniform_features'] ==
            np.array([True, False, True, False, False, True])).all()
    assert (results['fix_uniform_features'] ==
            np.array([True, False, True])).all()


def test_normalize_input_empty_A_fix():
    A_nonuniform = np.arange(12).reshape((4, 3))
    uniform_col = np.ones((4, 1))
    A = np.concatenate((-1 * uniform_col,
                        A_nonuniform[:, 0:1],
                        -2 * uniform_col,
                        A_nonuniform[:, 1:3],
                        -3 * uniform_col), axis=1)
    A_fix = np.zeros([4, 0])
    y = np.arange(4)

    results = fbmp2.normalize_input(A, y, A_fix,
                                    normalize_feature_scales=True)

    assert results['A_norm'].shape == (4, 3)
    assert is_approx(np.mean(results['A_norm'], axis=0), np.zeros(3))
    assert is_approx(np.std(results['A_norm'], axis=0), np.ones(3))

    assert results['A_fix_norm'].shape == (4, 0)
    assert is_approx(results['A_fix_norm'], A_fix)

    assert is_approx(np.mean(results['y_norm']), np.array(0.0))
    assert is_approx(np.std(results['y_norm']), np.array(1.0))

    assert (results['uniform_features'] ==
            np.array([True, False, True, False, False, True])).all()
    assert (results['fix_uniform_features'] ==
            np.array([])).all()


def test_unnormalize_posterior():
    A_nonuniform = np.arange(12).reshape((4, 3))
    uniform_col = np.ones((4, 1))
    A = np.concatenate((-1 * uniform_col,
                        A_nonuniform[:, 0:1],
                        -2 * uniform_col,
                        A_nonuniform[:, 1:3],
                        -3 * uniform_col), axis=1)
    A_fix_nonuniform = np.arange(4).reshape((4, 1))
    A_fix = np.concatenate((-1 * uniform_col,
                            A_fix_nonuniform[:, 0:1],
                            -2 * uniform_col), axis=1)
    y = np.arange(4)

    normalization_results = fbmp2.normalize_input(
        A, y, A_fix, normalize_feature_scales=True)
    A_norm = normalization_results['A_norm']
    y_norm = normalization_results['y_norm']
    A_fix_norm = normalization_results['A_fix_norm']
    M, N = A_norm.shape

    kappa = 2
    pi = 1.0 / N
    log_ps = fbmp2.calculate_s_logprior(N, kappa, pi)

    s_start = [0] * N
    N1_max = M - 1
    bandwidth = 1
    adaptive = False
    sweep_alpha_results = fbmp2.sweep_alpha(alpha_grid, A_norm, y_norm, A_fix,
                                            s_start, N1_max, bandwidth,
                                            adaptive, a, b,
                                            'N1-space',
                                            max_processes=None)

    post = fbmp2.calculate_posteriors(sweep_alpha_results, log_ps,
                                      A_norm, y_norm, A_fix_norm,
                                      True, True)
    unnormed_post = fbmp2.unnormalize_posteriors(post, normalization_results,
                                                 kappa, pi)

    assert is_approx(unnormed_post['alpha'], post['alpha'])
    assert is_approx(unnormed_post['logQ'], post['logQ'])
    assert len(unnormed_post['log_pn_0']) == 6
    assert is_approx(unnormed_post['log_pn_0']
                     [~normalization_results['uniform_features']],
                     post['log_pn_0'])
    s_matrix = unnormed_post['full_log_psy_df'][
        [f's_{n}' for n in range(6)]].values
    assert (np.isnan(s_matrix[:, normalization_results['uniform_features']]))\
        .all()
    assert (s_matrix[:, ~normalization_results['uniform_features']] ==
            post['full_log_psy_df'][[f's_{n}' for n in range(3)]].values).all()
    assert is_approx(unnormed_post['full_log_psy_df']['log_psy'].values,
                     post['full_log_psy_df']['log_psy'].values)
    assert (np.isnan(unnormed_post['x_mmse']) ==
            np.array([True, False, True, False, False, True])).all()
    assert is_approx(unnormed_post['x_mmse'][[1, 3, 4]],
                     post['x_mmse'] * normalization_results['y_scale'] /
                     normalization_results['A_scale'][[1, 3, 4]])
    assert (np.isnan(unnormed_post['x_fix_mmse']) ==
            np.array([True, False, True])).all()
    assert is_approx(unnormed_post['x_fix_mmse'][[1]],
                     post['x_fix_mmse'] * normalization_results['y_scale'] /
                     normalization_results['A_fix_scale'][[1]])
