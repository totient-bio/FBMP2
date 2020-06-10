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

import numpy as np
from scipy.special import betaln, logsumexp, gammaln
from sortedcontainers import SortedList
from multiprocessing import Pool, cpu_count
from functools import partial
import pandas as pd
from warnings import warn


def _betabinomial_logpmf(k, n, a, b):
    # See https://en.wikipedia.org/wiki/Beta-binomial_distribution
    return (gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
            - gammaln(n + a + b) + gammaln(k + a) + gammaln(n - k + b)
            + gammaln(a + b) - gammaln(a) - gammaln(b))


def _regularized_regression(A_active, y, alpha):
    A = A_active
    N1 = A.shape[1]
    x_mmse = (np.linalg.inv(A.T.dot(A) + alpha ** 2 * np.eye(N1))) \
        .dot(A.T.dot(y))
    return x_mmse


def normalize_input(A, y, A_fix, normalize_feature_scales):
    """Prepares input feature matrices A, A_fix and target vector y for FBMP

    Target and features get centered and their scales get  normalized
    with the emprical standard deviation.
    And uniform features are removed from the A and A_fix matrices.

    Args:
        A (numpy.ndarray): the feature matrix [A_{m,n}]
        y (numpy.ndarray): target vector [y_m]
        A_fix (numpy.ndarray): fixed feature matrix [Afix_{m,n_fix}]]
        normalize_scales (bool): if False, features are only centered,
            but their scales are not normalized, which is appropriate for
            binary features.

    Returns:
        dict: dict containing

            "A_norm": normalized feature matrix (with uniform features removed)
            "A_center": centers of all features
            "A_scale": scales of all features
            "y_norm": normalized target vector
            "y_center": center of target vector
            "y_scale": scale of target vector
            "uniform_features": boolean numpy vector marking uniform features
            "A_fix_norm": normalized feature matrix
            "A_fix_center": centers of all features
            "A_fix_scale": scales of all features
            "fix_uniform_features": boolean numpy vector marking
                uniform fixed features

    """
    y_mean = np.mean(y)
    y_std = np.std(y, axis=0)
    if y_std == 0:
        raise ValueError("input y vector is uniform")
    y_norm = (y - y_mean) / y_std

    A_mean = np.mean(A, axis=0)
    A_std = np.std(A, axis=0)
    if (A_std == 0).all():
        raise ValueError("all columns of input A matrix are uniform")
    feature_is_uniform = (A_std == 0)
    A_useful = A[:, ~feature_is_uniform]
    A_scale = A_std if normalize_feature_scales else np.ones_like(A_std)
    A_norm = ((A_useful - A_mean[~feature_is_uniform])
              / A_scale[~feature_is_uniform])

    if A_fix.shape[1] > 0:
        A_fix_mean = np.mean(A_fix, axis=0)
        A_fix_std = np.std(A_fix, axis=0)
        if (A_fix_std == 0).all():
            warn("all columns of input A_fix matrix are uniform")
        fix_feature_is_uniform = (A_fix_std == 0)
        A_fix_useful = A_fix[:, ~fix_feature_is_uniform]
        A_fix_scale = A_fix_std if normalize_feature_scales \
            else np.ones_like(A_fix_std)
        A_fix_norm = ((A_fix_useful - A_fix_mean[~fix_feature_is_uniform])
                      / A_fix_scale[~fix_feature_is_uniform])
    else:
        A_fix_mean = np.zeros(0)
        A_fix_scale = np.zeros(0)
        A_fix_norm = A_fix
        fix_feature_is_uniform = np.zeros(0, dtype=bool)

    return {'A_norm': A_norm, 'y_norm': y_norm, 'A_fix_norm': A_fix_norm,
            'A_center': A_mean, 'y_center': y_mean, 'A_fix_center': A_fix_mean,
            'A_scale': A_scale, 'y_scale': y_std, 'A_fix_scale': A_fix_scale,
            'uniform_features': feature_is_uniform,
            'fix_uniform_features': fix_feature_is_uniform}


def compute_logL(G, H, M, a, b):
    """Calculates ln(L_n^{next}(s)) from Eq. (9) for one s or a list of ss

    Args:
        G (Union[float, numpy.ndarray]): G(s), defined as ln(det(Phi(s))
        H (Union[float, numpy.ndarray]): H(s), defined as y^T[Phi(s)]^{-1}y
        M (int): number of samples
        a (float): "a" parameter of inverse-gamma prior of sigma_x^2 (>= 0)
        b (float): "b" parameter of inverse-gamma prior of sigma_x^2 (>= 0)

    Returns:
        Union[float, numpy.ndarray]: value of ln(L_n^{next}(s))

    """
    return - 0.5 * G - (0.5 * M + a) * np.log(b + 0.5 * H)


class ExtensionM:
    """Represents the a possible extension from current s to its neighbor s'

    This is a sample-space (M), aka. Phi-based extension, which operates on
    c_k(s) vectors of size M.

    Args:
        s (numpy.ndarray): binary s-vector of current node, s
        n (int): index of feature whose activity is changed between s and s'
        C (numpy.ndarray): C(s) matrix of current node s [size: (M, N)]
        beta_n (float): beta(s)_n of the current node s
        G (float): G(s') = ln(det(Phi(s')) of the target node s'
        H (float): H(s') = y^T[Phi(s')]^{-1}y of the target node s'
        logL (float): ln(L(s')) of the target node s'

    """

    def __init__(self, s, n, C, beta_n, G, H, logL):
        self.s = s
        self.n = n
        self.C = C
        self.beta_n = beta_n
        self.G = G
        self.H = H
        self.logL = logL

    def __lt__(self, other):
        return self.logL < other.logL

    def __le__(self, other):
        return self.logL <= other.logL

    def __gt__(self, other):
        return self.logL > other.logL

    def __ge__(self, other):
        return self.logL >= other.logL

    def get_target_node(self, A):
        """Computes C(s'), using Eq. (10), and constructs the target node s'

        Args:
            A (numpy.ndarray): the feature matrix [A_{m,n}]

        Returns:
            :obj:`NodeM`: the target node

        """
        s = self.s.copy()
        n = self.n

        sign_beta_n = (1 - 2 * s[n]) * self.beta_n
        c_n = self.C[:, n]
        C_new = self.C - sign_beta_n * np.einsum('i,k->ik', c_n,
                                                 np.einsum('j,jk->k', c_n, A))
        s[n] = 1 - s[n]
        return NodeM(s, self.G, self.H, C_new)


class NodeM:
    """Stores the values of G, H, C for an s-vector

    This is a sample-space (M), aka. Phi-based node, which is based on
    c_k(s) vectors of size M.

    Args:
        s (numpy.ndarray): binary s-vector
        G (float): G(s) = ln(det(Phi(s))
        H (float): H(s) = y^T[Phi(s)]^{-1}y
        C (numpy.ndarray): C(s) matrix [size: (M, N)]

    """

    def __init__(self, s, G, H, C):
        self.s = s.astype(np.uint8)
        self.G = G
        self.H = H
        self.C = C

    def discover_neighbors(self, A, y, bandwidth, a, b):
        """Computes s', ln(L(s')) of all neighbors and collects extensions

        This is the main function of FBMP2.

            1. Implements the "discovery" equations, Eqs. (6) - (9)
            2. Collects possible forward extensions

        Args:
            A (numpy.ndarray): the feature matrix [A_{m,n}]
            y (numpy.ndarray): target vector [y_m]
            bandwidth (int): number of the best possible extensions to return
            a (float): "a" parameter of inverse-gamma prior of sigma_x^2 (>= 0)
            b (float): "b" parameter of inverse-gamma prior of sigma_x^2 (>= 0)

        Returns:
            tuple: tuple containing:

                logL (numpy.ndarray): discovered ln(L(s)) values

                new_forward_extensions (:obj:`list` of :obj:`ExtensionM`):

                    possible extensions found among the neighbors

        """
        M, N = A.shape
        C = self.C
        s = self.s

        # vectorized computation of Eqs. (6) - (9)
        signs = (1 - 2 * s.astype(int))
        beta = 1.0 / (1 + signs * np.einsum('mn,mn->n', A, C))
        beta = np.max([beta, np.zeros_like(beta)], axis=0)
        G = self.G - np.log(beta)
        H = self.H - signs * beta * (np.einsum('mn,m->n', C, y)) ** 2
        logL = compute_logL(G, H, M, a, b)

        # compute the `bandwidth`th highest ln(L) value among forward neighbors
        forward_idxs = ~(s.astype(bool))
        forward_logL = logL[forward_idxs]
        if len(forward_logL) > bandwidth:
            lowest_best_logL = \
                np.partition(forward_logL, -bandwidth)[-bandwidth]
        else:
            lowest_best_logL = - np.inf

        # compile `bandwidth` number of best forward extensions
        best_fwd_idxs = np.argwhere((logL >= lowest_best_logL)
                                    & (forward_idxs))[:, 0]
        new_forward_extensions = []
        for n in best_fwd_idxs:
            new_forward_extensions.append(
                ExtensionM(s, n, self.C, beta[n],
                           G[n], H[n], logL[n]))

        return logL, new_forward_extensions


class ExtensionN1:
    """Represents the a possible extension from current s to its neighbor s'

    This is a feature-space (N1), aka. Psi-based extension, which operates on
    c_k(s) vectors of size N1(s).

    The extension formula is correct only for extensions when N1(s') > N1(s),
    i.e. "forward" extensions, but not for "backward" extensions.

    s_list, which is the path from starting node to this node is not stored,
    but implied by the row order of C, AAs and Ays.

    Args:
        s (numpy.ndarray): binary s-vector of current node, s
        n (int): index of feature whose activity is changed between s and s'
        C (numpy.ndarray): C(s) matrix of current node s [size: (N1(s), N)]
        AAs (numpy.ndarray): AAs matrix of the current node s [size: N1(s), N]
        Ays (numpy.ndarray): Ays vector of the current node s [size: N1(s)]
        beta_n (float): beta(s)_n of the current node s
        G (float): G(s') = ln(det(Phi(s')) of the target node s'
        H (float): H(s') = y^T[Phi(s')]^{-1}y of the target node s'
        logL (float): ln(L(s')) of the target node s'

    """

    def __init__(self, s, n, C, AAs, Ays, beta_n, G, H, logL):
        self.s = s
        self.n = n
        self.C = C
        self.AAs = AAs
        self.Ays = Ays
        self.beta_n = beta_n
        self.G = G
        self.H = H
        self.logL = logL

    def __lt__(self, other):
        return self.logL < other.logL

    def __le__(self, other):
        return self.logL <= other.logL

    def __gt__(self, other):
        return self.logL > other.logL

    def __ge__(self, other):
        return self.logL >= other.logL

    def get_target_node(self, A, y):
        """Computes C(s'), using Eq. (15), and constructs the target node s'

        This works only for

        Args:
            A (numpy.ndarray): the feature matrix [A_{m,n}]
            y (numpy.ndarray): target vector

        Returns:
            :obj:`NodeN1`: the target node

        """
        M, N = A.shape
        s = self.s.copy()
        n = self.n
        AAs = self.AAs
        Ays = self.Ays
        AAn = A[:, n].dot(A)
        Ayn = A[:, n].dot(y)
        c_n = self.C[:, n]
        C_new = (np.concatenate([self.C, np.zeros([1, N])], axis=0)
                 + self.beta_n
                 * np.einsum('p,n->pn',
                             np.concatenate([c_n, [-1]], axis=0),
                             np.einsum('p,pn->n', c_n, AAs) - AAn)
                 )

        AAs_new = np.concatenate([AAs, AAn.reshape([1, N])], axis=0)
        Ays_new = np.concatenate([Ays, [Ayn]], axis=0)
        s[n] = 1 - s[n]
        return NodeN1(s, self.G, self.H, C_new, AAs_new, Ays_new)


class NodeN1:
    """Stores the values of G, H, C for an s-vector

    This is a feature-space (N2), aka. Psi-based node, which is based on
    c_k(s) vectors of size N1(s).

    s_list, which is the path from starting node to this node is not stored,
    but implied by the row order of C, AAs and Ays.

    Args:
        s (numpy.ndarray): binary s-vector
        G (float): G(s) = ln(det(Phi(s))
        H (float): H(s) = y^T[Phi(s)]^{-1}y
        C (numpy.ndarray): C(s) matrix [size: (N1(s), N)],
            its rows are in s_list order
        AAs (numpy.ndarray): A[:,s_list].T.dot(A) matrix
            where s_list encodes the list of ancestors of the node
            [size: (N1(s), N)]
        Ays (numpy.ndarray): A[:, s_list].T.dot(y) vector
            where s_list encodes the list of ancestors of the node
            [size: N1(s)]

    """

    def __init__(self, s, G, H, C, AAs, Ays):
        self.s = s
        self.G = G
        self.H = H
        self.C = C
        self.AAs = AAs
        self.Ays = Ays

    def discover_neighbors(self, A, AAdiag, Ay, alphasq,
                           bandwidth, a, b):
        """Computes s', ln(L(s')) of all neighbors and collects extensions

        This is the main function of FBMP2.

            1. Implements the "discovery" equations, Eqs. (11) - (14)
            2. Collects possible forward extensions

        Args:
            A (numpy.ndarray): the feature matrix [A_{m,n}]
            AAdiag (numpy.ndarray): diagonal of A.T.dot(A), vector on size N
            Ay (numpy.ndarray): A.T.dot(y) vector of size N
            alphasq (float): alpha^2 hyperparameter
            y (numpy.ndarray): target vector [y_m]
            bandwidth (int): number of the best possible extensions to return
            a (float): "a" parameter of inverse-gamma prior of sigma_x^2 (>= 0)
            b (float): "b" parameter of inverse-gamma prior of sigma_x^2 (>= 0)

        Returns:
            tuple: tuple containing:

                s_new_list (:obj:`list` of :obj:`bytes`): discovered s-vectors

                logL (:obj:`list` of :obj:`float`): discovered ln(L(s)) values

                new_forward_extensions (:obj:`list` of :obj:`ExtensionN1`):

                    possible extensions found among the neighbors

        """
        M, N = A.shape
        s = self.s

        # vectorized computation of Eqs. (11) - (14)
        AAs = self.AAs
        Ays = self.Ays
        C = self.C

        signs = (1 - 2 * s.astype(int))
        beta = 1.0 / (alphasq + signs * (
            AAdiag - np.einsum('pn,pn->n', AAs, C)
        ))
        beta = np.max([beta, np.zeros_like(beta)], axis=0)
        G = self.G - np.log(alphasq * beta)
        H = self.H - (signs / alphasq * beta
                      * (np.einsum('p,pn->n', Ays, C) - Ay) ** 2)
        logL = compute_logL(G, H, M, a, b)

        # compute the `bandwidth`th highest ln(L) value among forward neighbors
        forward_idxs = ~(s.astype(bool))
        forward_logL = logL[forward_idxs]
        if len(forward_logL) > bandwidth:
            lowest_best_logL = \
                np.partition(forward_logL, -bandwidth)[-bandwidth]
        else:
            lowest_best_logL = - np.inf

        # compile `bandwidth` number of best forward extensions
        best_fwd_idxs = np.argwhere((logL >= lowest_best_logL)
                                    & (forward_idxs))[:, 0]
        new_forward_extensions = []
        for n in best_fwd_idxs:
            new_forward_extensions.append(
                ExtensionN1(s, n, C, AAs, Ays,
                            beta[n], G[n], H[n], logL[n]))

        return logL, new_forward_extensions


def initialize_nodeM(alpha, A, y, A_fix, s_start):
    """Computes the G, H, C for the starting node for sample-space recursion.

    Args:
        alpha (float): hyperparameter alpha = sigma / sigma_x
        A (numpy.ndarray): the feature matrix [A_{m,n}]
        y (numpy.ndarray): target vector [y_m]
        A_fix (numpy.ndarray): fixed feature matrix [A_fix_{m,n_fix}]
        s_start (numpy.ndarray): starting s-vector

    Returns:
        :obj: `NodeM`: starting node for the band search

    """
    M, N = A.shape
    Ds_start = np.diag(s_start)
    phi_start = (A.dot(Ds_start).dot(A.T)
                 + A_fix.dot(A_fix.T)
                 + alpha ** 2 * np.eye(M))
    phi_start_inv = np.linalg.inv(phi_start)
    G = np.linalg.slogdet(phi_start)[1]
    H = y.T.dot(phi_start_inv).dot(y)
    C = phi_start_inv.dot(A)

    return NodeM(s_start, G, H, C)


def initialize_nodeN1(alpha, A, y, A_fix, s_start):
    """Computes the G, H, C for the starting node for feature-space recursion.

        Args:
            alpha (float): hyperparameter alpha = sigma / sigma_x
            A (numpy.ndarray): the feature matrix [A_{m,n}]
            y (numpy.ndarray): target vector [y_m]
            A_fix (numpy.ndarray): fixed feature matrix [A_fix_{m,n_fix}]
            s_start (numpy.ndarray): starting s-vector

        Returns:
            :obj: `NodeN1`: starting node for the band search

        """
    s = np.array(s_start, dtype=np.uint8).astype(bool)
    A_active = np.concatenate([A_fix, A[:, s]], axis=1)
    M, N_active = A_active.shape
    AAs = A_active.T.dot(A)
    Ays = A_active.T.dot(y)
    psi = A_active.T.dot(A_active) + alpha**2 * np.eye(N_active)
    psi_inv = np.linalg.inv(psi)
    G = 2*(M - N_active) * np.log(alpha) + np.linalg.slogdet(psi)[1]
    H = 1.0 / alpha**2 * (y.dot(y) - Ays.dot(psi_inv.dot(Ays)))
    C = psi_inv.dot(AAs)

    return NodeN1(s_start, G, H, C, AAs, Ays)


def band_search(alpha, A, y, A_fix, s_start, N1_max, bandwidth, adaptive,
                a, b, algorithm):
    """Performs the forward band search fo FBMP2 from a given s-vector

    Args:
        alpha (float): hyperparameter alpha = sigma / sigma_x
        A (numpy.ndarray): the feature matrix [A_{m,n}]
        y (numpy.ndarray): target vector [y_m]
        A_fix (numpy.ndarray): fixed feature matrix [A_fix_{m,n_fix}]
        s_start (sequence of :obj:`int`): starting s-vector
        N1_max (int): highest total number of active feature to be investigated
        bandwidth (int): If `adaptive`==False, this is the number of extensions
            in each layer of the search. If `adaptive`==True, it is the minimal
            number of times any one feature's s'_n must be found 0 and 1 among
            the set of nodes (s') targeted by the extensions.
        adaptive (bool): If True, the set of extensions is adaptively chosen,
            such that it includes both 0 and 1 for all s'_n values at least
            `bandwidth` number of times.
        a (float): "a" parameter of inverse-gamma prior of sigma_x^2 (>= 0)
        b (float): "b" parameter of inverse-gamma prior of sigma_x^2 (>= 0)
        algorithm (str): "M-space" or "N1-space", which selects
            whether NodeM and ExtensionM, or NodeN1 and ExtensionN1 classes
            are used

    Returns:
        dict: dict containing

            "alpha" (float): input alpha value

            "s" (:obj:`list` of :obj:`bytes`): discovered s vectors

            "logL" (:obj:`list` of :obj:`float`): discovered ln(L(s)) values

    """
    M, N = A.shape
    s_start = np.array(s_start, dtype=np.uint8)
    if algorithm not in {'M-space', 'N1-space'}:
        raise ValueError('argument `algorithm` must be either '
                         '"M-space" or "N1-space"')

    if adaptive:
        # In the adaptive mode, we do not know how many of the possible forward
        # extensions will we need from each discovery. We need to keep all.
        discovery_bandwidth = N
    else:
        # When bandwidth is truly fixed, it's enough to keep the best
        # `bandwidth` from each discovery
        discovery_bandwidth = bandwidth

    # Initialize starting node
    if algorithm == 'M-space':
        starting_node = initialize_nodeM(alpha, A, y, A_fix, s_start)
    elif algorithm == 'N1-space':
        starting_node = initialize_nodeN1(alpha, A, y, A_fix, s_start)
        AAdiag = np.einsum('mn,mn->n', A, A)
        Ay = A.T.dot(y)
        alphasq = alpha**2
    else:
        raise ValueError('argument `algorithm` must be either '
                         '"M-space" or "N1-space"')

    # initialize result containers
    nodes_to_extend = [starting_node]
    results = {bytes(s_start):
               compute_logL(starting_node.G, starting_node.H, M, a, b)}

    # perform layer-by-layer discovery + extension iterations
    for N1 in range(sum(s_start), min(N + 1, N1_max), 1):
        best_forward_extensions = SortedList()  # always sorted

        # We discover the neighbors
        # and keep track of all discovered s, ln(L(s)),
        # and all necessary forward extensions.
        #
        # Crucially, `s_record_set` is also updated in each iteration.
        # This enables avoiding re-computing already discovered nodes
        for node in nodes_to_extend:
            if algorithm == 'M-space':
                new_logLs, new_forward_extensions = \
                    node.discover_neighbors(A, y,
                                            discovery_bandwidth,
                                            a, b)
            elif algorithm == 'N1-space':
                new_logLs, new_forward_extensions = \
                    node.discover_neighbors(A, AAdiag, Ay, alphasq,
                                            bandwidth, a, b)
            else:
                raise ValueError('argument `algorithm` must be either '
                                 '"M-space" or "N1-space"')

            s = node.s
            s_neighbors = list(
                map(bytes,
                    (np.repeat(s[None, :],
                               len(s),
                               axis=0)
                     != np.eye(len(s), dtype=np.uint8)
                     ).astype(np.uint8)
                    )
            )
            results.update(dict(zip(s_neighbors, new_logLs)))
            best_forward_extensions.update(new_forward_extensions)
            if not adaptive:
                # trim list to contain only the best `bandwidth` extensions
                best_forward_extensions = SortedList(
                    best_forward_extensions.islice(-bandwidth, None))

        if adaptive:
            # The adaptive logic works the following way:
            #   - For each feature n, we keep two counters, counting
            #     how many s-vectors we've included in the set of nodes to
            #     extend that have s_n = 0 (and 1, respectively)
            #   - We iterate through the possible extensions in descending
            #     order with respect to their log_L values
            #   - We select an extension only if it moves at least one of the
            #     counters forward that is below the goal count,
            #     i.e. the `bandwidth`

            c0 = np.zeros(N, dtype=int)  # counter for (s_n == 0)
            c1 = np.zeros(N, dtype=int)  # counter for (s_n == 1)
            selected_forward_extensions = []
            for ext in best_forward_extensions[::-1]:

                # check if this extension moves any of the low counter up
                s_new = ext.s.copy()
                s_new[ext.n] = 1 - s_new[ext.n]
                if (1 - s_new)[c0 < bandwidth].any() or \
                        s_new[c1 < bandwidth].any():
                    c0 = c0 + 1 - s_new  # record the inactive features
                    c1 = c1 + s_new  # record the active features
                    selected_forward_extensions.append(ext)

            best_forward_extensions = selected_forward_extensions

        # Compute the target nodes of each selected extension.
        # This enables the next iteration to start.
        if algorithm == 'M-space':
            nodes_to_extend = [ext.get_target_node(A) for ext in
                               best_forward_extensions]
        elif algorithm == 'N1-space':
            nodes_to_extend = [ext.get_target_node(A, y) for ext in
                               best_forward_extensions]
        else:
            raise ValueError('argument `algorithm` must be either '
                             '"M-space" or "N1-space"')

    s_list, logL_list = zip(*results.items())
    return {'alpha': alpha, 's': s_list, 'logL': logL_list}


def sweep_alpha(alpha_grid, A, y, A_fix, s_start, N1_max, bandwidth, adaptive,
                a, b, algorithm,
                max_processes=None):
    """Runs band_search for each alpha value in `alpha_grid` in parallel.

    See arguments of :func:`band_search`

    Args:
        alpha_grid (sequence of :obj:`float`): alpha values for band_search
        max_processes (:obj:`int`, optional): max number of parallel processes
            Note: The number of processes is capped at 1000 to prevent
            overwhelming the system by accident.

    Returns:
        :obj:`list` of :obj:`dict`: alpha-sorted results of each band search

    """
    process_one_alpha = partial(band_search,
                                A=A, y=y, A_fix=A_fix, s_start=s_start,
                                N1_max=N1_max,
                                bandwidth=bandwidth, adaptive=adaptive,
                                a=a, b=b, algorithm=algorithm)

    if max_processes is None:
        max_processes = cpu_count()
    processes = min([max_processes, len(alpha_grid), 1000])

    with Pool(processes) as pool:
        results = pool.map(process_one_alpha, alpha_grid)

    results.sort(key=lambda res: res['alpha'])
    return results


def calculate_s_logprior(N, kappa, pi):
    """Computes log(P(s)) from section 2.1 (based on Beta prior for lambda)

    Note: Although the returned array is N+1 long (indexed by N1(s))
    its values are not (log-)normalized, because they correspond to log(P(s)),
    and not log(P(N1))

    Args:
        N (int): number of features
        kappa (float): pseudo count of the Beta prior
        pi (float): expectation value of the Beta prior

    Returns:
        numpy.ndarray: (N+1)-vector of log(P(s)) for each each N1(s) in 0..N

    """
    N1s = np.arange(0, N + 1, 1)
    log_ps = (betaln(kappa * pi + N1s, kappa * (1 - pi) + N - N1s)
              - betaln(kappa * pi, kappa * (1 - pi)))
    return log_ps


def estimate_coefficients(sweep_alpha_results, logQ, log_ps, A, y, A_fix,
                          test=False):
    M, N = A.shape
    M, N_fix = A_fix.shape
    alphas = [res['alpha'] for res in sweep_alpha_results]
    best_alpha = np.exp(np.sum(np.log(alphas) * np.exp(logQ)))
    best_res = sweep_alpha_results[np.argmax(logQ)]
    best_res_s = \
        np.frombuffer(b''.join(best_res['s']), dtype=np.uint8) \
        .reshape([len(best_res['s']), N])
    best_res_N1 = np.sum(best_res_s, axis=1).astype(int)
    best_res_logpL = log_ps[best_res_N1] + np.array(best_res['logL'])

    sort_order = np.argsort(-best_res_logpL)
    counters = np.zeros(N)
    goal_count = 10
    selected_s = []
    selected_logpL = []
    selected_x = []
    selected_x_fix = []
    for s, logpL in zip(best_res_s[sort_order],
                        best_res_logpL[sort_order]):
        if (s & (counters < goal_count)).any():
            counters += s
            s_bool = s.astype(bool)
            A_active = np.concatenate([A_fix, A[:, s_bool]],
                                      axis=1)
            selected_s.append(s)
            selected_logpL.append(logpL)
            x_full = _regularized_regression(A_active, y, best_alpha)
            selected_x_fix.append(x_full[:N_fix])
            x = np.zeros(N)
            x[s_bool] = x_full[N_fix:]
            selected_x.append(x)

        if (counters >= goal_count).all():
            break

    x = np.array(selected_x)
    x_fix = np.array(selected_x_fix)
    logpL = np.array(selected_logpL)
    logpL -= logsumexp(logpL)
    prob = np.exp(logpL)
    x_mmse = np.sum(x * np.outer(prob, np.ones(x.shape[1])),
                    axis=0)
    x_fix_mmse = np.sum(x_fix * np.outer(prob, np.ones(x_fix.shape[1])),
                        axis=0)
    if test:
        return np.array(selected_s), x, x_fix
    return x_mmse, x_fix_mmse


def calculate_posteriors(sweep_alpha_results, log_ps, A, y, A_fix,
                         estimate_coeffs, full):
    """Computes the posteriors according to Eq. (3), (4) and (5)

    Args:
        sweep_alpha_results (:obj:`list` of :obj:`dict`): returned by
            :func:`sweep_alpha`
        log_ps (numpy.ndarray): (N+1)-vector,
            returned by :func:`calculate_s_logprior`
        A (numpy.ndarray): the feature matrix [A_{m,n}]
        y (numpy.ndarray): target vector [y_m]
        A_fix (numpy.ndarray): fixed feature matrix [A_fix_{m,n_fix}]
        estimate_coeffs (bool): If True, compute estimates of x|y and x_fix|y
        full (bool): If True, P(s|y) is computed for all discovered s-vectors

    Returns:
        dict: dict containing:

            "alpha" (numpy.ndarray): |alphas|-vector of alpha values

            "logQ" (numpy.ndarray): |alphas|-vector of ln(P(alpha | y))

            "log_pn_0" (numpy.ndarray): N-vector of ln(P(s_n == 0 | y))

            "log_pN1" (numpy.ndarray): (N+1)-vector of ln(P(N1 | y))

            "x_mmse" (None or numpy.ndarray): N coefficients

            "x_fix_mmse" (None or numpy.ndarray): N_fix coefficients

            "full_log_psy_df" (None or pandas.DataFrame): data frame with cols

                 "s_{n}": binary integers for n=0..(N-1), the s-vectors

                 "log_psy": floats, ln(P(s | y))

    """
    alpha_size = len(sweep_alpha_results)
    N = len(log_ps) - 1

    alphas = np.zeros(alpha_size)
    logQ = np.zeros(alpha_size)
    log_marginal_0 = - np.inf * np.ones([alpha_size, N])
    log_marginal_1 = - np.inf * np.ones([alpha_size, N])
    log_marginal_N1 = - np.inf * np.ones([alpha_size, N + 1])
    for idx, res in enumerate(sweep_alpha_results):
        alphas[idx] = res['alpha']
        res_s = np.frombuffer(b''.join(res['s']),
                              dtype=np.uint8).reshape([len(res['s']), N])
        res_N1 = np.sum(res_s, axis=1).astype(int)
        res_logpL = log_ps[res_N1] + np.array(res['logL'])
        logQ[idx] = logsumexp(res_logpL)

        for n in range(N):
            log_marginal_0[idx, n] = logsumexp(
                res_logpL[res_s[:, n] == 0])
            log_marginal_1[idx, n] = logsumexp(
                res_logpL[res_s[:, n] == 1])

        for N1 in range(np.max(res_N1) + 1):
            log_marginal_N1[idx, N1] = logsumexp(res_logpL[res_N1 == N1])

    logQ -= logsumexp(logQ)

    logQ_Nextended = np.einsum('i,j->ij', logQ, np.ones(N))
    log_pn_0 = logsumexp(logQ_Nextended + log_marginal_0, axis=0)
    log_pn_1 = logsumexp(logQ_Nextended + log_marginal_1, axis=0)
    log_total = logsumexp([log_pn_0, log_pn_1], axis=0)
    log_pn_0 -= log_total
    log_pn_1 -= log_total

    logQ_Nplus1extended = np.einsum('i,j->ij', logQ, np.ones(N + 1))
    log_pN1 = logsumexp(logQ_Nplus1extended + log_marginal_N1, axis=0)
    log_pN1 -= logsumexp(log_pN1)

    if estimate_coeffs:
        x_mmse, x_fix_mmse = estimate_coefficients(sweep_alpha_results,
                                                   logQ, log_ps, A, y, A_fix)
    else:
        x_mmse = None
        x_fix_mmse = None

    if full:
        feature_cols = [f's_{n}' for n in range(N)]
        logQpL_cols = [f'logQpL_{i}' for i in range(alpha_size)]
        dfs = [pd.DataFrame() for _ in alphas]
        for logQpL_col, df, res, logQ_alpha in zip(logQpL_cols, dfs,
                                                   sweep_alpha_results, logQ):
            res_s = np.frombuffer(b''.join(res['s']),
                                  dtype=np.uint8).reshape([len(res['s']), N])
            res_N1 = np.sum(res_s, axis=1).astype(int)
            res_logpL = log_ps[res_N1] + np.array(res['logL'])
            for n, feature_col in enumerate(feature_cols):
                df[feature_col] = res_s[:, n]
            df[logQpL_col] = logQ_alpha + res_logpL

        df_all = dfs[0]
        for df in dfs[1:]:
            df_all = pd.merge(df_all, df, on=feature_cols, how='outer')
        df_all.fillna(-np.inf, inplace=True)
        df_all['log_psy'] = logsumexp(df_all[logQpL_cols].values, axis=1)
        df_all['log_psy'] -= logsumexp(df_all['log_psy'].values)
        full_log_psy_df = df_all[feature_cols + ['log_psy']]
    else:
        full_log_psy_df = None

    return {'alpha': alphas,
            'logQ': logQ,
            'log_pn_0': log_pn_0,
            'log_pN1': log_pN1,
            'x_mmse': x_mmse,
            'x_fix_mmse': x_fix_mmse,
            'full_log_psy_df': full_log_psy_df}


def unnormalize_posteriors(posteriors, normalization_results, kappa, pi):
    """Return the posteriors to the same shape and scale as the raw inputs.

    Uniform features are assumed to be inactive with 100% probability.

    Args:
        posteriors (dict): results of :func:`calculate_posteriors`
        normalization_results (dict): results of :func:`normalize_input`
        kappa (float): pseudo count of the Beta prior of lambda
        pi (float): expectation value of the Beta prior of lambda

    Returns:
        dict: dict with the same structure as `posteriors`
    """
    uniform_features = normalization_results['uniform_features']
    N_full = len(uniform_features)
    N = np.sum(~uniform_features)

    fix_uniform_features = normalization_results['fix_uniform_features']
    N_fix_full = len(fix_uniform_features)
    N_fix = np.sum(~fix_uniform_features)

    target_scale = normalization_results['y_scale']
    feature_scales = normalization_results['A_scale']
    fix_feature_scales = normalization_results['A_fix_scale']

    # check if inputs are consistent
    if len(posteriors['log_pn_0']) != N:
        raise ValueError("len(log_pn_0) must be == sum(~uniform_features)")
    if len(posteriors['log_pN1']) != N + 1:
        raise ValueError("len(log_pN1) must be == sum(~uniform_features) + 1")
    if posteriors['full_log_psy_df'] is not None:
        if len(posteriors['full_log_psy_df'].columns) - 1 != N:
            raise ValueError("Number of s columns in full_log_psy_df must  be "
                             "== sum(~uniform_features)")

    new_posteriors = {}

    # alpha and logQ need no change
    new_posteriors['alpha'] = posteriors['alpha']
    new_posteriors['logQ'] = posteriors['logQ']

    if N < N_full:
        # We set P(s_n=0 | y) = 1 - EV(lambda) for every uniform feature n
        new_log_pn_0 = np.zeros(N_full)
        new_log_pn_0[~uniform_features] = posteriors['log_pn_0']
        new_log_pn_0[uniform_features] = np.log(1.0 - pi)
        new_posteriors['log_pn_0'] = new_log_pn_0

        # N1 from the uniform features is
        # N1_uniform ~ BetaBinomial(N_uniform, kappa*pi, kappa*(1-pi))
        # and N1 = N1_uniform + N1_nonuniform
        # P(N1 | y) need to be evaluated
        N1_unif = np.arange(0, N_full - N + 1, 1)
        N1_nonunif = np.arange(0, N + 1, 1)
        log_pN1_unif = _betabinomial_logpmf(N1_unif, N_full - N + 1,
                                            kappa * pi, kappa * (1.0 - pi))
        log_pN1_nonunif = posteriors['log_pN1']
        N1_matrix = N1_unif[:, None] + N1_nonunif[None, :]
        log_pN1_matrix = log_pN1_unif[:, None] + log_pN1_nonunif[None, :]
        new_log_pN1 = - np.inf * np.ones(N_full + 1)
        for N1 in range(N_full):
            new_log_pN1[N1] = logsumexp(log_pN1_matrix[N1_matrix == N1])
        new_posteriors['log_pN1'] = new_log_pN1

        # Coefficients x_mmse_if_nonzero of uniform features are set to NaN
        if posteriors['x_mmse'] is not None:
            new_x_mmse = np.zeros(N_full)
            new_x_mmse[uniform_features] = np.nan
            new_x_mmse[~uniform_features] = posteriors['x_mmse']
        else:
            new_x_mmse = None
        new_posteriors['x_mmse'] = new_x_mmse

        # We set all s_n = nan for every uniform feature n
        if posteriors['full_log_psy_df'] is not None:
            df = posteriors['full_log_psy_df'].copy()
            rows = len(df)
            nans = np.repeat([np.nan], rows)
            if df is not None:
                for n_full, is_uniform in enumerate(uniform_features):
                    if is_uniform:
                        df.insert(n_full, f'unif_{n_full}', nans)
                df.columns = [f's_{n_full}' for n_full in range(N_full)] + \
                             ['log_psy']
            new_posteriors['full_log_psy_df'] = df
        else:
            new_posteriors['full_log_psy_df'] = None
    else:
        new_posteriors['log_pn_0'] = posteriors['log_pn_0']
        new_posteriors['log_pN1'] = posteriors['log_pN1']
        new_posteriors['x_mmse'] = posteriors['x_mmse']
        new_posteriors['full_log_psy_df'] = posteriors['full_log_psy_df']

    if N_fix < N_fix_full:

        # Coefficients x_fix_mmse uniform fixed features are set to NaN
        if posteriors['x_fix_mmse'] is not None:
            new_x_fix_mmse = np.zeros(N_fix_full)
            new_x_fix_mmse[fix_uniform_features] = np.nan
            new_x_fix_mmse[~fix_uniform_features] = posteriors['x_fix_mmse']
        else:
            new_x_fix_mmse = None
        new_posteriors['x_fix_mmse'] = new_x_fix_mmse
    else:
        new_posteriors['x_fix_mmse'] = posteriors['x_fix_mmse']

    # x_mmse and x_fix_mmse are unnormalized with the scales of the original
    # features and original targets
    if new_posteriors['x_mmse'] is not None:
        new_posteriors['x_mmse'][~uniform_features] *= \
            (target_scale / feature_scales[~uniform_features])
    if new_posteriors['x_fix_mmse'] is not None:
        new_posteriors['x_fix_mmse'][~fix_uniform_features] *= \
            (target_scale / fix_feature_scales[~fix_uniform_features])

    return new_posteriors
