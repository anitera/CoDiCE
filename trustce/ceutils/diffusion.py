import numpy as np
from sklearn.neighbors import NearestNeighbors
import scipy.sparse as sps
import scipy.sparse.linalg as spsl
try:
    from scipy.special import logsumexp
except ModuleNotFoundError:
    from scipy.misc import logsumexp

import numpy as np
from sklearn.neighbors import NearestNeighbors
import scipy.sparse as sps
import scipy.sparse.linalg as spsl
try:
    from scipy.special import logsumexp
except ModuleNotFoundError:
    from scipy.misc import logsumexp

class STDiffusionMap(object):
    """Diffusion Map implementation with self-tuning type kernel"""
    def __init__(self, n_neighbors, alpha=1.0):
        self.n_neighbors = n_neighbors
        self.alpha = alpha
        self.eigenvalues = None
        self.eigenvectors = None
        self.dmap = None
        self.oos = "nystroem"

    def fit(self,X):
        self.data = X
        K = self.construct_affinity_matrix(X)
        self.construct_transition_matrix(K)
        dmap, evecs, evals = self._make_diffusion_coords(self.L)
        self.eigenvalues = evals
        self.eigenvectors = evecs
        self.dmap = dmap
        return self

    def _make_diffusion_coords(self, L):
        evals, evecs = spsl.eigs(L, k=((len(self.data) - 1)//2), which='LR')
        ix = evals.argsort()[::-1][1:]
        evals = np.real(evals[ix])
        evecs = np.real(evecs[:, ix])
        dmap = np.dot(evecs, np.diag(np.sqrt(-1. / evals)))
        #dmap = np.dot(evecs, np.diag(np.sqrt(evals)))
        return dmap, evecs, evals

    def construct_transition_matrix(self, kernel_matrix):
        q, right_norm_vec = self._make_right_norm_vec(kernel_matrix)
        P = self._right_normalize(kernel_matrix, right_norm_vec)
        P = self._left_normalize(P)
        m, n = P.shape
        L = (P - sps.eye(m, n, k=(n - m))) / 1
        #self.local_kernel = my_kernel
        self.kernel_matrix = kernel_matrix
        self.L = L
        self.q = q
        self.right_norm_vec = right_norm_vec
        return L

    def _make_right_norm_vec(self, kernel_matrix):
        q = np.array(kernel_matrix.sum(axis=1)).ravel()
        right_norm_vec = np.power(q, -self.alpha)
        return q, right_norm_vec

    def _right_normalize(self, kernel_matrix, right_norm_vec):
        m = right_norm_vec.shape[0]
        Dalpha = sps.spdiags(right_norm_vec, 0, m, m)
        kernel_matrix = kernel_matrix * Dalpha
        return kernel_matrix

    def _left_normalize(self, kernel_matrix):
        row_sum = kernel_matrix.sum(axis=1).transpose()
        n = row_sum.shape[1]
        Dalpha = sps.spdiags(np.power(row_sum, -1), 0, n, n)
        P = Dalpha * kernel_matrix
        return P

    def construct_affinity_matrix(self, X):
        # Compute the distance matrix
        self.neigh = NearestNeighbors(n_neighbors=self.n_neighbors+1,
                                          metric='euclidean')
        self.neigh.fit(X)
        scaled_dists = self.neigh.kneighbors_graph(X, mode='distance')
        #self.epsilon_fitted, d = self.choose_optimal_eps_bgh(scaled_dists.data**2, epsilons=None)
        self.local_scale = self.calculate_local_scale(scaled_dists)
        # Make kernel symmetric
        local_scale_matrix = self.local_scale*self.local_scale.T
        #scaled_dists.data = self.gaussian_kfxn(scaled_dists.data, self.epsilon_fitted)
        local_scale_data = self.local_scale_to_data(scaled_dists.data, self.local_scale)
        scaled_dists.data = self.selftuning_kernel(scaled_dists.data, local_scale_data)
        Ktrans = scaled_dists.transpose()
        dK = abs(scaled_dists - Ktrans)
        scaled_dists = scaled_dists + Ktrans
        scaled_dists = 0.5*(scaled_dists + dK)
        return scaled_dists

    def gaussian_kfxn(self, d, epsilon):
        return np.exp(-d**2 / (4. * epsilon))

    def selftuning_kernel(self, d, local_scale_data):
        return np.exp(-d**2 / (local_scale_data**2))

    def calculate_local_scale(self, scaled_dists):
        local_scale = np.zeros(self.data.shape[0])
        for i in range(len(local_scale)):
          local_scale[i] = scaled_dists[i,:].data[-1]
        self.local_scale = local_scale[:, np.newaxis]  # Reshape for consistency
        return self.local_scale

    def local_scale_to_data(self, data, local_scale):
        length_data = data.shape[0]
        repetition_factor = length_data/len(local_scale)
        extended_length = len(local_scale) * repetition_factor  # Each element repeated 7 times
        # Extend the array
        local_scale_to_data = np.repeat(local_scale, repetition_factor)
        return local_scale_to_data


    def choose_optimal_eps_bgh(self, dists, epsilons=None):
        """
        Calculates the optimal epsilon for kernel density estimation according to
        the criteria in Berry, Giannakis, and Harlim.

        Parameters
        ----------
        scaled_distsq : numpy array
            Values for scaled distance squared values, in no particular order or shape. (This is the exponent in the Gaussian Kernel, aka the thing that gets divided by epsilon).
        epsilons : array-like, optional
            Values of epsilon from which to choose the optimum.  If not provided, uses all powers of 2. from 2^-40 to 2^40

        Returns
        -------
        epsilon : float
            Estimated value of the optimal length-scale parameter.
        d : int
            Estimated dimensionality of the system.

        Notes
        -----
        This code explicitly assumes the kernel is gaussian, for now.

        References
        ----------
        The algorithm given is based on [1]_.  If you use this code, please cite them.

        .. [1] T. Berry, D. Giannakis, and J. Harlim, Physical Review E 91, 032915
        (2015).
        """
        if epsilons is None:
            epsilons = 2**np.arange(-40., 41., 1.)

        epsilons = np.sort(epsilons).astype('float')
        log_T = [logsumexp(-dists/(4. * eps)) for eps in epsilons]
        log_eps = np.log(epsilons)
        log_deriv = np.diff(log_T)/np.diff(log_eps)
        max_loc = np.argmax(log_deriv)
        # epsilon = np.max([np.exp(log_eps[max_loc]), np.exp(log_eps[max_loc+1])])
        epsilon = np.exp(log_eps[max_loc])
        d = np.round(2.*log_deriv[max_loc])
        return epsilon, d

    def transform(self, Y):
        # Compute the kernel matrix
        if (Y.ndim == 1):
            Y = Y[np.newaxis, :]
        if self.oos == "nystroem":
            #return self.project_into_diffusion_space_selft(Y)
            return nystroem_oos(self, Y)
        elif self.oos == "power":
            return power_oos(self, Y)
        K = self.neigh.kneighbors_graph(Y, mode='distance')
        K.data = self.gaussian_kfxn(K.data, self.epsilon_fitted)
        dmap = np.dot(K.toarray(), self.eigenvectors)
        return dmap

    def distance_between_two_points(self, x, y):
        """Calculate diffusion distance between two points from original space"""
        # Transform the points to diffusion coordinates
        x = self.transform(x)
        y = self.transform(y)
        # Calculate the distance
        dist = np.linalg.norm(x - y)
        return dist


def nystroem_oos(dmap_object, Y):
    """
    Performs Nystroem out-of-sample extension to calculate the values of the diffusion coordinates at each given point.

    Parameters
    ----------
    dmap_object : DiffusionMap object
        Diffusion map upon which to perform the out-of-sample extension.
    Y : array-like, shape (n_query, n_features)
        Data for which to perform the out-of-sample extension.

    Returns
    -------
    phi : numpy array, shape (n_query, n_eigenvectors)
        Transformed value of the given values.
    """
    # check if Y is equal to data. If yes, no computation needed.
    # compute the values of the kernel matrix
    kernel_extended = dmap_object.neigh.kneighbors_graph(Y, mode='distance')
    local_scale_Y = kernel_extended[0,:].data[-1]
    kernel_extended.data = dmap_object.selftuning_kernel(kernel_extended.data, dmap_object.local_scale*local_scale_Y)

    #kernel_extended_scaled = kernel_extended/local_scale_Y
    P = dmap_object._left_normalize(dmap_object._right_normalize(kernel_extended, dmap_object.right_norm_vec))
    oos_evecs = P * dmap_object.dmap
    # evals_p = dmap_object.local_kernel.epsilon_fitted * dmap_object.evals + 1.
    # oos_dmap = np.dot(oos_evecs, np.diag(1. / evals_p))
    return oos_evecs



def power_oos(dmap_object, Y):
    """
    Performs out-of-sample extension to calculate the values of the diffusion coordinates at each given point using the power-like method.

    Parameters
    ----------
    dmap_object : DiffusionMap object
        Diffusion map upon which to perform the out-of-sample extension.
    Y : array-like, shape (n_query, n_features)
        Data for which to perform the out-of-sample extension.

    Returns
    -------
    phi : numpy array, shape (n_query, n_eigenvectors)
        Transformed value of the given values.
    """
    m = int(Y.shape[0])
    k_yx, y_bandwidths = dmap_object.local_kernel.compute(Y, return_bandwidths=True)  # Evaluate on ref points
    yy_right_norm_vec = dmap_object._make_right_norm_vec(k_yx, y_bandwidths)[1]
    k_yy_diag = dmap_object.local_kernel.kernel_fxn(0, dmap_object.epsilon_fitted)
    data_full = np.vstack([dmap_object.local_kernel.data, Y])
    k_full = sps.hstack([k_yx, sps.eye(m) * k_yy_diag])
    right_norm_full = np.hstack([dmap_object.right_norm_vec, yy_right_norm_vec])
    weights = dmap_object._compute_weights(data_full)

    P = dmap_object._left_normalize(dmap_object._right_normalize(k_full, right_norm_full, weights))
    L = dmap_object._build_generator(P, dmap_object.epsilon_fitted, y_bandwidths)
    L_yx = L[:, :-m]
    L_yy = np.array(L[:, -m:].diagonal())
    adj_evals = dmap_object.evals - L_yy.reshape(-1, 1)
    dot_part = np.array(L_yx.dot(dmap_object.dmap))
    return (1. / adj_evals) * dot_part

        