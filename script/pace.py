import h5py
import numpy as np
from abc import ABC, abstractmethod
from scipy.linalg import cho_solve, cho_factor
from script.utils import inv
import script.dataset as ds

"""
TODO:
1. select bandwidth?

"""


class LocalLinear(ABC):
    """
    Abstract class for local linear estimator.
    
    """
    def __init__(self, ldrs, time, n_obs,):
        """
        ldrs (n_obs, n_ldrs): a np.array of ldrs in long format 
        time (n_obs,): a np.array of time points
        n_obs (n_sub,): a np.array of numbers of obs for each subject
        
        """
        self.ldrs = ldrs
        self.time = time
        self.n_obs = n_obs
        self.time = self.time / np.max(self.time)
        self.unique_time = np.unique(self.time)
        self.n_obs_adj = np.repeat(1 / self.n_obs, self.n_obs)
        self.n_time = len(self.unique_time)
        self.n_ldrs = self.ldrs.shape[1]

    def _gau_kernel(self, x):
        """
        Calculating the Gaussian density

        """
        gau_k = 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * x**2)
        if len(gau_k.shape) == 2:
            gau_k = np.prod(gau_k, axis=1).reshape(-1, 1)
        return gau_k.astype(np.float32)
    
    @staticmethod
    def _wls(x, y, weights):
        """
        Weighted least squares
        
        """
        if len(weights.shape) == 1:
            weights = weights.reshape(-1, 1)
        xw = x * weights
        xtx = np.dot(xw.T, x)
        xty = np.dot(xw.T, y)
        c, lower = cho_factor(xtx)
        beta = cho_solve((c, lower), xty).astype(np.float32)
        return beta[0]
    
    @abstractmethod
    def _get_design_matrix(self):
        """
        Get design matrix for regression
        
        """
        pass

    @abstractmethod
    def estimate(self):
        pass
    

class Mean(LocalLinear):
    def _get_design_matrix(self, t, bw):
        time_diff = self.time - t
        weights = self._gau_kernel(time_diff / bw) / bw * self.n_obs_adj
        time_diff = time_diff.reshape(-1, 1)
        x = np.hstack([np.ones_like(time_diff), time_diff])
        return x, weights
    
    def estimate(self, bw):
        mean_function = np.zeros((self.n_time, self.n_ldrs), dtype=np.float32)
        for i, t in enumerate(self.unique_time):
            x, weights = self._get_design_matrix(t, bw)
            mean_function[i] = self._wls(x, self.ldrs, weights)
        mean_function = mean_function.T
        return mean_function    
    

class Covariance(LocalLinear):
    def __init__(self, ldrs, time, sub_obs):
        super().__init__(ldrs, time, sub_obs)
        self.n_obs_adj = np.repeat(
            1 / (self.n_obs * (self.n_obs - 1)), self.n_obs * (self.n_obs - 1)
        ).reshape(-1, 1)
        
        self.two_way_ldrs = np.zeros(
            (np.sum(self.n_obs ** 2 - self.n_obs), self.n_ldrs), 
            dtype=np.float32
        )
        self.two_way_time = np.zeros(
            (np.sum(self.n_obs ** 2 - self.n_obs), 2), 
            dtype=np.float32
        )

        start1, end1 = 0, 0
        start2, end2 = 0, 0
        for n_obs in self.n_obs:
            end1 += n_obs
            end2 += n_obs ** 2 - n_obs
            off_diag = ~np.eye(n_obs, dtype=bool)

            # time
            time_stack_by_col = np.tile(self.time[start1: end1].reshape(-1, 1), n_obs)
            time_stack_by_row = time_stack_by_col.T
            self.two_way_time[start2: end2, 0] = time_stack_by_row[off_diag]
            self.two_way_time[start2: end2, 1] = time_stack_by_col[off_diag]

            for i in range(self.n_ldrs):
                # ldr
                sub_ldr = self.ldrs[start1: end1, i].reshape(-1, 1)
                outer_prod_sub_ldr = np.dot(sub_ldr, sub_ldr.T)
                self.two_way_ldrs[start2: end2, i] = outer_prod_sub_ldr[off_diag]

            start1 = end1
            start2 = end2

    def _get_design_matrix(self, t1, t2, bw):
        time_diff = self.two_way_time - np.array([t1, t2])
        weights = self._gau_kernel(time_diff / bw) / bw * self.n_obs_adj
        x = np.hstack([np.ones(time_diff.shape[0], dtype=np.float32).reshape(-1, 1), time_diff])
        return x, weights
    
    def estimate(self, mean, bw):
        cov_function = np.zeros((self.n_ldrs, self.n_time, self.n_time), dtype=np.float32)
        for t1 in range(self.n_time):
            for t2 in range(t1, self.n_time):
                x, weights = self._get_design_matrix(self.unique_time[t1], self.unique_time[t2], bw)
                cov_function[:, t1, t2] = (
                    self._wls(x, self.two_way_ldrs, weights) - mean[:, t1] * mean[:, t2]
                )
        
        iu_rows, iu_cols = np.triu_indices(self.n_time, k=1)
        for i in range(self.n_ldrs):
            cov_function[i][(iu_cols, iu_rows)] = cov_function[i][(iu_rows, iu_cols)]
            # cov_function[i] = cov_function[i] - np.dot(mean[i].reshape(-1, 1).T, mean[i])

        return cov_function
    

class ResidualVariance(LocalLinear):
    def _get_design_matrix(self, t, bw):
        time_diff = self.time - t
        weights = self._gau_kernel(time_diff / bw) / bw * self.n_obs_adj
        time_diff = time_diff.reshape(-1, 1)
        x = np.hstack([np.ones_like(time_diff), time_diff])
        return x, weights
    
    def estimate(self, mean, cov, bw):
        resid_var = np.zeros((self.n_time, self.n_ldrs), dtype=np.float32)
        for i, t in enumerate(self.unique_time):
            x, weights = self._get_design_matrix(t, bw)
            resid_var[i] = self._wls(x, self.ldrs**2, weights) - mean[:, i]**2
        resid_var = resid_var.T

        for i in range(self.n_ldrs):
            resid_var[i] = resid_var[i] - np.diag(cov[i])
            # TODO: negative var?

        return resid_var
    

def pace(ldrs, sub_time, unique_time_map, mean, cov, resid_var):
    """
    PACE estimator for time LDRs

    Parameters:
    ------------
    ldrs (n_obs, n_ldrs): a np.array of ldrs in long format 
    sub_time: a dictionary of sub:time, where sub should start with 0
    unique_time_map: a dictionary of mapping time to index
    mean (n_ldrs, n_time): a np.array of mean estimate
    cov (n_ldrs, n_time, n_time): a np.array of cov estimate
    resid_var (n_ldrs, n_time): a np.array of resid var estimate

    Returns:
    ---------
    recon_spatial_ldrs (n_time, n_sub, n_ldrs): reconstructed spatial LDRs at each time point  
    
    """
    n_sub = len(sub_time)
    n_ldrs, n_time = mean.shape
    time_spatial_ldrs = np.zeros((n_ldrs, n_sub, n_time), dtype=np.float32)
    recon_spatial_ldrs = np.zeros((n_ldrs, n_sub, n_time), dtype=np.float32)
    eg_values = np.zeros((n_ldrs, n_time), dtype=np.float32)
    eg_vectors = np.zeros((n_ldrs, n_time, n_time), dtype=np.float32)

    for i in range(n_ldrs):
        eg_values_, eg_vectors_ = np.linalg.eigh(cov[i])
        eg_values_ = np.flip(eg_values_) # (n_time, )
        eg_vectors_ = np.flip(eg_vectors_, axis=1) # (n_time, n_time)
        eg_values[i] = eg_values_
        eg_vectors[i] = eg_vectors_
        eg_vectors_ = eg_vectors_ * eg_values_
        
        # TODO: map time to index
        start, end = 0, 0
        for sub_idx, (_, time) in enumerate(sub_time.items()):
            end += len(time)
            time_idx = np.array([unique_time_map.get(t, 0) for t in time])
            y_i = ldrs[start: end, i] # (n_time_i, )
            # mu_i = mean[i, time_idx] # (n_time_i, )
            Sigma_i_inv = np.linalg.inv(cov[i, time_idx, time_idx] + np.diag(resid_var[i, time_idx])) # (n_time_i, n_time_i)
            eg_vector = eg_vectors_[time_idx] # (n_time_i, n_time)
            time_spatial_ldrs[i, sub_idx] = np.dot(np.dot(eg_vector.T, Sigma_i_inv), y_i)
            start = end
        
        # do reconstruction for time
        recon_spatial_ldrs[i] = np.dot(time_spatial_ldrs[i], eg_vectors[i].T)
        recon_spatial_ldrs = recon_spatial_ldrs.transpose(2, 1, 0)

    return recon_spatial_ldrs
    

def ldr_cov(recon_spatial_ldrs, covar):
    """
    Computing S'(I - M)S/n = S'S - S'X(X'X)^{-1}X'S/n,
    where I is the identity matrix,
    M = X(X'X)^{-1}X' is the project matrix for X,
    S is the LDR matrix.

    Parameters:
    ------------
    recon_spatial_ldrs (n_time, n_sub, n_ldrs): reconstructed spatial LDRs at each time point
    covar (n_sub, n_covar): covariates, including the intercept

    Returns:
    ---------
    ldr_cov_matrix (n_time, n_ldrs, n_ldrs): variance-covariance matrix of LDRs at each time point

    """
    n_time, n_sub, n_ldrs = recon_spatial_ldrs.shape
    ldr_cov_matrix = np.zeros((n_time, n_ldrs, n_ldrs), dtype=np.float32)
    inner_covar = np.dot(covar.T, covar)
    inner_covar_inv = inv(inner_covar)

    for t in range(n_time):
        ldrs = recon_spatial_ldrs[t]
        inner_ldr = np.dot(ldrs.T, ldrs)
        ldr_covar = np.dot(ldrs.T, covar)
        part2 = np.dot(np.dot(ldr_covar, inner_covar_inv), ldr_covar.T)
        ldr_cov = (inner_ldr - part2) / n_sub
        ldr_cov_matrix[t] = ldr_cov

    return ldr_cov_matrix
    

def check_input(args):
    # required arguments
    if args.spatial_ldrs is None:
        raise ValueError("--ldrs is required")
    if args.covar is None:
        raise ValueError("--covar is required")


def run(args, log):
    check_input(args)

    # read ldrs
    log.info(f"Read LDRs from {args.spatial_ldrs}")
    ldrs = ds.Dataset(args.spatial_ldrs)
    log.info(f"{ldrs.data.shape[1]-1} LDRs and {ldrs.data.shape[0]} observations.")
    if args.n_ldrs is not None:
        ldrs.data = ldrs.data.iloc[:, :args.n_ldrs+1]
        if ldrs.data.shape[1]-1 > args.n_ldrs:
            log.info(f"WARNING: --n-ldrs greater than #LDRs, using all LDRs.")
        else:
            log.info(f"Keeping the top {args.n_ldrs} LDRs.")

    # remove subjects with a single obs
    n_obs = ldrs.data.index.value_counts(sort=False).values
    obs_to_exclude = np.cumsum(n_obs)[n_obs == 1] - 1
    ids_to_exclude = ldrs.data.index[obs_to_exclude]
    log.info(f"Removed {len(obs_to_exclude)} subjects with only one observation.")

    # read covariates
    log.info(f"Read covariates from {args.covar}")
    covar = ds.Covar(args.covar, args.cat_covar_list)

    # keep common subjects
    common_idxs = ds.get_common_idxs(ldrs.data.index, covar.data.index, args.keep)
    if args.remove is not None:
        ids_to_exclude = ids_to_exclude.union(args.remove)
    common_idxs = ds.remove_idxs(common_idxs, ids_to_exclude)
    log.info(f"{len(common_idxs)} subjects common in these files.")
    ldrs.keep_and_remove(common_idxs)
    covar.keep_and_remove(common_idxs)
    covar.cat_covar_intercept()

    # estimation
    log.info("Reconstruct time by PACE ...")
    ldrs_data = np.array(ldrs.data.iloc[:, 1:])
    time = ldrs.data['time'].values
    ids = ldrs.data.index
    ldrs.to_single_index()
    n_obs = ldrs.data.index.value_counts(sort=False).values

    mean_estimator = Mean(ldrs_data, time, n_obs)
    mean = mean_estimator.estimate(0.1)
    
    cov_estimator = Covariance(ldrs_data, time, n_obs)
    cov = cov_estimator.estimate(mean, 0.1)

    resid_var_estimator = ResidualVariance(ldrs_data, time, n_obs)
    resid_var = resid_var_estimator.estimate(mean, cov, 0.1)

    # PACE
    sub_time = ldrs.data.groupby("IID")["time"].apply(list).to_dict()
    unique_time_map = {t: i for i, t in enumerate(np.unique(ldrs.data["time"]))}
    recon_spatial_ldrs = pace(ldrs_data, sub_time, unique_time_map, mean, cov, resid_var)

    # cov matrix of LDRs for each time
    ldr_cov_matrix = ldr_cov(recon_spatial_ldrs, np.array(covar.data))

    # save
    with h5py.File(f"{args.out}_recon_ldr.h5", 'w') as file:
        file.create_dataset("ldrs", data=recon_spatial_ldrs, dtype="float32")
        file.create_dataset("id", data=np.array(ids.tolist(), dtype="S10"))
        file.create_dataset("time", data=np.unique(time))

    np.save(f"{args.out}_ldr_cov.npy", ldr_cov_matrix)

    log.info(f"\nSaved spatial temporal LDRs to {args.out}_recon_ldr.h5")
    log.info(
        (
            f"Saved the variance-covariance matrix of covariate-effect-removed LDRs "
            f"to {args.out}_ldr_cov.npy"
        )
    )