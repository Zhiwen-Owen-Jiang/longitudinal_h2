import h5py
import logging
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from scipy.linalg import cho_solve, cho_factor
from script.utils import inv
import script.dataset as ds

"""
TODO:
1. select bandwidth?

"""


def traz(f, g):
    if len(f) != len(g):
        raise ValueError('f and g have different lengths')
    if not all(f[i] < f[i+1] for i in range(len(f)-1)):
        raise ValueError('f must be strictly increasing')
    res = 0
    for i in range(len(f) - 1):
        res += 0.5 * (f[i+1] - f[i]) * (g[i] + g[i+1])
    return res


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
        grid_size = (self.unique_time[-1] - self.unique_time[0]) / 50
        self.time_grid = np.arange(self.unique_time[0], self.unique_time[-1] + grid_size, grid_size)

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
        # weights = self._gau_kernel(time_diff / bw) / bw * self.n_obs_adj
        weights = self._gau_kernel(time_diff / bw)
        time_diff = time_diff.reshape(-1, 1)
        x = np.hstack([np.ones_like(time_diff), time_diff])
        return x, weights
    
    def estimate(self, bw):
        mean_function = np.zeros((self.n_time, self.n_ldrs), dtype=np.float32)
        grid_mean_function = np.zeros((51, self.n_ldrs), dtype=np.float32)
        
        for i, t in enumerate(self.unique_time):
            x, weights = self._get_design_matrix(t, bw)
            mean_function[i] = self._wls(x, self.ldrs, weights)
        mean_function = mean_function.T

        for i, t in enumerate(self.time_grid):
            x, weights = self._get_design_matrix(t, bw)
            grid_mean_function[i] = self._wls(x, self.ldrs, weights)
        grid_mean_function = grid_mean_function.T

        return mean_function, grid_mean_function   
    

class Covariance(LocalLinear):
    def __init__(self, ldrs, time, sub_obs, mean, time_idx):
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
                sub_ldr = ldrs[start1: end1, i]
                sub_ldr = sub_ldr - mean[i, time_idx[start1: end1]]
                sub_ldr = sub_ldr.reshape(-1, 1)
                outer_prod_sub_ldr = np.dot(sub_ldr, sub_ldr.T)
                self.two_way_ldrs[start2: end2, i] = outer_prod_sub_ldr[off_diag]

            start1 = end1
            start2 = end2

        self.unique_time_comb = np.zeros((self.n_time ** 2 - self.n_time, 2), dtype=np.float32)
        self.mean_ldrs = np.zeros((self.n_time ** 2 - self.n_time, self.n_ldrs), dtype=np.float32)
        self.time_comb_count = np.zeros((self.n_time ** 2 - self.n_time, 1), dtype=np.int32)
        k = 0
        for i in range(self.n_time):
            for j in range(self.n_time):
                if i != j:
                    t1 = self.unique_time[i]
                    t2 = self.unique_time[j]
                    self.unique_time_comb[k] = np.array([t1, t2])
                    time_comb_idx = (self.two_way_time == self.unique_time_comb[k]).all(axis=1)
                    self.time_comb_count[k] = np.sum(time_comb_idx)
                    self.mean_ldrs[k] = np.mean(self.two_way_ldrs[time_comb_idx], axis=0)
                    k += 1

    def _get_design_matrix(self, t1, t2, bw):
        """
        TODO: set large distances as 0.
        
        """
        time_diff = self.unique_time_comb - np.array([t1, t2])
        # weights = self._gau_kernel(time_diff / bw) / bw * self.n_obs_adj
        weights = self._gau_kernel(time_diff / bw) * self.time_comb_count
        x = np.hstack([np.ones(time_diff.shape[0], dtype=np.float32).reshape(-1, 1), time_diff])
        return x, weights
    
    def _get_design_matrix2(self, time, t, bw):
        """
        local quadratic
        
        """
        time_diff = time - t
        weights = self._gau_kernel(time_diff / bw) * self.time_comb_count
        time_diff[:, 1] = time_diff[:, 1] ** 2
        x = np.hstack([np.ones(time_diff.shape[0], dtype=np.float32).reshape(-1, 1), time_diff])
        return x, weights
    
    def estimate(self, bw):
        grid_cov_function = np.zeros((self.n_ldrs, 51, 51), dtype=np.float32)

        for t1 in range(51):
            for t2 in range(t1, 51):
                x, weights = self._get_design_matrix(self.time_grid[t1], self.time_grid[t2], bw)
                grid_cov_function[:, t1, t2] = self._wls(x, self.mean_ldrs, weights)
        
        iu_rows, iu_cols = np.triu_indices(51, k=1)
        for i in range(self.n_ldrs):
            grid_cov_function[i][(iu_cols, iu_rows)] = grid_cov_function[i][(iu_rows, iu_cols)]

        cut_time_grid = self.time_grid[
            (self.time_grid > np.quantile(self.time_grid, 0.25)) & 
            (self.time_grid < np.quantile(self.time_grid, 0.75))
        ]
        cut_time_grid = np.tile(cut_time_grid.reshape(-1, 1), 2)
        n_cut_time_grid = cut_time_grid.shape[0]
        rotation_matrix = np.array([[1, 1], [-1, 1]]).T * (np.sqrt(2) / 2)
        rotated_time_comb = np.dot(self.unique_time_comb, rotation_matrix)
        rotated_cut_time_grid = np.dot(cut_time_grid, rotation_matrix)
        cut_time_grid_diag = np.zeros((self.n_ldrs, n_cut_time_grid), dtype=np.float32)
        for t in range(n_cut_time_grid):
            x, weights = self._get_design_matrix2(
                rotated_time_comb, 
                rotated_cut_time_grid[t],
                0.1
            )
            cut_time_grid_diag[:, t] = self._wls(x, self.mean_ldrs, weights)

        return grid_cov_function, cut_time_grid_diag
    

class ResidualVariance(LocalLinear):
    def _get_design_matrix(self, t, bw):
        time_diff = self.time - t
        # weights = self._gau_kernel(time_diff / bw) / bw * self.n_obs_adj
        weights = self._gau_kernel(time_diff / bw)
        time_diff = time_diff.reshape(-1, 1)
        x = np.hstack([np.ones_like(time_diff), time_diff])
        return x, weights
    
    def estimate(self, mean, diag, time_idx, bw):
        one_way_mean = mean[0, time_idx].reshape(-1, 1)
        resid_var = np.zeros(self.n_ldrs, dtype=np.float32)
        grid_resid_var = np.zeros((51, self.n_ldrs), dtype=np.float32)

        for i, t in enumerate(self.time_grid):
            x, weights = self._get_design_matrix(t, bw)
            grid_resid_var[i] = self._wls(x, (self.ldrs - one_way_mean)**2, weights)
        grid_resid_var = grid_resid_var.T

        cut_time_grid = self.time_grid[
            (self.time_grid > np.quantile(self.time_grid, 0.25)) & 
            (self.time_grid < np.quantile(self.time_grid, 0.75))
        ]
        grid_resid_var = grid_resid_var[:, 
            (self.time_grid > np.quantile(self.time_grid, 0.25)) & 
            (self.time_grid < np.quantile(self.time_grid, 0.75))
        ]
        for i in range(self.n_ldrs):
            resid_var[i] = traz(cut_time_grid, (grid_resid_var[i] - diag[i])) / 0.5

        return resid_var
    

def pace(ldrs, sub_time, unique_time, unique_time_map, time_grid, grid_mean, grid_cov, resid_var):
    """
    PACE estimator for time LDRs

    Parameters:
    ------------
    ldrs (n_obs, n_ldrs): a np.array of ldrs in long format 
    sub_time: a dictionary of sub:time, where sub should start with 0
    unique_time: 
    unique_time_map: a dictionary of mapping time to index
    time_grid: 
    mean: 
    grid_mean (n_ldrs, 51): a np.array of mean estimate
    grid_cov (n_ldrs, 51, 51): a np.array of cov estimate
    resid_var (n_ldrs, ): a np.array of resid var estimate

    Returns:
    ---------
    recon_spatial_ldrs (n_time, n_sub, n_ldrs): reconstructed spatial LDRs at each time point  
    
    """
    n_sub = len(sub_time)
    n_time = len(unique_time_map)
    n_ldrs = grid_mean.shape[0]
    time_spatial_ldrs = np.zeros((n_ldrs, n_sub, n_time), dtype=np.float32)
    recon_spatial_ldrs = np.zeros((n_ldrs, n_sub, n_time), dtype=np.float32)
    eg_values = np.zeros((n_ldrs, n_time), dtype=np.float32)
    eg_vectors = np.zeros((n_ldrs, n_time, n_time), dtype=np.float32)

    unique_time = unique_time / np.max(unique_time)
    time_grid = time_grid / np.max(time_grid)

    for i in range(n_ldrs):
        eg_values_, eg_vectors_ = np.linalg.eigh(grid_cov[i])
        eg_values_ = np.flip(eg_values_) # (n_time, )
        eg_vectors_ = np.flip(eg_vectors_, axis=1) # (n_time, n_time)
        eg_vectors_ = eg_vectors_[:, eg_values_ > 0]
        eg_values_ = eg_values_[eg_values_ > 0]
        
        fve = np.cumsum(eg_values_) / np.sum(eg_values_)
        n_opt = np.min([n_time, np.argmax(fve > 0.98) + 1])
        eg_values_ = eg_values_[:n_opt] * 0.02
        eg_vectors_ = eg_vectors_[:, :n_opt]
        interp_eg_vectors_ = np.zeros((n_time, n_opt), dtype=np.float32)

        for j in range(n_opt):
            eg_vectors_[:, j] = eg_vectors_[:, j] / np.sqrt(traz(time_grid, eg_vectors_[:, j] ** 2))
            if np.sum(eg_vectors_[:, j] * grid_mean[i]) < 0:
                eg_vectors_[:, j] = -eg_vectors_[:, j]
            interp_eg_vectors_[:, j] = np.interp(unique_time, time_grid, eg_vectors_[:, j])
        eg_values[i, :n_opt] = eg_values_
        eg_vectors[i, :, :n_opt] = interp_eg_vectors_

        fitted_cov = np.dot(interp_eg_vectors_ * eg_values_, interp_eg_vectors_.T)
        fitted_cov += np.diag([resid_var[i]] * n_time)
        interp_eg_vectors_ = interp_eg_vectors_ * eg_values_
        start, end = 0, 0
        for sub_idx, (_, time) in enumerate(sub_time.items()):
            end += len(time)
            time_idx = np.array([unique_time_map[t] for t in time])
            y_i = ldrs[start: end, i] # (n_time_i, )
            Sigma_i_inv = inv(fitted_cov[time_idx][:, time_idx]) # (n_time_i, n_time_i)
            eg_vector = interp_eg_vectors_[time_idx] # (n_time_i, n_opt)
            time_spatial_ldrs[i, sub_idx, :n_opt] = np.dot(np.dot(eg_vector.T, Sigma_i_inv), y_i)
            start = end
        
        # do reconstruction for time
        recon_spatial_ldrs[i] = np.dot(time_spatial_ldrs[i, :, :n_opt], eg_vectors[i, :, :n_opt].T)
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


class ReconLDRs:
    """
    Reading and managing reconstructed spatial LDRs
    
    """
    def __init__(self, file_path):
        self.file = h5py.File(file_path, "r")
        self.ldrs = self.file["ldrs"]
        self.n_time, self.n_sub, self.n_ldrs = self.ldrs.shape
        self.time = self.file["time"][:]
        ids = self.file["id"][:]
        self.ids = pd.MultiIndex.from_arrays(ids.astype(str).T, names=["FID", "IID"])
        self.id_idxs = np.arange(len(self.ids))
        # self.time_idxs = np.arange(len(self.time))
        self.ldr_col = (0, self.n_ldrs)
        self.logger = logging.getLogger(__name__)
        
    def close(self):
        self.file.close()
        
    def select_ldrs(self, ldr_col=None):
        """
        ldr_col: [start, end) of zero-based LDR index

        """
        if ldr_col is not None:
            if ldr_col[1] <= self.n_ldrs:
                self.ldr_col = ldr_col
                self.logger.info(
                    f"Keeping LDR{ldr_col[0]+1} to LDR{ldr_col[1]}."
                )
            else:
                raise ValueError(
                    f"{ldr_col[1]} is greater than #LDRs"
                )

    def keep(self, keep_idvs):
        """
        Keep subjects
        this method will only be invoked after extracting common subjects

        Parameters:
        ------------
        keep_idvs: a list or pd.MultiIndex of subject ids

        Returns:
        ---------
        self.id_idxs: numeric indices of subjects

        """
        if isinstance(keep_idvs, list):
            keep_idvs = pd.MultiIndex.from_arrays(
                [keep_idvs, keep_idvs], names=["FID", "IID"]
            )
        common_ids = ds.get_common_idxs(keep_idvs, self.ids).get_level_values("IID")
        ids_df = pd.DataFrame(
            {"id": self.id_idxs}, index=self.ids.get_level_values("IID")
        )
        ids_df = ids_df.loc[common_ids]
        self.id_idxs = ids_df["id"].values
        if len(self.id_idxs) == 0:
            raise ValueError("no subject remaining in LOCO predictions")

    def data_reader(self):
        """
        Reading LDRs for a time point

        """
        for t in range(self.n_time):
            spatial_ldrs = self.ldrs[t][self.id_idxs, self.ldr_col[0]:self.ldr_col[1]]
            spatial_ldrs_df = pd.DataFrame(spatial_ldrs, index=self.ids[self.id_idxs])
            spatial_ldrs_df = spatial_ldrs_df.reset_index(level=0, drop=True)
            yield spatial_ldrs_df


def check_input(args):
    # required arguments
    if args.spatial_ldrs is None:
        raise ValueError("--spatial-ldrs is required")
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
    unique_time = np.unique(time)
    unique_time_idx = {x: i for i, x in enumerate(unique_time)}
    time_idx = np.array([unique_time_idx[x] for x in time])
    grid_size = (unique_time[-1] - unique_time[0]) / 50
    time_grid = np.arange(unique_time[0], unique_time[-1] + grid_size, grid_size)
    ids = covar.data.index
    ldrs.to_single_index()
    n_obs = ldrs.data.index.value_counts(sort=False).values

    mean_estimator = Mean(ldrs_data, time, n_obs)
    mean, grid_mean = mean_estimator.estimate(0.05)
    
    cov_estimator = Covariance(ldrs_data, time, n_obs, mean, time_idx)
    grid_cov, cut_time_grid_diag = cov_estimator.estimate(0.1)

    resid_var_estimator = ResidualVariance(ldrs_data, time, n_obs)
    resid_var = resid_var_estimator.estimate(
        mean, cut_time_grid_diag, time_idx, 0.1
    )

    # PACE
    sub_time = ldrs.data.groupby("IID")["time"].apply(list).to_dict()
    unique_time_map = {t: i for i, t in enumerate(unique_time)}
    recon_spatial_ldrs = pace(
        ldrs_data, sub_time, unique_time, unique_time_map, time_grid, grid_mean, grid_cov, resid_var
    )

    # cov matrix of LDRs for each time
    ldr_cov_matrix = ldr_cov(recon_spatial_ldrs, np.array(covar.data))

    # save
    with h5py.File(f"{args.out}_recon_ldrs.h5", 'w') as file:
        file.create_dataset("ldrs", data=recon_spatial_ldrs, dtype="float32")
        file.create_dataset("id", data=np.array(ids.tolist(), dtype="S10"))
        file.create_dataset("time", data=unique_time)

    np.save(f"{args.out}_ldr_cov.npy", ldr_cov_matrix)

    log.info(f"\nSaved spatial temporal LDRs to {args.out}_recon_ldrs.h5")
    log.info(
        (
            f"Saved the variance-covariance matrix of covariate-effect-removed LDRs "
            f"to {args.out}_ldr_cov.npy"
        )
    )