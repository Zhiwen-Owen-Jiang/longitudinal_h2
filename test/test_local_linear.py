import numpy as np
import pandas as pd
from abc import ABC, abstractmethod


class LocalLinearSlow(ABC):
    """
    Abstract class for local linear estimator (slow).
    
    """
    def __init__(self, ldrs, sub_time_idx, unique_time, bw):
        """
        ldrs (n_sub, n_time): a np.array of ldrs across time; fill 0 for missing time.
        sub_time: a dictionary of time points for each subject. 
        unique_time: a np.array of unique time points (normalized).
        
        """
        self.ldrs = ldrs
        self.sub_time_idx = sub_time_idx
        self.unique_time = unique_time / np.max(unique_time) # between 0 and 1
        self.n_sub, self.n_time = self.ldrs.shape
        self.bw = bw

        self.time_diff_table, self.kernel_table = self._get_tables()
        
    def _gau_kernel(self, x):
        """
        Calculating the Gaussian density

        """
        gau_k = 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * x**2)

        return gau_k
    
    def _get_tables(self):
        """
        Get kernel and time diff tables for quick look up.
        
        """
        time_diff = self.unique_time[:, np.newaxis] - self.unique_time
        time_diff_table = time_diff / self.bw
        kernel_table = self._gau_kernel(time_diff_table) / self.bw
        return time_diff_table, kernel_table
    
    @abstractmethod
    def sr(self):
        pass

    @abstractmethod
    def estimate(self):
        pass


class MeanSlow(LocalLinearSlow):
    """
    Local linear estimator for mean function using sparse data.
    
    """
    def sr(self, t):
        res_s = np.zeros(3)
        res_r = np.zeros(2)
        
        for sub, time_idxs in self.sub_time_idx.items():
            w = 1 / len(time_idxs)
            for time_idx in time_idxs:
                s0 = self.kernel_table[time_idx, t] * w
                s1 = s0 * self.time_diff_table[time_idx, t]
                s2 = s1 * self.time_diff_table[time_idx, t]
                r0 = s0 * self.ldrs[sub, time_idx]
                r1 = r0 * self.time_diff_table[time_idx, t]

                res_s[0] += s0
                res_s[1] += s1
                res_s[2] += s2
                res_r[0] += r0
                res_r[1] += r1
        
        return res_s / self.n_sub, res_r / self.n_sub
    
    def estimate(self):
        s = np.zeros((3, self.n_time))
        r = np.zeros((2, self.n_time))
        
        for i in range(self.n_time):
            s[:, i], r[:, i] = self.sr(i)
            
        mean = (r[0] * s[2] - r[1] * s[1]) / (s[0] * s[2] - s[1]**2)
        
        return mean


class CovarianceSlow(LocalLinearSlow):
    """
    Local linear estimator for covairiance function using sparse data.
    
    """
    def sr(self, t1, t2):
        """
        S00, S01, S02, 
        S10, S11, 
        S20
        
        R00, R01, 
        R10

        """
        res_s = np.zeros((3, 3))
        res_r = np.zeros((2, 2))
        for sub, time_idxs in self.sub_time_idx.items():
            n_time = len(time_idxs)
            w = 1 / (n_time * (n_time - 1))
            for i in range(len(time_idxs)):
                for j in range(len(time_idxs)):
                    if i != j:
                        s00 = (self.kernel_table[time_idxs[i], t1] * 
                               self.kernel_table[time_idxs[j], t2]) * w
                        s01 = s00 * self.time_diff_table[time_idxs[j], t2]
                        s02 = s01 * self.time_diff_table[time_idxs[j], t2]
                        s10 = s00 * self.time_diff_table[time_idxs[i], t1]
                        s11 = s10 * self.time_diff_table[time_idxs[j], t2]
                        s20 = s10 * self.time_diff_table[time_idxs[i], t1]
                        r00 = s00 * self.ldrs[sub, time_idxs[i]] * self.ldrs[sub, time_idxs[j]]
                        r01 = r00 * self.time_diff_table[time_idxs[j], t2]
                        r10 = r00 * self.time_diff_table[time_idxs[i], t1]

                        res_s[0, 0] += s00
                        res_s[0, 1] += s01
                        res_s[0, 2] += s02
                        res_s[1, 0] += s10
                        res_s[1, 1] += s11
                        res_s[2, 0] += s20
                        res_r[0, 0] += r00
                        res_r[0, 1] += r01
                        res_r[1, 0] += r10
        
        return res_s / self.n_sub, res_r / self.n_sub
    
    def estimate(self, mean):
        s = np.zeros((3, 3, self.n_time, self.n_time))
        r = np.zeros((2, 2, self.n_time, self.n_time))
        
        for i in range(self.n_time):
            for j in range(self.n_time):
                s[:, :, i, j], r[:, :, i, j] = self.sr(i, j)
                
        a1 = s[2, 0] * s[0, 2] - s[1, 1] * s[1, 1]
        a2 = s[1, 0] * s[0, 2] - s[1, 1] * s[0, 1]
        a3 = s[0, 1] * s[2, 0] - s[1, 0] * s[1, 1]
        b = a1 * s[0, 0] - a2 * s[1, 0] - a3 * s[0, 1]
        cov = (a1 * r[0, 0] - a2 * r[1, 0] - a3 * r[0, 1]) / b
        
        mean = mean.reshape(-1, 1)
        cov = cov - np.dot(mean, mean.T)
        
        return cov
    

class ResidualVarianceSlow(LocalLinearSlow):
    def sr(self, t):
        res_s = np.zeros(3, dtype=np.float32)
        res_r = np.zeros(2, dtype=np.float32)
        
        for sub, time_idxs in self.sub_time_idx.items():
            w = 1 / len(time_idxs)
            for time_idx in time_idxs:
                s0 = self.kernel_table[time_idx, t] * w
                s1 = s0 * self.time_diff_table[time_idx, t]
                s2 = s1 * self.time_diff_table[time_idx, t]
                r0 = s0 * self.ldrs[sub, time_idx] ** 2
                r1 = r0 * self.time_diff_table[time_idx, t]

                res_s[0] += s0
                res_s[1] += s1
                res_s[2] += s2
                res_r[0] += r0
                res_r[1] += r1
        
        return res_s / self.n_sub, res_r / self.n_sub
    
    def estimate(self, mean, cov):
        s = np.zeros((3, self.n_time), dtype=np.float32)
        r = np.zeros((2, self.n_time), dtype=np.float32)
        
        for i in range(self.n_time):
            s[:, i], r[:, i] = self.sr(i)
            
        residual_var = (r[0] * s[2] - r[1] * s[1]) / (s[0] * s[2] - s[1]**2)
        residual_var -= mean**2
        residual_var -= np.diag(cov)
        
        return residual_var
    

def run():
    ldrs = pd.read_csv('test/data/max_tib_area.txt', sep='\t')
    unique_time = np.unique(ldrs["time"])
    n_sub = len(ldrs['IID'].unique())
    unique_time_map = {t: i for i, t in enumerate(unique_time)}
    ldrs['time_idx'] = ldrs['time'].map(unique_time_map)
    
    ldrs_data = np.zeros((n_sub, len(unique_time)), dtype=np.float32)
    sub_time_idx = dict()
    for i, (_, data) in enumerate(ldrs.groupby("IID")):
        ldrs_data[i, data["time_idx"].values] = data["Max_Tib_area"].values 
        sub_time_idx[i] = data["time_idx"].values
        
    # mean
    mean_estimator = MeanSlow(ldrs_data, sub_time_idx, unique_time, 0.1)
    mean = mean_estimator.estimate()

    # cov
    cov_estimator = CovarianceSlow(ldrs_data, sub_time_idx, unique_time, 0.1)
    cov = cov_estimator.estimate(mean)

    # resid_var
    resid_var_estimator = ResidualVarianceSlow(ldrs_data, sub_time_idx, unique_time, 0.1)
    resid_var = resid_var_estimator.estimate(mean, cov)
    

if __name__ == "__main__":
    run()