import numpy as np
import pandas as pd
from abc import ABC, abstractmethod


class LocalLinearSlow(ABC):
    """
    Abstract class for local linear estimator (slow).
    
    """
    def __init__(self, ldrs, sub_time, unique_time, bw):
        """
        ldrs (n_sub, n_time): a np.array of ldrs across time; fill 0 for missing time.
        sub_time: a dictionary of time points for each subject. 
        unique_time: a np.array of unique time points (normalized).
        
        """
        self.ldrs = ldrs
        self.sub_time = sub_time
        self.unique_time = unique_time # between 0 and 1
        self.n_sub, self.n_time = self.ldrs.shape
        self.bw = bw

        self.kernel_table = self._get_kernel_table()
        self.time_diff_table = self._get_time_diff_table()
        
    def _gau_kernel(self, x):
        """
        Calculating the Gaussian density

        """
        gau_k = 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * x**2)

        return gau_k
    
    def _get_kernel_table(self):
        """
        Get kernel table for quick look up.
        
        """
        time_diff = self.unique_time - self.unique_time[:, np.newaxis]
        kernel_table = self._gau_kernel(time_diff / self.bw) / self.bw
        return kernel_table
    
    def _get_time_diff_table(self):
        """
        Get time difference table for quick look up.
        
        """
        time_diff = self.unique_time - self.unique_time[:, np.newaxis]
        time_diff_table = time_diff / self.bw
        
        return time_diff_table
    
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
        res_s = np.zeros(3, dtype=np.float32)
        res_r = np.zeros(2, dtype=np.float32)
        
        for sub, time_idxs in self.sub_time.items():
            w = 1 / len(time_idxs)
            for time_idx in time_idxs:
                res_s[0] += self.kernel_table[time_idx, t] * w
                res_s[1] += res_s[0] * self.time_diff_table[time_idx, t]
                res_s[2] += res_s[1] * self.time_diff_table[time_idx, t]

                res_r[0] += res_s[0] * self.ldrs[sub, time_idx]
                res_r[1] += res_r[0] * self.time_diff_table[time_idx, t]
        
        return res_s / self.n_sub, res_r / self.n_sub
    
    def estimate(self):
        s = np.zeros((3, self.n_time), dtype=np.float32)
        r = np.zeros((2, self.n_time), dtype=np.float32)
        
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
        res_s = np.zeros((3, 3), dtype=np.float32)
        res_r = np.zeros((2, 2), dtype=np.float32)
        for sub, time_idxs in self.sub_time.items():
            n_time = len(time_idxs)
            w = 1 / n_time * (n_time - 1)
            for i in range(len(time_idxs) - 1):
                for j in range(i + 1, len(time_idxs)):
                    res_s[0, 0] += (
                        self.kernel_table[time_idxs[i], t1] * 
                        self.kernel_table[time_idxs[j], t2]
                    ) * w
                    res_s[0, 1] += res_s[0, 0] * self.time_diff_table[time_idxs[j], t2]
                    res_s[0, 2] += res_s[0, 1] * self.time_diff_table[time_idxs[j], t2]
                    res_s[1, 0] += res_s[0, 0] * self.time_diff_table[time_idxs[i], t1]
                    res_s[1, 1] += res_s[1, 0] * self.time_diff_table[time_idxs[j], t2]
                    res_s[2, 0] += res_s[1, 0] * self.time_diff_table[time_idxs[i], t1]
                    
                    res_r[0, 0] += res_s[0, 0] * self.ldrs[sub, time_idxs[i]] * self.ldrs[sub, time_idxs[j]]
                    res_r[0, 1] += res_r[0, 0] * self.time_diff_table[time_idxs[j], t2]
                    res_r[1, 0] += res_r[0, 0] * self.time_diff_table[time_idxs[i], t1]
        
        return res_s * 2 / self.n_sub, res_r * 2 / self.n_sub
    
    def estimate(self):
        s = np.zeros((3, 3, self.n_time, self.n_time), dtype=np.float32)
        r = np.zeros((2, 2, self.n_time, self.n_time), dtype=np.float32)
        
        for i in range(self.n_time):
            for j in range(i, self.n_time):
                s[:, :, i, j], r[:, :, i, j] = self.sr(i, j)
                
        a1 = s[2, 0] * s[0, 2] - s[1, 1] * s[1, 1]
        a2 = s[1, 0] * s[0, 2] - s[1, 1] * s[0, 1]
        a3 = s[0, 1] * s[2, 0] - s[1, 0] * s[1, 1]
        b = a1 * s[0, 0] - a2 * s[1, 0] - a3 * s[0, 1]
        cov = (a1 * r[0, 0] - a2 * r[1, 0] - a3 * r[0, 1]) / b
        
        return cov
    

class ResidualVarianceSlow(LocalLinearSlow):
    def sr(self, t):
        res_s = np.zeros(3, dtype=np.float32)
        res_r = np.zeros(2, dtype=np.float32)
        
        for sub, time_idxs in self.sub_time.items():
            w = 1 / len(time_idxs)
            for time_idx in time_idxs:
                res_s[0] += self.kernel_table[time_idx, t] * w
                res_s[1] += res_s[0] * self.time_diff_table[time_idx, t]
                res_s[2] += res_s[1] * self.time_diff_table[time_idx, t]

                res_r[0] += res_s[0] * self.ldrs[sub, time_idx] ** 2
                res_r[1] += res_r[0] * self.time_diff_table[time_idx, t]
        
        return res_s / self.n_sub, res_r / self.n_sub
    
    def estimate(self):
        s = np.zeros((3, self.n_time), dtype=np.float32)
        r = np.zeros((2, self.n_time), dtype=np.float32)
        
        for i in range(self.n_time):
            s[:, i], r[:, i] = self.sr(i)
            
        residual_var = (r[0] * s[2] - r[1] * s[1]) / (s[0] * s[2] - s[1]**2)
        
        return residual_var