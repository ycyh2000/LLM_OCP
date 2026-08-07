import math
import warnings
from typing import Callable, Iterable, Tuple
import torch
from torch import nn
from torch.optim import Optimizer
from transformers.utils.versions import require_version
global epsilon
epsilon = 1e-6

class adarankgradProjector:
    def __init__(self, min_rank, max_rank, rank_step, verbose=False, update_proj_gap=200, scale=1.0, proj_type='std', xi_threshold=0.2):
        self.min_rank = min_rank
        self.max_rank = max_rank
        self.rank_step = rank_step
        self.verbose = verbose
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.ortho_matrix = None
        # self.ortho_matrix_old = None
        self.const_k = False
        self.proj_type = proj_type
        self.xi_threshold = xi_threshold
        self.k_t = None

    def project(self, full_rank_grad, iter, update_projection_matrix=False):
        flag_updated = False
        if self.proj_type == 'std':
            if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
                if self.ortho_matrix is None or update_projection_matrix or iter % self.update_proj_gap == 0:
                    self.ortho_matrix, self.k_t, self.x_i = self.get_orthogonal_matrix(full_rank_grad, type='right')
                    flag_updated = True
                low_rank_grad = torch.matmul(full_rank_grad, self.ortho_matrix.t())
            else:
                if self.ortho_matrix is None or update_projection_matrix or iter % self.update_proj_gap == 0:
                    self.ortho_matrix, self.k_t, self.x_i = self.get_orthogonal_matrix(full_rank_grad, type='left')
                    flag_updated = True
                low_rank_grad = torch.matmul(self.ortho_matrix.t(), full_rank_grad)
        elif self.proj_type == 'reverse_std':
            if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
                if self.ortho_matrix is None or update_projection_matrix or iter % self.update_proj_gap == 0:
                    self.ortho_matrix, self.k_t, self.x_i = self.get_orthogonal_matrix(full_rank_grad, type='left')
                    flag_updated = True
                low_rank_grad = torch.matmul(self.ortho_matrix.t(), full_rank_grad)
            else:
                if self.ortho_matrix is None or update_projection_matrix or iter % self.update_proj_gap == 0:
                    self.ortho_matrix, self.k_t, self.x_i = self.get_orthogonal_matrix(full_rank_grad, type='right')
                    flag_updated = True
                low_rank_grad = torch.matmul(full_rank_grad, self.ortho_matrix.t())

        elif self.proj_type == 'right':
            if self.ortho_matrix is None or update_projection_matrix or iter % self.update_proj_gap == 0:
                self.ortho_matrix, self.k_t, self.x_i = self.get_orthogonal_matrix(full_rank_grad, type='right')
                flag_updated = True
            low_rank_grad = torch.matmul(full_rank_grad, self.ortho_matrix.t())
        elif self.proj_type == 'left':
            if self.ortho_matrix is None or update_projection_matrix or iter % self.update_proj_gap == 0:
                self.ortho_matrix, self.k_t, self.x_i = self.get_orthogonal_matrix(full_rank_grad, type='left')
                flag_updated = True
            low_rank_grad = torch.matmul(self.ortho_matrix.t(), full_rank_grad)
        elif self.proj_type == 'full':
            if self.ortho_matrix is None or update_projection_matrix or iter % self.update_proj_gap == 0:
                self.ortho_matrix, self.k_t, self.x_i = self.get_orthogonal_matrix(full_rank_grad, type='full')
                flag_updated = True
            low_rank_grad = torch.matmul(self.ortho_matrix[0].t(), full_rank_grad) @ self.ortho_matrix[1].t()
        return low_rank_grad, self.k_t, flag_updated

    def project_back(self, low_rank_grad):

        if self.proj_type == 'std':
            if low_rank_grad.shape[0] >= low_rank_grad.shape[1]:
                full_rank_grad = torch.matmul(low_rank_grad, self.ortho_matrix)
            else:
                full_rank_grad = torch.matmul(self.ortho_matrix, low_rank_grad)
        elif self.proj_type == 'reverse_std':
            if low_rank_grad.shape[0] <= low_rank_grad.shape[1]: # note this is different from std
                full_rank_grad = torch.matmul(self.ortho_matrix, low_rank_grad)
            else:
                full_rank_grad = torch.matmul(low_rank_grad, self.ortho_matrix)
        elif self.proj_type == 'right':
            full_rank_grad = torch.matmul(low_rank_grad, self.ortho_matrix)
        elif self.proj_type == 'left':
            full_rank_grad = torch.matmul(self.ortho_matrix, low_rank_grad)
        elif self.proj_type == 'full':
            full_rank_grad = torch.matmul(self.ortho_matrix[0], low_rank_grad) @ self.ortho_matrix[1]

        return full_rank_grad * self.scale


    def get_orthogonal_matrix(self, weights, type):
        module_params = weights
        if module_params.data.dtype != torch.float:
            float_data = False
            original_type = module_params.data.dtype
            original_device = module_params.data.device
            matrix = module_params.data.float()
        else:
            float_data = True
            matrix = module_params.data
        if self.const_k == True:
            U, s, Vh = torch.linalg.svd(matrix, full_matrices=False)
        else:
            U, S, V, k_t, xi = self.adaptive_projection(matrix)
        if type == 'right':
            if self.const_k == True:
                B = Vh[:k_t, :]
            else:
                B = V  # Vh[:rank, :]U#[:, :rank]
            if not float_data:
                B = B.to(original_device).type(original_type)
            return B, k_t, xi
        elif type == 'left':
            if self.const_k == True:
                A = U[:, :k_t]
            else:
                A = U
            if not float_data:
                A = A.to(original_device).type(original_type)
            return A, k_t, xi
        elif type == 'full':
            if self.const_k == True:
                B = Vh[:k_t, :]
                A = U[:, :k_t]
            else:
                A = U
                B = V
            if not float_data:
                A = A.to(original_device).type(original_type)
                B = B.to(original_device).type(original_type)
            return [A, B], k_t, xi
        else:
            raise ValueError('type should be left, right or full')

    def adaptive_projection(self, A):
        k_t = self.min_rank
        xi = torch.inf
        U, S, V = torch.svd_lowrank(A, self.max_rank * 2) # we multiply the rank value for more stable results
        V_T = V.T
        U, S, V_T = U[:, :self.max_rank], S[:self.max_rank], V_T[:self.max_rank,:]
        while xi >= self.xi_threshold and k_t <= self.max_rank:
            U_proj = U[:, :k_t]
            S_proj = S[:k_t]
            V_T_proj = V_T[:k_t, :]
            A_approx = torch.matmul(U_proj, torch.matmul(torch.diag(S_proj), V_T_proj))
            xi = torch.norm(A - A_approx) / (torch.norm(A) + 1e-8)
            k_t = k_t + self.rank_step
        return U_proj, S_proj, V_T_proj , k_t-self.rank_step, xi