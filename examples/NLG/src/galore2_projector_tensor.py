import torch
from tensorly import tenalg


class GaLore2ProjectorTensor:
    """
    GaLore2-style tensor projector using randomized low-rank projection.

    Compared with GaLoreProjectorTensor:
    - Original GaLore tensor version uses Tucker decomposition.
    - This version uses randomized SVD-style subspace estimation for each tensor mode.
    """

    def __init__(
        self,
        rank,
        verbose=False,
        update_proj_gap=200,
        scale=1.0,
        oversampling=8,
        power_iter=1,
    ):
        self.rank = rank
        self.verbose = verbose
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.oversampling = oversampling
        self.power_iter = power_iter

        self.ortho_matrix = None
        self.transformed_low_rank = None

    def project(self, full_rank_grad, iter):
        """
        Project full-rank tensor gradient into low-rank tensor subspace.
        """
        if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
            self.ortho_matrix = self.get_orthogonal_matrix(
                full_rank_grad,
                self.rank,
            )

        self.transformed_low_rank = self.transform(
            self.ortho_matrix,
            full_rank_grad,
        )
        return self.transformed_low_rank

    def project_back(self, low_rank_grad=None):
        """
        Project low-rank tensor gradient back to full-rank tensor space.
        """
        if low_rank_grad is None:
            low_rank_grad = self.transformed_low_rank

        full_rank_grad = self.inverse_transform(
            self.ortho_matrix,
            low_rank_grad,
        )

        return full_rank_grad * self.scale

    def get_orthogonal_matrix(self, weights, rank_all):
        """
        Compute randomized orthogonal bases for each tensor mode.

        Args:
            weights: full-rank tensor gradient
            rank_all: int or list/tuple of ranks for each mode

        Returns:
            list of orthogonal factor matrices [Q_0, Q_1, ..., Q_{N-1}]
            where Q_i has shape [dim_i, rank_i]
        """
        if weights.dtype != torch.float32:
            tensor = weights.float()
        else:
            tensor = weights

        ndim = tensor.ndim

        if isinstance(rank_all, int):
            ranks = [min(rank_all, tensor.shape[i]) for i in range(ndim)]
        else:
            assert len(rank_all) == ndim
            ranks = [min(rank_all[i], tensor.shape[i]) for i in range(ndim)]

        factors = []

        for mode in range(ndim):
            unfolded = self.unfold(tensor, mode)

            q = self.randomized_range_finder(
                unfolded,
                rank=ranks[mode],
                oversampling=self.oversampling,
                power_iter=self.power_iter,
            )

            factors.append(q)

        return factors

    def randomized_range_finder(self, matrix, rank, oversampling=8, power_iter=1):
        """
        Randomized SVD-style range finder.

        Given matrix A, approximate its dominant left singular subspace.

        Returns:
            Q with shape [A.shape[0], rank]
        """
        m, n = matrix.shape
        sample_rank = min(rank + oversampling, min(m, n))

        omega = torch.randn(
            n,
            sample_rank,
            device=matrix.device,
            dtype=matrix.dtype,
        )

        y = matrix @ omega

        for _ in range(power_iter):
            y = matrix @ (matrix.T @ y)

        q, _ = torch.linalg.qr(y, mode="reduced")

        return q[:, :rank].contiguous()

    def transform(self, factors, x):
        """
        Project tensor x into low-rank tensor space.

        Equivalent to:
            core = x ×_1 Q_1^T ×_2 Q_2^T ... ×_N Q_N^T
        """
        return tenalg.multi_mode_dot(x, factors, transpose=True)

    def inverse_transform(self, factors, x):
        """
        Project low-rank tensor back to original tensor space.

        Equivalent to:
            x_full = core ×_1 Q_1 ×_2 Q_2 ... ×_N Q_N
        """
        return tenalg.multi_mode_dot(x, factors, transpose=False)

    def unfold(self, tensor, mode):
        """
        Unfold tensor along a given mode.

        For tensor shape [d0, d1, ..., dn],
        mode-k unfolding has shape [dk, prod(other dims)].
        """
        ndim = tensor.ndim
        permute_order = [mode] + [i for i in range(ndim) if i != mode]
        unfolded = tensor.permute(permute_order).reshape(tensor.shape[mode], -1)
        return unfolded