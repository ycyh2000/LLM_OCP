import torch
from tensorly import tenalg


class PowerIterationSVDProjectorTensor:
    """
    Tensor projector using basic Power Iteration SVD-style subspace estimation.

    Compared with GaLore2ProjectorTensor:
    - GaLore2ProjectorTensor uses randomized range finder.
    - This version uses basic power iteration to estimate dominant left singular subspace
      for each tensor unfolding mode.
    """

    def __init__(
        self,
        rank,
        verbose=False,
        update_proj_gap=200,
        scale=1.0,
        power_iter=1,
    ):
        self.rank = rank
        self.verbose = verbose
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.power_iter = power_iter

        self.ortho_matrix = None
        self.transformed_low_rank = None

    def project(self, full_rank_grad, iter):
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
        if low_rank_grad is None:
            low_rank_grad = self.transformed_low_rank

        full_rank_grad = self.inverse_transform(
            self.ortho_matrix,
            low_rank_grad,
        )

        return full_rank_grad * self.scale

    @torch.no_grad()
    def get_orthogonal_matrix(self, weights, rank_all):
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

            q = self.power_iteration_svd_basis(
                unfolded,
                rank=ranks[mode],
                power_iter=self.power_iter,
            )

            factors.append(q)

        return factors

    @torch.no_grad()
    def power_iteration_svd_basis(self, matrix, rank, power_iter=1):
        """
        Basic Power Iteration SVD basis estimator.

        Given unfolded matrix A with shape [m, n],
        estimate its dominant left singular subspace Q with shape [m, rank].

        This approximates the left singular vectors U_r of A.
        """
        m, n = matrix.shape
        rank = min(rank, m, n)

        # Initialize right subspace: [n, rank]
        q = torch.randn(
            n,
            rank,
            device=matrix.device,
            dtype=matrix.dtype,
        )
        q, _ = torch.linalg.qr(q, mode="reduced")

        for _ in range(power_iter):
            # Estimate left subspace: [m, rank]
            p = matrix @ q
            p, _ = torch.linalg.qr(p, mode="reduced")

            # Refine right subspace: [n, rank]
            q = matrix.T @ p
            q, _ = torch.linalg.qr(q, mode="reduced")

        # Final left basis
        p = matrix @ q
        p, _ = torch.linalg.qr(p, mode="reduced")

        return p[:, :rank].contiguous()

    def transform(self, factors, x):
        return tenalg.multi_mode_dot(x, factors, transpose=True)

    def inverse_transform(self, factors, x):
        return tenalg.multi_mode_dot(x, factors, transpose=False)

    def unfold(self, tensor, mode):
        ndim = tensor.ndim
        permute_order = [mode] + [i for i in range(ndim) if i != mode]
        unfolded = tensor.permute(permute_order).reshape(tensor.shape[mode], -1)
        return unfolded