import torch
from tensorly import tenalg
import math

class lotusProjectorTensor:
    """
    Tensor projector using randomized power-iteration SVD-style subspace estimation.

    Compared with PowerIterationSVDProjectorTensor:
    - Basic power iteration initializes a rank-r right subspace directly.
    - This version first uses randomized oversampling, then applies power iteration:
        Y = A Ω
        Y = (A A^T)^q A Ω
      to estimate the dominant left singular subspace of each tensor unfolding.
    """

    def __init__(
        self,
        rank,
        verbose=False,
        update_proj_gap=200,
        scale=1.0,
        oversampling=8,
        power_iter=1

        # CLEAR parameter
        , gamma=0.5
    ):
        self.rank = rank
        self.verbose = verbose
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.oversampling = oversampling
        self.power_iter = power_iter

        self.ortho_matrix = None
        self.transformed_low_rank = None


        # lotus
        # self.d_init = None
        # self.d_cur = None
        # self.gamma = gamma
        # self.current_iteration = 0
        # self.adaptive_projection_changing = True


        #CLEAR
        self.d_init = None
        self.g_init_fro_norm = 0
        self.adaptive_projection_changing = True
        self.rho_init = 0
        self.rho_cur = 0
        self.low_rank_fro_cos = 0
        self.full_rank_fro_cos_lower = 0
        self.full_rank_fro_cos_upper = 0
        self.gamma = gamma


    def project(self, full_rank_grad, iter):
        should_update_subspace = (
                self.ortho_matrix is None
                or self.adaptive_projection_changing
        )


        if should_update_subspace:
            self.ortho_matrix = self.get_orthogonal_matrix(
                full_rank_grad,
                self.rank,
            )

        self.transformed_low_rank = self.transform(
            self.ortho_matrix,
            full_rank_grad,
        )
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # calculate the gradient energy saved after SVD
        full_rank_grad_fro_norm = torch.linalg.matrix_norm(
            full_rank_grad.float(),
            ord="fro"
        )
        low_rank_grad_fro_norm = torch.linalg.matrix_norm(
            self.transformed_low_rank.float(),
            ord="fro"
        )
        self.rho_cur = low_rank_grad_fro_norm / full_rank_grad_fro_norm

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # calculate frobenius norm of current low rank gradient


        self.d_cur = self.transformed_low_rank / torch.linalg.matrix_norm(
            self.transformed_low_rank,
            ord="fro"
        ).clamp_min(1e-8)

        self.current_iteration = self.current_iteration + 1

        # save the low rank gradient
        if should_update_subspace:
            self.d_init = self.d_cur.detach().clone()
            self.current_iteration = 1

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # calculate needed to reefresh or not
        self.low_rank_fro_cos = torch.sum(self.d_cur * self.d_init)
        self.full_rank_fro_cos_lower = self.low_rank_fro_cos * math.sqrt(self.rho_cur * self.rho_init) - math.sqrt(
            (1 - self.rho_cur) * (1 - self.rho_init))
        # self.full_rank_fro_cos_upper = self.low_rank_fro_cos * math.sqrt(self.rho_cur * self.rho_init) + math.sqrt((1 - self.rho_cur) * (1 - self.rho_init))


        self.adaptive_projection_changing = (self.current_iteration != 1) and (self.current_iteration % self.update_proj_gap == 0) and (self.full_rank_fro_cos_lower < self.gamma)


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

            q = self.randomized_power_iteration_basis(
                unfolded,
                rank=ranks[mode],
                oversampling=self.oversampling,
                power_iter=self.power_iter,
            )

            factors.append(q)

        return factors

    @torch.no_grad()
    def randomized_power_iteration_basis(
        self,
        matrix,
        rank,
        oversampling=8,
        power_iter=1,
    ):
        """
        Randomized power-iteration basis estimator.

        Given unfolded matrix A with shape [m, n], estimate its dominant
        left singular subspace Q with shape [m, rank].

        Core formula:
            Y = A Ω
            Y = (A A^T)^q A Ω
            Q = orth(Y)
        """
        m, n = matrix.shape
        rank = min(rank, m, n)
        sample_rank = min(rank + oversampling, m, n)

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
        return tenalg.multi_mode_dot(x, factors, transpose=True)

    def inverse_transform(self, factors, x):
        return tenalg.multi_mode_dot(x, factors, transpose=False)

    def unfold(self, tensor, mode):
        ndim = tensor.ndim
        permute_order = [mode] + [i for i in range(ndim) if i != mode]
        unfolded = tensor.permute(permute_order).reshape(tensor.shape[mode], -1)
        return unfolded