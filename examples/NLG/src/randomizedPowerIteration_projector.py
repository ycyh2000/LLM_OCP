import torch


class RandomizedPowerIterationProjector:
    def __init__(
        self,
        rank,
        verbose=False,
        update_proj_gap=200,
        scale=1.0,
        proj_type='std',
        oversampling=10,
        power_iter=1,
        sign_align=True,
    ):
        self.rank = rank
        self.verbose = verbose
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.ortho_matrix = None
        self.proj_type = proj_type

        self.oversampling = oversampling
        self.power_iter = power_iter
        self.sign_align = sign_align

    def project(self, full_rank_grad, iter):
        if self.proj_type == 'std':
            if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    new_matrix = self.get_orthogonal_matrix(
                        full_rank_grad, self.rank, type='right'
                    )
                    self.ortho_matrix = self._maybe_sign_align(
                        new_matrix, self.ortho_matrix, type='right'
                    )

                low_rank_grad = full_rank_grad @ self.ortho_matrix.t().to(full_rank_grad.device)

            else:
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    new_matrix = self.get_orthogonal_matrix(
                        full_rank_grad, self.rank, type='left'
                    )
                    self.ortho_matrix = self._maybe_sign_align(
                        new_matrix, self.ortho_matrix, type='left'
                    )

                low_rank_grad = self.ortho_matrix.t().to(full_rank_grad.device) @ full_rank_grad

        elif self.proj_type == 'right':
            if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                new_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='right')
                self.ortho_matrix = self._maybe_sign_align(new_matrix, self.ortho_matrix, type='right')

            low_rank_grad = full_rank_grad @ self.ortho_matrix.t().to(full_rank_grad.device)

        elif self.proj_type == 'left':
            if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                new_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='left')
                self.ortho_matrix = self._maybe_sign_align(new_matrix, self.ortho_matrix, type='left')

            low_rank_grad = self.ortho_matrix.t().to(full_rank_grad.device) @ full_rank_grad

        elif self.proj_type == 'full':
            if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                new_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='full')
                self.ortho_matrix = self._maybe_sign_align(new_matrix, self.ortho_matrix, type='full')

            low_rank_grad = (
                self.ortho_matrix[0].t().to(full_rank_grad.device)
                @ full_rank_grad
                @ self.ortho_matrix[1].t().to(full_rank_grad.device)
            )

        else:
            raise ValueError("proj_type should be std, right, left, or full")

        return low_rank_grad

    def project_back(self, low_rank_grad):
        if self.proj_type == 'std':
            if low_rank_grad.shape[0] >= low_rank_grad.shape[1]:
                full_rank_grad = low_rank_grad @ self.ortho_matrix.to(low_rank_grad.device)
            else:
                full_rank_grad = self.ortho_matrix.to(low_rank_grad.device) @ low_rank_grad

        elif self.proj_type == 'right':
            full_rank_grad = low_rank_grad @ self.ortho_matrix.to(low_rank_grad.device)

        elif self.proj_type == 'left':
            full_rank_grad = self.ortho_matrix.to(low_rank_grad.device) @ low_rank_grad

        elif self.proj_type == 'full':
            full_rank_grad = (
                self.ortho_matrix[0].to(low_rank_grad.device)
                @ low_rank_grad
                @ self.ortho_matrix[1].to(low_rank_grad.device)
            )

        else:
            raise ValueError("proj_type should be std, right, left, or full")

        return full_rank_grad * self.scale

    @torch.no_grad()
    def get_orthogonal_matrix(self, weights, rank, type):
        if weights.data.dtype != torch.float32:
            float_data = False
            original_type = weights.data.dtype
            original_device = weights.data.device
            matrix = weights.data.float()
        else:
            float_data = True
            matrix = weights.data

        m, n = matrix.shape
        rank = min(rank, min(m, n))

        U, S, Vh = self._randomized_power_iteration_svd(matrix, rank)

        if type == 'right':
            B = Vh[:rank, :]
            if not float_data:
                B = B.to(original_device).type(original_type)
            return B

        elif type == 'left':
            A = U[:, :rank]
            if not float_data:
                A = A.to(original_device).type(original_type)
            return A

        elif type == 'full':
            A = U[:, :rank]
            B = Vh[:rank, :]
            if not float_data:
                A = A.to(original_device).type(original_type)
                B = B.to(original_device).type(original_type)
            return [A, B]

        else:
            raise ValueError("type should be left, right or full")

    @torch.no_grad()
    def _randomized_power_iteration_svd(self, matrix, rank):
        """
        Randomized power-iteration SVD.

        Core idea:
            Y = A Omega
            Y = (A A^T)^q A Omega
            Q = orth(Y)
            B = Q^T A
            B = U_hat S Vh
            U = Q U_hat
        """
        m, n = matrix.shape
        k = min(rank + self.oversampling, min(m, n))

        if m <= n:
            omega = torch.randn(n, k, device=matrix.device, dtype=matrix.dtype)

            Y = matrix @ omega

            for _ in range(self.power_iter):
                Y = matrix @ (matrix.t() @ Y)

            Q, _ = torch.linalg.qr(Y, mode='reduced')

            B = Q.t() @ matrix

            U_hat, S, Vh = torch.linalg.svd(B, full_matrices=False)

            U = Q @ U_hat

        else:
            omega = torch.randn(m, k, device=matrix.device, dtype=matrix.dtype)

            Y = matrix.t() @ omega

            for _ in range(self.power_iter):
                Y = matrix.t() @ (matrix @ Y)

            Q, _ = torch.linalg.qr(Y, mode='reduced')

            B = matrix @ Q

            U, S, V_hat_h = torch.linalg.svd(B, full_matrices=False)

            Vh = V_hat_h @ Q.t()

        return U, S, Vh

    @torch.no_grad()
    def _maybe_sign_align(self, new_matrix, old_matrix, type):
        if not self.sign_align or old_matrix is None:
            return new_matrix

        if type in ['left', 'right']:
            if new_matrix.shape != old_matrix.shape:
                return new_matrix

            if type == 'left':
                dots = torch.sum(
                    new_matrix.float() * old_matrix.to(new_matrix.device).float(),
                    dim=0
                )
                signs = torch.where(
                    dots < 0,
                    torch.tensor(-1.0, device=new_matrix.device),
                    torch.tensor(1.0, device=new_matrix.device)
                ).to(new_matrix.dtype)
                return new_matrix * signs.view(1, -1)

            else:
                dots = torch.sum(
                    new_matrix.float() * old_matrix.to(new_matrix.device).float(),
                    dim=1
                )
                signs = torch.where(
                    dots < 0,
                    torch.tensor(-1.0, device=new_matrix.device),
                    torch.tensor(1.0, device=new_matrix.device)
                ).to(new_matrix.dtype)
                return new_matrix * signs.view(-1, 1)

        elif type == 'full':
            if old_matrix is None:
                return new_matrix

            A_new, B_new = new_matrix
            A_old, B_old = old_matrix

            if A_new.shape != A_old.shape or B_new.shape != B_old.shape:
                return new_matrix

            dots = torch.sum(
                A_new.float() * A_old.to(A_new.device).float(),
                dim=0
            )
            signs = torch.where(
                dots < 0,
                torch.tensor(-1.0, device=A_new.device),
                torch.tensor(1.0, device=A_new.device)
            ).to(A_new.dtype)

            A_new = A_new * signs.view(1, -1)
            B_new = B_new * signs.view(-1, 1)

            return [A_new, B_new]

        else:
            return new_matrix