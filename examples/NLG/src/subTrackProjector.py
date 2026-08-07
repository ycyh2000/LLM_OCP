import torch
class subTrackProjector:
    def __init__(
            self,
            rank,
            verbose=False,
            update_proj_gap=200,
            scale=1.0,
            proj_type='std',
            st_init_step_size=0,
            st_step_size_scheduler="iterative_decrease",
            st_step_size_coef=None,
            st_noise_sigma2=None,
            st_subspace_coef=None,
            use_acc_gradient=False
    ):
        self.rank = rank
        self.verbose = verbose
        self.update_proj_gap = update_proj_gap   # use as the minimum projection gap in  lotus
        self.scale = scale
        self.ortho_matrix = None
        self.prev_ortho_matrix = None
        self.proj_type = proj_type


        # subTrack parameter
        self.st_init_step_size = st_init_step_size
        self.st_step_size = st_init_step_size
        self.st_step_size_scheduler = st_step_size_scheduler
        self.st_step_size_coef = st_step_size_coef
        self.st_noise_sigma2 = st_noise_sigma2
        self.st_subspace_coef = st_subspace_coef

        self.use_acc_grad = use_acc_gradient
        self.accumulated_grad = None



    def projector(self, full_rank_grad, iter, rand=False):
        matrix = None
        if iter == 0:
            if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
                matrix = self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='right',
                                                                        rand=rand)
                low_rank_grad = torch.matmul(full_rank_grad, self.ortho_matrix.t())
                if self.ortho_matrix is not None:
                    self.prev_ortho_matrix = self.ortho_matrix.clone().detach()
            else:
                matrix = self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='left',
                                                                        rand=rand)
                low_rank_grad = torch.matmul(self.ortho_matrix.t(), full_rank_grad)
                if self.ortho_matrix is not None:
                    self.prev_ortho_matrix = self.ortho_matrix.clone().detach()

            if self.use_acc_grad:
                self.accumulated_grad = torch.zeros(full_rank_grad.shape, requires_grad=False, device='cuda')
                self.accumulated_grad += full_rank_grad

        elif (iter % self.update_proj_gap) != 0:
            if self.use_acc_grad:
                self.accumulated_grad += full_rank_grad

            if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
                low_rank_grad = torch.matmul(full_rank_grad, self.ortho_matrix.t())
            else:
                low_rank_grad = torch.matmul(self.ortho_matrix.t(), full_rank_grad)

        else:
            if self.st_step_size_scheduler == "iterative_decrease":
                self.st_step_size = self.st_init_step_size / iter

            if self.ortho_matrix is not None:
                self.prev_ortho_matrix = self.ortho_matrix.clone().detach()

            if self.use_acc_grad:
                self.accumulated_grad += full_rank_grad
                self.track_the_subspace(self.accumulated_grad / self.update_proj_gap)
                self.accumulated_grad = torch.zeros(full_rank_grad.shape, requires_grad=False, device='cuda')
            else:
                self.track_the_subspace(full_rank_grad)

            if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
                low_rank_grad = torch.matmul(full_rank_grad, self.ortho_matrix.t())
            else:
                low_rank_grad = torch.matmul(self.ortho_matrix.t(), full_rank_grad)

        return low_rank_grad, matrix

    def track_the_subspace(self, full_rank_grad):
        if self.ortho_matrix.dtype != torch.float:
            float_data = False
            original_type = self.ortho_matrix.dtype
            full_rank_grad = full_rank_grad.float()
            self.ortho_matrix = self.ortho_matrix.float()
        else:
            float_data = True

        if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
            estimated_w = torch.linalg.lstsq(
                self.ortho_matrix.t(), full_rank_grad.t()
            ).solution.t()
            residual = full_rank_grad - torch.matmul(estimated_w, self.ortho_matrix)
            partial_derivative = -2 * torch.matmul(estimated_w.t(), residual)
            tangent_vector = torch.matmul(
                partial_derivative,
                (torch.eye(self.ortho_matrix.shape[1]).to('cuda') - torch.matmul(self.ortho_matrix.t(),
                                                                                 self.ortho_matrix))
            )
        else:
            estimated_w = torch.linalg.lstsq(
                self.ortho_matrix, full_rank_grad
            ).solution
            residual = full_rank_grad - torch.matmul(self.ortho_matrix, estimated_w)
            partial_derivative = -2 * torch.matmul(residual, estimated_w.t())
            tangent_vector = torch.matmul(
                (torch.eye(self.ortho_matrix.shape[0]).to('cuda') - torch.matmul(self.ortho_matrix,
                                                                                 self.ortho_matrix.t())),
                partial_derivative
            )

        U, Sigma, V = subTrackProjector.rank_k_matrix_estimation(tangent_vector, k=1)

        if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
            self.ortho_matrix = torch.matmul(
                U, torch.matmul(
                    torch.concat([torch.cos(self.st_step_size * Sigma), torch.sin(-1 * self.st_step_size * Sigma)], 0),
                    torch.concat([torch.matmul(U.t(), self.ortho_matrix), V.t()], 0)
                ).reshape(Sigma.shape[0], self.ortho_matrix.shape[1])
            ) + torch.matmul(
                (torch.eye(U.shape[0]).to("cuda") - torch.matmul(U, U.t())), self.ortho_matrix
            )
        else:
            self.ortho_matrix = torch.matmul(
                torch.matmul(
                    torch.concat([torch.matmul(self.ortho_matrix, V), U], 1),
                    torch.concat([torch.cos(self.st_step_size * Sigma), torch.sin(-1 * self.st_step_size * Sigma)], 0)
                ).reshape((self.ortho_matrix.shape[0]), Sigma.shape[0]), V.t()
            ) + torch.matmul(
                self.ortho_matrix, (torch.eye(V.shape[0]).to("cuda") - torch.matmul(V, V.t()))
            )

        if not float_data:
            self.ortho_matrix = self.ortho_matrix.to(original_type)

    def project_back(self, low_rank_grad):
        if self.proj_type == 'std':
            if low_rank_grad.shape[0] >= low_rank_grad.shape[1]:
                full_rank_grad = torch.matmul(low_rank_grad, self.ortho_matrix)
            else:
                full_rank_grad = torch.matmul(self.ortho_matrix, low_rank_grad)
        elif self.proj_type == 'reverse_std':
            if low_rank_grad.shape[0] <= low_rank_grad.shape[1]:  # note this is different from std
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

    # svd decomposition
    def get_orthogonal_matrix(self, weights, rank, type, rand):
        module_params = weights

        if module_params.data.dtype != torch.float:
            float_data = False
            original_type = module_params.data.dtype
            original_device = module_params.data.device
            matrix = module_params.data.float()
        else:
            float_data = True
            matrix = module_params.data

        if not rand:
            U, s, Vh = torch.linalg.svd(matrix, full_matrices=False)

            # make the smaller matrix always to be orthogonal matrix
            if type == 'right':
                A = U[:, :rank] @ torch.diag(s[:rank])
                B = Vh[:rank, :]

                if not float_data:
                    B = B.to(original_device).type(original_type)
                return B
            elif type == 'left':
                A = U[:, :rank]
                B = torch.diag(s[:rank]) @ Vh[:rank, :]
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
                raise ValueError('type should be left, right or full')
        else:
            if type == 'right':
                B = self.get_random_orthogonal_matrix(rank, weights.shape[1])
                if not float_data:
                    B = B.to(original_device).type(original_type)
                return B
            elif type == 'left':
                A = self.get_random_orthogonal_matrix(weights.shape[0], rank)
                if not float_data:
                    A = A.to(original_device).type(original_type)
                return A

    @torch.no_grad()
    def get_random_orthogonal_matrix(self, n, m):
        # return torch.randn(n, m, dtype = dtype)
        Z = torch.randn(n, m)
        U, S, Vh = torch.linalg.svd(Z.T @ Z)
        S = 1 / torch.sqrt(S)
        return (Z @ U @ torch.diag(S) @ Vh)

    @staticmethod
    def rank_k_matrix_estimation(matrix, k=1):

        U, Sigma, Vt = torch.linalg.svd(matrix, full_matrices=False)

        if k == 1:
            return U[:, :k], Sigma[:k], Vt.t()[:, :k]
        return U[:, :k], Sigma[:k, :k], Vt.t()[:, :k]