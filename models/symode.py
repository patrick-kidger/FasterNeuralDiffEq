# Taken from https://github.com/d-biswa/Symplectic-ODENet/blob/e56a5d5d63bebf3810125f19ff84edc6cc277286/symoden.py#L90
# I've cut out all the options that aren't used for this experiment.
# As a result this isn't very flexible.

import numpy as np
import torch
import torchdiffeq

from . import common


class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h = torch.nn.functional.softplus(self.linear1(x))
        h = torch.nn.functional.softplus(self.linear2(h))
        return self.linear3(h)


class PSD(torch.nn.Module):
    '''A Neural Net which outputs a positive semi-definite matrix'''

    def __init__(self, input_dim, hidden_dim, diag_dim):
        super(PSD, self).__init__()
        assert diag_dim > 1
        self.diag_dim = diag_dim
        self.off_diag_dim = int(diag_dim * (diag_dim - 1) / 2)
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = torch.nn.Linear(hidden_dim, self.diag_dim + self.off_diag_dim)

    def forward(self, q):
        bs = q.shape[0]
        h = torch.nn.functional.softplus(self.linear1(q))
        h = torch.nn.functional.softplus(self.linear2(h))
        h = torch.nn.functional.softplus(self.linear3(h))
        diag, off_diag = torch.split(self.linear4(h), [self.diag_dim, self.off_diag_dim], dim=1)

        L = torch.diag_embed(diag)

        ind = np.tril_indices(self.diag_dim, k=-1)
        flat_ind = np.ravel_multi_index(ind, (self.diag_dim, self.diag_dim))
        L = torch.flatten(L, start_dim=1)
        L[:, flat_ind] = off_diag
        L = torch.reshape(L, (bs, self.diag_dim, self.diag_dim))

        D = torch.bmm(L, L.permute(0, 2, 1))
        D[:, 0, 0] = D[:, 0, 0] + 0.1
        D[:, 1, 1] = D[:, 1, 1] + 0.1
        return D


class MatrixNet(torch.nn.Module):
    ''' a neural net which outputs a matrix'''

    def __init__(self, input_dim, hidden_dim, output_dim, shape=(2, 2)):
        super(MatrixNet, self).__init__()
        self.mlp = MLP(input_dim, hidden_dim, output_dim)
        self.shape = shape

    def forward(self, x):
        flatten = self.mlp(x)
        return flatten.view(-1, *self.shape)


class SymODEN_T(common.NFECounter):
    '''
    Architecture for input (cos q, sin q, q_dot, u),
    where q represent angles, a tensor of size (bs, n),
    cos q, sin q and q_dot are tensors of size (bs, n), and
    u is a tensor of size (bs, 1).
    '''

    def __init__(self, input_dim, M_net=None, V_net=None, g_net=None, u_dim=1):
        super(SymODEN_T, self).__init__()
        self.M_net = M_net
        self.u_dim = u_dim
        self.V_net = V_net
        self.g_net = g_net

        self.input_dim = input_dim

    def forward(self, t, x):
        with torch.enable_grad():
            bs = x.shape[0]
            zero_vec = torch.zeros(bs, self.u_dim, dtype=torch.float32, device=x.device)

            cos_q_sin_q, q_dot, u = torch.split(x, [2 * self.input_dim, self.input_dim, self.u_dim], dim=1)
            M_q_inv = self.M_net(cos_q_sin_q)
            q_dot_aug = torch.unsqueeze(q_dot, dim=2)
            p = torch.squeeze(torch.matmul(torch.inverse(M_q_inv), q_dot_aug), dim=2)
            cos_q_sin_q_p = torch.cat((cos_q_sin_q, p), dim=1)
            cos_q_sin_q, p = torch.split(cos_q_sin_q_p, [2 * self.input_dim, 1 * self.input_dim], dim=1)
            M_q_inv = self.M_net(cos_q_sin_q)
            cos_q, sin_q = torch.chunk(cos_q_sin_q, 2, dim=1)

            V_q = self.V_net(cos_q_sin_q)
            p_aug = torch.unsqueeze(p, dim=2)
            H = torch.squeeze(torch.matmul(torch.transpose(p_aug, 1, 2),
                                           torch.matmul(M_q_inv, p_aug))) / 2.0 + torch.squeeze(V_q)
            dH = torch.autograd.grad(H.sum(), cos_q_sin_q_p, create_graph=True)[0]
            dHdcos_q, dHdsin_q, dHdp = torch.split(dH, [self.input_dim, self.input_dim, self.input_dim], dim=1)
            g_q = self.g_net(cos_q_sin_q)
            F = torch.squeeze(torch.matmul(g_q, torch.unsqueeze(u, dim=2)))

            dq = dHdp
            dp = sin_q * dHdcos_q - cos_q * dHdsin_q + F

            dM_inv_dt = torch.zeros_like(M_q_inv)
            for row_ind in range(self.input_dim):
                for col_ind in range(self.input_dim):
                    dM_inv = \
                    torch.autograd.grad(M_q_inv[:, row_ind, col_ind].sum(), cos_q_sin_q, create_graph=True)[0]
                    dM_inv_dt[:, row_ind, col_ind] = (dM_inv * torch.cat((-sin_q * dq, cos_q * dq), dim=1)).sum(-1)
            ddq = torch.squeeze(torch.matmul(M_q_inv, torch.unsqueeze(dp, dim=2)), dim=2) \
                  + torch.squeeze(torch.matmul(dM_inv_dt, torch.unsqueeze(p, dim=2)), dim=2)

            return torch.cat((-sin_q * dq, cos_q * dq, ddq, zero_vec), dim=1)


class SymODE(torch.nn.Module):
    def __init__(self, times, norm, rtol, atol):
        super(SymODE, self).__init__()

        self.register_buffer('times', times)

        self.norm = norm
        self.rtol = rtol
        self.atol = atol

        M_net = PSD(4, 400, 2)
        V_net = MLP(4, 300, 1)
        g_net = MatrixNet(4, 300, 4, shape=(2, 2))
        self.func = SymODEN_T(2, M_net=M_net, V_net=V_net, g_net=g_net, u_dim=2)

    @property
    def nfe(self):
        return self.func.nfe

    @property
    def ts(self):
        return self.func.ts

    def reset_nfe_ts(self):
        self.func.reset_nfe_ts()

    def forward(self, x):
        if self.norm:
            adjoint_options = dict(norm=common.make_norm(x))
        else:
            adjoint_options = None

        z = torchdiffeq.odeint_adjoint(self.func, x[:, 0], self.times, rtol=self.rtol, atol=self.atol,
                                       adjoint_options=adjoint_options)

        loss = torch.nn.functional.mse_loss(x, z.transpose(0, 1))
        return loss
