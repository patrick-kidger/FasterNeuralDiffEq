import torch
import torchcde

from . import common


# Represents the system of a CDE; just calculates the change in hidden state
class _CDEFunc(common.NFECounter):
    def __init__(self, input_channels, hidden_channels, hidden_hidden_channels, num_hidden_layers):
        super(_CDEFunc, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.linear_in = torch.nn.Linear(hidden_channels, hidden_hidden_channels)
        self.linears = torch.nn.ModuleList(torch.nn.Linear(hidden_hidden_channels, hidden_hidden_channels)
                                           for _ in range(num_hidden_layers - 1))
        self.linear_out = torch.nn.Linear(hidden_hidden_channels, input_channels * hidden_channels)

    def forward(self, t, z):
        # t is of shape ()
        # z is of shape (batch, hidden_channels)

        hidden = z

        f_hidden = self.linear_in(hidden)
        f_hidden = f_hidden.relu()
        for linear in self.linears:
            f_hidden = linear(f_hidden)
            f_hidden = f_hidden.relu()
        f_hidden = self.linear_out(f_hidden).view(*f_hidden.shape[:-1], self.hidden_channels, self.input_channels)
        f_hidden = f_hidden.tanh()

        return f_hidden  # shape (batch, hidden_channels, input_channels)


class NeuralCDE(torch.nn.Module):
    def __init__(self, times, input_channels, hidden_channels, hidden_hidden_channels, num_hidden_layers,
                 output_channels, norm, rtol, atol):
        super(NeuralCDE, self).__init__()

        self.register_buffer('times', times)

        self.norm = norm
        self.rtol = rtol
        self.atol = atol

        self.func = _CDEFunc(input_channels, hidden_channels, hidden_hidden_channels, num_hidden_layers)
        self.initial = torch.nn.Linear(input_channels, hidden_channels)
        self.readout = torch.nn.Linear(hidden_channels, output_channels)

    @property
    def nfe(self):
        return self.func.nfe

    @property
    def ts(self):
        return self.func.ts

    def reset_nfe_ts(self):
        self.func.reset_nfe_ts()

    def forward(self, coeffs, y):
        # coeffs is of shape (batch, length, input_channels) if using any linear interpolation
        # y is of shape (batch,)

        X = torchcde.LinearInterpolation(coeffs, self.times)
        z0 = self.initial(X.evaluate(self.times[0]))
        options = dict(grid_points=X.grid_points, eps=1e-5)
        adjoint_options = options.copy()
        if self.norm:
            adjoint_options['norm'] = common.make_norm(z0)
        z_t = torchcde.cdeint(X=X,
                              z0=z0,
                              func=self.func,
                              t=self.times[[0, -1]],
                              rtol=self.rtol,
                              atol=self.atol,
                              options=options,
                              adjoint_options=adjoint_options)
        z_T = z_t[:, -1]
        pred_y = self.readout(z_T)

        loss = torch.nn.functional.cross_entropy(pred_y, y)
        thresholded_y = torch.argmax(pred_y, dim=1)
        accuracy = (thresholded_y == y).sum().to(pred_y.dtype)
        return loss, accuracy
