import bisect
import torch
import torchdiffeq

from . import common


class _CNNFunc(common.NFECounter):
    def __init__(self, hidden_channels, hidden_hidden_channels, num_pieces, tanh):
        super(_CNNFunc, self).__init__()

        self.num_pieces = num_pieces
        self.tanh = tanh

        self.convs = torch.nn.ModuleList()
        for _ in range(num_pieces):
            piece = torch.nn.Sequential(torch.nn.Conv2d(1 + hidden_channels, hidden_hidden_channels, 1),
                                        torch.nn.Softplus(),
                                        torch.nn.Conv2d(hidden_hidden_channels, hidden_hidden_channels, 3, padding=1),
                                        torch.nn.Softplus(),
                                        torch.nn.Conv2d(hidden_hidden_channels, hidden_channels, 1))
            self.convs.append(piece)

    def forward(self, t, x):
        # t is of shape ()
        # x is of shape (batch, hidden_channels, height, width)

        index = bisect.bisect(range(self.num_pieces + 1), t) - 1
        index = max(0, min(index, self.num_pieces - 1))
        conv = self.convs[index]

        t = t.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(x.size(0), 1, x.size(2), x.size(3))
        x = torch.cat([t, x], dim=1)
        out = conv(x)
        if self.tanh:
            out = out.tanh()
        return out


class NeuralODECNN(torch.nn.Module):
    def __init__(self, img_size, num_classes, hidden_channels, hidden_hidden_channels, num_pieces, norm, rtol, atol,
                 tanh):
        super(NeuralODECNN, self).__init__()

        input_channels, height, width = img_size

        self.num_pieces = num_pieces

        self.norm = norm
        self.rtol = rtol
        self.atol = atol

        self.augment = torch.nn.Linear(input_channels, hidden_channels)
        self.func = _CNNFunc(hidden_channels, hidden_hidden_channels, num_pieces, tanh)
        self.readout = torch.nn.Linear(hidden_channels * height * width, num_classes)

    @property
    def nfe(self):
        return self.func.nfe

    @property
    def ts(self):
        return self.func.ts

    def reset_nfe_ts(self):
        self.func.reset_nfe_ts()

    def forward(self, x, y):
        x = self.augment(x.transpose(1, 3)).transpose(1, 3)
        t = torch.tensor([0., self.num_pieces], dtype=x.dtype, device=x.device)

        # Tell the solver about the discontinuities
        grid_points = torch.linspace(0, self.num_pieces, self.num_pieces + 1, dtype=x.dtype, device=x.device)
        options = dict(grid_points=grid_points, eps=1e-5)
        adjoint_options = options.copy()
        
        if self.norm:
            adjoint_options['norm'] = common.make_norm(x)

        z_t = torchdiffeq.odeint_adjoint(self.func, x, t, rtol=self.rtol, atol=self.atol, options=options,
                                         adjoint_options=adjoint_options)
        z_T = z_t[1]
        pred_y = self.readout(z_T.flatten(1, -1))

        loss = torch.nn.functional.cross_entropy(pred_y, y)
        thresholded_y = torch.argmax(pred_y, dim=1)
        accuracy = (thresholded_y == y).sum().to(pred_y.dtype)
        return loss, accuracy
