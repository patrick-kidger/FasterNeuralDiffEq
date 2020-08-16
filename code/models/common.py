import torch


def _rms_norm(tensor):
    return tensor.pow(2).mean().sqrt()


def make_norm(state):
    state_size = state.numel()

    def _norm(aug_state):
        # aug_state is a tensor of shape [1 + state_size + state_size + number_of_parameters]
        # we want to extract the two bits of size 'state_size', corresponding to the state and to the adjoint wrt the
        # the state. We ignore the '1' (which is the gradient wrt time, which isn't used as we don't compute derivatives
        # wrt time), and the 'number_of_parameters' (which is the whole point of this work).
        y = aug_state[1:1 + state_size]
        adj_y = aug_state[1 + state_size:1 + 2 * state_size]
        return max(_rms_norm(y), _rms_norm(adj_y))
    return _norm


# Counts the number of function evaluations
class NFECounter(torch.nn.Module):
    def __init__(self):
        super(NFECounter, self).__init__()
        self.nfe = 0
        self.ts = []

    def reset_nfe_ts(self):
        self.nfe = 0
        self.ts = []

    def __call__(self, t, z):
        self.nfe += 1
        self.ts.append(t.item())
        return super(NFECounter, self).__call__(t, z)


def count_accept_rejects(ts, first_step_given=False, evals_per_step=6):
    if first_step_given:
        offset = evals_per_step - 1
    else:
        offset = evals_per_step + 1

    step_times = ts[offset::evals_per_step]
    accepts = []
    rejects = []
    for t, next_t in zip(step_times[:-1], step_times[1:]):
        if next_t < t:
            rejects.append(t)
        else:
            accepts.append(t)
    accepts.append(step_times[-1])

    return accepts, rejects


if __name__ == "__main__":

    from torchdiffeq import odeint

    class ODEFunc(NFECounter):

        def forward(self, t, x):
            return (x - 0.1) * -0.1

    torch.manual_seed(0)
    x0 = torch.randn(10, 1)
    func = ODEFunc()

    xs = odeint(func, x0, torch.linspace(0, 10, 10), method="dopri5", options={"first_step": 10.0})

    print(func.nfe)
    print(func.ts)

    accepts, rejects = count_accept_rejects(func.ts, first_step_given=True)
    print(accepts)
    print(rejects)

    func.reset_nfe_ts()

    xs = odeint(func, x0, torch.linspace(0, 10, 10), method="dopri5", options={"first_step": None})

    print(func.nfe)
    print(func.ts)

    accepts, rejects = count_accept_rejects(func.ts, first_step_given=False)
    print(accepts)
    print(rejects)
