# Taken from https://github.com/d-biswa/Symplectic-ODENet/blob/master/myenv/fa_acrobot.py
# Slight adaptations to tidy the code a bit (it could still do with a lot more tidying).


import numpy as np
from numpy import sin, cos
from gym import core, spaces
from gym.utils import seeding
from scipy.integrate import solve_ivp
import torch
import warnings

from . import common


class AcrobotEnv(core.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 15
    }

    dt = .05

    LINK_LENGTH_1 = 1.  # [m]
    LINK_LENGTH_2 = 1.  # [m]
    LINK_MASS_1 = 1.  #: [kg] mass of link 1
    LINK_MASS_2 = 1.  #: [kg] mass of link 2
    LINK_COM_POS_1 = 0.5  #: [m] position of the center of mass of link 1
    LINK_COM_POS_2 = 0.5  #: [m] position of the center of mass of link 2
    LINK_MOI = 1.  #: moments of inertia for both links

    MAX_VEL_1 = 4 * np.pi
    MAX_VEL_2 = 9 * np.pi

    AVAIL_TORQUE = [-1., 0., +1]

    torque_noise_max = 0.

    #: use dynamics equations from the SymODENet nips paper or the book
    book_or_nips = "book"
    action_arrow = None
    domain_fig = None
    actions_num = 3

    def __init__(self):
        self.viewer = None
        high = np.array([1.0, 1.0, 1.0, 1.0, self.MAX_VEL_1, self.MAX_VEL_2])
        low = -high
        high_a = np.array([15.0, 15.0])
        low_a = -high_a
        warnings.simplefilter('ignore')
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Box(low=low_a, high=high_a, dtype=np.float32)
        self.state = None
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.state = self.np_random.uniform(low=-3.14, high=3.14, size=(4,))
        return self._get_ob()

    def step(self, a):
        # s = self.state

        ivp = solve_ivp(fun=lambda t, y: self.dyna_wrapper(t, y, a), t_span=[0, self.dt], y0=self.state)
        self.state = ivp.y[:, -1]
        terminal = self._terminal()
        reward = -1. if not terminal else 0.
        return self._get_ob(), reward, False, {}

    def _get_ob(self):
        s = self.state
        return np.array([cos(s[0]), np.sin(s[0]), cos(s[1]), sin(s[1]), s[2], s[3]])

    def _terminal(self):
        s = self.state
        return bool(-np.cos(s[0]) - np.cos(s[1] + s[0]) > 1.)

    def dyna_wrapper(self, t, y, u):
        f = np.zeros_like(y)
        f[0], f[1], f[2], f[3], _, _ = self._dsdt(np.concatenate((y, np.array(u))), t)
        return f

    def _dsdt(self, s_augmented, t):
        m1 = self.LINK_MASS_1
        m2 = self.LINK_MASS_2
        l1 = self.LINK_LENGTH_1
        lc1 = self.LINK_COM_POS_1
        lc2 = self.LINK_COM_POS_2
        I1 = self.LINK_MOI
        I2 = self.LINK_MOI
        g = 9.8
        a = s_augmented[-2:]
        s = s_augmented[:-2]
        theta1 = s[0]
        theta2 = s[1]
        dtheta1 = s[2]
        dtheta2 = s[3]
        d1 = m1 * lc1 ** 2 + m2 * \
             (l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * np.cos(theta2)) + I1 + I2
        d2 = m2 * (lc2 ** 2 + l1 * lc2 * np.cos(theta2)) + I2
        phi2 = m2 * lc2 * g * np.cos(theta1 + theta2 - np.pi / 2.)
        phi1 = - m2 * l1 * lc2 * dtheta2 ** 2 * np.sin(theta2) \
               - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * np.sin(theta2) \
               + (m1 * lc1 + m2 * l1) * g * np.cos(theta1 - np.pi / 2) + phi2
        if self.book_or_nips == "nips":
            # the following line is consistent with the description in the
            # paper
            ddtheta2 = (a[0] + d2 / d1 * phi1 - phi2) / \
                       (m2 * lc2 ** 2 + I2 - d2 ** 2 / d1)
        else:
            # the following line is consistent with the java implementation and the
            # book
            ddtheta2 = (a[0] + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1 ** 2 * np.sin(theta2) - phi2) \
                       / (m2 * lc2 ** 2 + I2 - d2 ** 2 / d1)
        ddtheta1 = -(a[1] + d2 * ddtheta2 + phi1) / d1
        return (dtheta1, dtheta2, ddtheta1, ddtheta2, 0., 0.)

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering

        s = self.state

        if self.viewer is None:
            self.viewer = rendering.Viewer(500, 500)
            bound = self.LINK_LENGTH_1 + self.LINK_LENGTH_2 + 0.2  # 2.2 for default
            self.viewer.set_bounds(-bound, bound, -bound, bound)

        if s is None: return None

        p1 = [-self.LINK_LENGTH_1 *
              np.cos(s[0]), self.LINK_LENGTH_1 * np.sin(s[0])]

        p2 = [p1[0] - self.LINK_LENGTH_2 * np.cos(s[0] + s[1]),
              p1[1] + self.LINK_LENGTH_2 * np.sin(s[0] + s[1])]

        xys = np.array([[0, 0], p1, p2])[:, ::-1]
        thetas = [s[0] - np.pi / 2, s[0] + s[1] - np.pi / 2]
        link_lengths = [self.LINK_LENGTH_1, self.LINK_LENGTH_2]

        self.viewer.draw_line((-2.2, 1), (2.2, 1))
        for ((x, y), th, llen) in zip(xys, thetas, link_lengths):
            l, r, t, b = 0, llen, .1, -.1
            jtransform = rendering.Transform(rotation=th, translation=(x, y))
            link = self.viewer.draw_polygon([(l, b), (l, t), (r, t), (r, b)])
            link.add_attr(jtransform)
            link.set_color(0, .8, .8)
            circ = self.viewer.draw_circle(.1)
            circ.set_color(.8, .8, 0)
            circ.add_attr(jtransform)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


def sample_gym(seed, trials, u, timesteps):
    env = AcrobotEnv()
    env.seed(seed)

    trajs = []
    for trial in range(trials):
        valid = False
        while not valid:
            env.reset()
            traj = []
            for step in range(timesteps):
                obs, _, _, _ = env.step(u)  # action
                x = np.array([obs[0], obs[2], obs[1], obs[3], obs[4], obs[5], u[0], u[1]])
                traj.append(x)
            traj = np.stack(traj)
            if np.amax(traj[:, 4]) < env.MAX_VEL_1 - 0.001 and np.amin(traj[:, 4]) > -env.MAX_VEL_1 + 0.001:
                if np.amax(traj[:, 5]) < env.MAX_VEL_2 - 0.001 and np.amin(traj[:, 5]) > -env.MAX_VEL_2 + 0.001:
                    valid = True
        trajs.append(traj)
    trajs = np.stack(trajs)  # (trials, timesteps, 2)
    trajs = np.transpose(trajs, (1, 0, 2))  # (timesteps, trails, 2)
    tspan = np.arange(timesteps) * 0.05
    return trajs, tspan


def get_dataset(seed, samples, us, timesteps):
    trajs_force = []
    for u in us:
        trajs, tspan = sample_gym(seed=seed, trials=samples, u=u, timesteps=timesteps)
        trajs_force.append(trajs)
    x = np.stack(trajs_force, axis=0)  # (3, 45, 50, 3)
    # make a train/test split
    split_ix = int(samples * 0.7)
    split_ix2 = int(samples * 0.15)
    train_x, val_x, test_x = x[:, :, :split_ix, :], x[:, :, split_ix:split_ix + split_ix2, :], x[:, :, split_ix + split_ix2:, :]
    return tspan, train_x, val_x, test_x


def arrange_data(x, t, num_points=2):
    assert num_points >= 2 and num_points <= len(t)
    x_stack = []
    for i in range(num_points):
        if i < num_points - 1:
            x_stack.append(x[:, i:-num_points + i + 1, :, :])
        else:
            x_stack.append(x[:, i:, :, :])
    x_stack = np.stack(x_stack, axis=1)
    x_stack = np.reshape(x_stack,
                         (x.shape[0], num_points, -1, x.shape[3]))
    t_eval = t[0:num_points]

    x_stack = torch.tensor(x_stack, dtype=torch.float32).transpose(1, 2).reshape(-1, num_points, x_stack.shape[-1])
    t_eval = torch.tensor(t_eval, dtype=torch.float32)
    return x_stack, t_eval


def acrobot(batch_size):
    us = [[0.0, 0.0], [0.0, 1.0], [0.0, -1.0], [0.0, 2.0], [0.0, -2.0],
          [1.0, 0.0], [-1.0, 0.0], [2.0, 0.0], [-2.0, 0.0]]
    tspan, train_x, val_x, test_x = get_dataset(seed=456789, timesteps=32, us=us, samples=128)

    train_x, times = arrange_data(train_x, tspan, num_points=8)
    val_x, times = arrange_data(val_x, tspan, num_points=8)
    test_x, times = arrange_data(test_x, tspan, num_points=8)

    # train_x, val_x, test_x have shape (batch, time=2, channels=8)
    # where channels consists of cos, sin, derivative, control for each of the two actuators
    # The data is essentially (feature, label) concatenated via the time axis: given the initial condition, predict the
    # next state.

    train_dataset = torch.utils.data.TensorDataset(train_x)
    val_dataset = torch.utils.data.TensorDataset(val_x)
    test_dataset = torch.utils.data.TensorDataset(test_x)

    train_dataloader = common.dataloader(train_dataset, batch_size)
    val_dataloader = common.dataloader(val_dataset, batch_size)
    test_dataloader = common.dataloader(test_dataset, batch_size)
    return times, train_dataloader, val_dataloader, test_dataloader
