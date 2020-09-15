# Adapted from https://github.com/d-biswa/Symplectic-ODENet/blob/master/analyze-fa-acrobot.ipynb

import gym
import gym.wrappers
import imageio
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import torch

import datasets

_out = str(pathlib.Path(__file__).resolve().parent / 'results/acrobot-visual')


def main(name, model, device):
    # name is some string to identify the saved outputs of this function
    # model should be a model as given by acrobot.main(...).model
    # device is a device for PyTorch, like 'cpu' or 0.

    model = model.func
    n_eval = 400

    env = datasets.AcrobotEnv()
    env = gym.wrappers.Monitor(env, _out, force=True)

    env.reset()
    env.env.state = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    obs = env.env._get_ob()
    y = torch.tensor([obs[0], obs[2], obs[1], obs[3], obs[4], obs[5], 0.0, 0.0],
                     device=device, dtype=torch.float32, requires_grad=True).unsqueeze(0)

    y_traj = [y]
    frames = [env.render(mode='rgb_array')]
    for i in range(n_eval - 1):
        cos_q_sin_q, q_dot, u = torch.split(y, [4, 2, 2], dim=1)
        V_q = model.V_net(cos_q_sin_q)
        dV = torch.autograd.grad(V_q, cos_q_sin_q)[0]
        dVdcos_q, dVdsin_q = torch.chunk(dV, 2, dim=1)
        cos_q, sin_q = torch.chunk(cos_q_sin_q, 2, dim=1)
        dV_q = - dVdcos_q * sin_q + dVdsin_q * cos_q  # (1, 2)
        g_q = model.g_net(cos_q_sin_q)  # (1, 2, 2)

        g_q_T = torch.transpose(g_q, 1, 2)
        inv_g_g_T = torch.inverse(torch.matmul(g_q, g_q_T))
        g_T_inv_g_g_T = torch.matmul(g_q_T, inv_g_g_T)

        energy_shaping = 2 * dV_q.T
        damping_injection = -1 * q_dot.T

        u = torch.matmul(g_T_inv_g_g_T, energy_shaping + damping_injection)
        u = u.squeeze().detach().cpu().numpy()

        obs, _, _, _ = env.step(u)
        y = torch.tensor([obs[0], obs[2], obs[1], obs[3], obs[4], obs[5], u[0], u[1]],
                         device=device, dtype=torch.float32, requires_grad=True).unsqueeze(0)
        y_traj.append(y)
        frames.append(env.render(mode='rgb_array'))

    env.close()
    imageio.mimsave(_out + '/' + name + '-motion.gif', frames, duration=0.02)

    y_traj = torch.cat(y_traj).detach().cpu().numpy()

    with plt.style.context("seaborn-white"):
        fig = plt.figure(figsize=(12, 1.3), dpi=600)
        plt.rcParams["axes.grid"] = False
        ax = plt.subplot(1, 10, 1)
        ax.imshow(frames[0])
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax = plt.subplot(1, 10, 2)
        ax.imshow(frames[20])
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax = plt.subplot(1, 10, 3)
        ax.imshow(frames[40])
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax = plt.subplot(1, 10, 4)
        ax.imshow(frames[60])
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax = plt.subplot(1, 10, 5)
        ax.imshow(frames[80])
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax = plt.subplot(1, 10, 6)
        ax.imshow(frames[100])
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax = plt.subplot(1, 10, 7)
        ax.imshow(frames[120])
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax = plt.subplot(1, 10, 8)
        ax.imshow(frames[140])
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax = plt.subplot(1, 10, 9)
        ax.imshow(frames[160])
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax = plt.subplot(1, 10, 10)
        ax.imshow(frames[180])
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)

        plt.tight_layout()
        fig.subplots_adjust(hspace=0, wspace=0)
        fig.savefig(_out + '/' + name + '-frames.png')

    t_eval = torch.linspace(0, (n_eval - 1) * 0.05, n_eval, device=device)

    fig = plt.figure(figsize=[12, 4], dpi=600)
    plt.subplot(2, 3, 1)
    plt.plot(t_eval.numpy(), -1 * np.ones_like(t_eval.numpy()), 'k--', linewidth=0.5)
    plt.plot(t_eval.numpy(), 0 * np.ones_like(t_eval.numpy()), 'k-', linewidth=0.5)
    plt.plot(t_eval.numpy(), y_traj[:, 0], 'b--', label=r"$\cos(q_1)$", linewidth=2)
    plt.plot(t_eval.numpy(), y_traj[:, 2], 'b-', label=r"$\sin(q_1)$", linewidth=2)
    plt.title("$q_1$", fontsize=14)
    plt.xlabel('t')
    plt.legend(fontsize=9)

    plt.subplot(2, 3, 4)
    plt.plot(t_eval.numpy(), 1 * np.ones_like(t_eval.numpy()), 'k--', linewidth=0.5)
    plt.plot(t_eval.numpy(), 0 * np.ones_like(t_eval.numpy()), 'k-', linewidth=0.5)
    plt.plot(t_eval.numpy(), y_traj[:, 1], 'b--', label=r"$\cos(q_2)$", linewidth=2)
    plt.plot(t_eval.numpy(), y_traj[:, 3], 'b-', label=r"$\sin(q_2)$", linewidth=2)
    plt.title("$q_2$", fontsize=14)
    plt.xlabel('t')
    plt.legend(fontsize=9)

    plt.subplot(2, 3, 2)
    plt.plot(t_eval.numpy(), 0 * np.ones_like(t_eval.numpy()), 'k-', linewidth=0.5)
    plt.plot(t_eval.numpy(), y_traj[:, 4], color='b', linewidth=2)
    plt.title("$\dot{q}_1$", fontsize=14)
    plt.xlabel('t')

    plt.subplot(2, 3, 5)
    plt.plot(t_eval.numpy(), 0 * np.ones_like(t_eval.numpy()), 'k-', linewidth=0.5)
    plt.plot(t_eval.numpy(), y_traj[:, 5], color='b', linewidth=2)
    plt.title("$\dot{q}_2$", fontsize=14)
    plt.xlabel('t')

    plt.subplot(2, 3, 3)
    plt.plot(t_eval.numpy(), 0 * np.ones_like(t_eval.numpy()), 'k-', linewidth=0.5)
    plt.plot(t_eval.numpy(), y_traj[:, 6], color='b', linewidth=2)
    plt.title("$u_1$", fontsize=14)
    plt.xlabel('t')

    plt.subplot(2, 3, 6)
    plt.plot(t_eval.numpy(), 0 * np.ones_like(t_eval.numpy()), 'k-', linewidth=0.5)
    plt.plot(t_eval.numpy(), y_traj[:, 7], color='b', linewidth=2)
    plt.title("$u_2$", fontsize=14)
    plt.xlabel('t')

    plt.tight_layout()
    fig.savefig(_out + '/' + name + '-graph.png')
