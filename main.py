import cvxpy as cp
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import cont2discrete

from continuous_cartpole import \
    ContinuousCartPoleEnv  # assuming your env is in a .py file


def main():

    # Environment
    env = ContinuousCartPoleEnv()
    obs = env.reset()

    # Linearized dynamics
    g = 9.8
    m = 0.1
    M = 1.0
    l = 0.5
    total_mass = M + m

    A = np.array(
        [
            [0, 1, 0, 0],
            [0, 0, -(m * g) / M, 0],
            [0, 0, 0, 1],
            [0, 0, (total_mass * g) / (l * M), 0],
        ]
    )
    B = np.array([[0], [1 / M], [0], [-1 / (l * M)]])

    # Discretize
    T = 0.02
    Ad, Bd, _, _, _ = cont2discrete((A, B, np.eye(4), np.zeros((4, 1))), T)

    # MPC setup
    N = 20  # horizon
    n, m = Ad.shape[0], Bd.shape[1]

    Q = np.diag([20, 1, 100, 1])
    R = np.diag([0.01])
    x_ref = np.zeros((n, 1))

    u_max = 0.9  # matches env range

    # MPC variables
    x = cp.Variable((n, N + 1))
    u = cp.Variable((m, N))

    def mpc_control(x0):
        cost = 0
        constraints = [x[:, 0] == x0.flatten()]
        for t in range(N):
            cost += cp.quad_form(x[:, t] - x_ref.flatten(), Q)
            cost += cp.quad_form(u[:, t], R)
            constraints += [x[:, t + 1] == Ad @ x[:, t] + Bd @ u[:, t]]
            constraints += [cp.abs(u[:, t]) <= u_max]
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver=cp.OSQP)
        return u[:, 0].value if prob.status == cp.OPTIMAL else np.array([0.0])

    # Simulation
    steps = 300
    states = []
    actions = []

    obs = env.reset()
    for i in range(steps):
        env.render("human")
        x0 = np.array(obs).reshape(-1, 1)
        u_cmd = np.float32(mpc_control(x0))
        obs, reward, done, _ = env.step(u_cmd)
        states.append(obs)
        actions.append(u_cmd)
        if done:
            break

    env.close()


if __name__ == "__main__":
    main()
