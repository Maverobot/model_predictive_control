import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np


def main():

    # Time and horizon
    T = 0.1
    N_sim = 60  # Total simulation steps
    Np = 10  # MPC prediction horizon

    # System dynamics (double integrator)
    A = np.array([[1, T], [0, 1]])
    B = np.array([[0.5 * T**2], [T]])

    n = A.shape[0]  # state dim
    m = B.shape[1]  # input dim

    # Cost matrices
    Q = np.diag([1.0, 0.1])  # penalize position and velocity
    R = np.diag([0.01])  # penalize control input

    # Constraints
    u_max = 1.0
    v_max = 2.0

    # Reference state
    x_ref = np.array([[10], [0]])  # target: position 10, velocity 0

    # Initialization
    x0 = np.array([[0], [0]])  # start at rest
    x_log = [x0.flatten()]
    u_log = []

    # MPC loop
    x_current = x0.copy()
    for t in range(N_sim):
        # MPC optimization problem
        x = cp.Variable((n, Np + 1))
        u = cp.Variable((m, Np))

        cost = 0
        constraints = [x[:, 0] == x_current.flatten()]

        for k in range(Np):
            cost += cp.quad_form(x[:, k] - x_ref.flatten(), Q)
            cost += cp.quad_form(u[:, k], R)

            constraints += [x[:, k + 1] == A @ x[:, k] + B @ u[:, k]]
            constraints += [cp.abs(u[:, k]) <= u_max]
            constraints += [cp.abs(x[1, k]) <= v_max]  # velocity constraint

        # Solve the problem
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver=cp.OSQP)  # efficient QP solver

        if prob.status != cp.OPTIMAL:
            raise Exception(f"MPC failed at step {t} with status: {prob.status}")

        # Apply first control input
        u_applied = u[:, 0].value
        x_next = A @ x_current + B @ u_applied.reshape(-1, 1)

        # Log
        u_log.append(u_applied.flatten())
        x_log.append(x_next.flatten())

        # Update current state
        x_current = x_next

    # Convert logs
    x_log = np.array(x_log)
    u_log = np.array(u_log)

    # Plot results
    time = np.linspace(0, N_sim * T, N_sim + 1)

    plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 1)
    plt.plot(time, x_log[:, 0], label="Position")
    plt.axhline(x_ref[0, 0], color="r", linestyle="--", label="Target Pos")
    plt.ylabel("Position")
    plt.legend()
    plt.grid()

    plt.subplot(3, 1, 2)
    plt.plot(time, x_log[:, 1], label="Velocity")
    plt.axhline(v_max, color="r", linestyle="--", label="Velocity Limit")
    plt.axhline(-v_max, color="r", linestyle="--")
    plt.ylabel("Velocity")
    plt.legend()
    plt.grid()

    plt.subplot(3, 1, 3)
    plt.plot(time[:-1], u_log[:, 0], label="Control (Accel)")
    plt.axhline(u_max, color="r", linestyle="--", label="Accel Limit")
    plt.axhline(-u_max, color="r", linestyle="--")
    plt.xlabel("Time [s]")
    plt.ylabel("Acceleration")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
