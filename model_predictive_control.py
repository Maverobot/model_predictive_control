import matplotlib.pyplot as plt
import numpy as np

# Time settings
T = 0.1  # Time step
N = 100  # Number of steps
time = np.linspace(0, N * T, N)

# System dynamics (double integrator)
A = np.array([[1, T], [0, 1]])
B = np.array([[0.5 * T**2], [T]])

# Initialize state and control
x = np.zeros((2, N))  # state: [position; velocity]
u = np.zeros(N)  # control input

# Constraints
u_max = 1.0
v_max = 2.0

# Define time-varying control input
raw_u = 2 * np.sin(0.2 * time)  # unbounded signal

# Simulate the system
for k in range(N - 1):
    # Apply input constraint
    u[k] = np.clip(raw_u[k], -u_max, u_max)

    # System update
    # @: matrix multiplication
    # flatten(): convert B to a 1D array for multiplication
    x[:, k + 1] = A @ x[:, k] + B.flatten() * u[k]

    # Optionally apply a velocity constraint (clip after dynamics)
    if abs(x[1, k + 1]) > v_max:
        x[1, k + 1] = np.sign(x[1, k + 1]) * v_max
        # Re-adjust position so it matches the constrained velocity (approx)
        x[0, k + 1] = x[0, k] + T * x[1, k + 1]

# Plotting
plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
plt.plot(time, x[0], label="Position")
plt.ylabel("Position")
plt.grid()
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(time, x[1], label="Velocity")
plt.axhline(v_max, color="r", linestyle="--", label="Velocity Limit")
plt.axhline(-v_max, color="r", linestyle="--")
plt.ylabel("Velocity")
plt.grid()
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(time, raw_u, "--", label="Raw Input")
plt.plot(time, u, label="Clipped Input")
plt.axhline(u_max, color="r", linestyle="--", label="Input Limit")
plt.axhline(-u_max, color="r", linestyle="--")
plt.xlabel("Time [s]")
plt.ylabel("Control Input")
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()
