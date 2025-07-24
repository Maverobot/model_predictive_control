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

# Define a control input (constant acceleration)
u[:] = 0.5  # try also time-varying: u = np.sin(time)

# Simulate the system
for k in range(N - 1):
    # @: matrix multiplication
    # flatten(): convert B to a 1D array for multiplication
    x[:, k + 1] = A @ x[:, k] + B.flatten() * u[k]

# Plotting
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(time, x[0], label="Position")
plt.plot(time, x[1], label="Velocity")
plt.title("State Evolution")
plt.xlabel("Time [s]")
plt.ylabel("State")
plt.grid()
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(time, u, label="Control Input (Force)", color="red")
plt.xlabel("Time [s]")
plt.ylabel("u")
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()
