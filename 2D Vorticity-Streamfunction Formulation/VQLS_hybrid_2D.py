import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.optimize import minimize
import time

from qiskit.circuit.library import EfficientSU2, UnitaryGate
from qiskit.quantum_info import Statevector, Operator, SparsePauliOp
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit_aer.noise import NoiseModel, depolarizing_error

Nx, Ny = 4, 4
Lx, Ly = 1.0, 1.0
dx, dy = Lx / Nx, Ly / Ny
N = Nx * Ny
n_qubits = int(np.log2(N))

x = np.linspace(0, Lx, Nx, endpoint=False)
y = np.linspace(0, Ly, Ny, endpoint=False)
X, Y = np.meshgrid(x, y, indexing="ij")
last_params = None

U0 = 0.5
nu = 1e-3
dt = 0.01
n_steps = 80

cx, cy = 0.3, 0.5
sigma = 0.08
omega = np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))

def flatten(f):
    return f.reshape(-1)

def unflatten(v):
    return v.reshape((Nx, Ny))

def ddx(f):
    return (np.roll(f, -1, 0) - np.roll(f, 1, 0)) / (2 * dx)

def ddy(f):
    return (np.roll(f, -1, 1) - np.roll(f, 1, 1)) / (2 * dy)

def laplacian(f):
    return (
        (np.roll(f, -1, 0) - 2*f + np.roll(f, 1, 0)) / dx**2 +
        (np.roll(f, -1, 1) - 2*f + np.roll(f, 1, 1)) / dy**2
    )

def build_laplacian_2d():
    A = np.zeros((N, N))

    def idx(i, j):
        return i * Ny + j

    for i in range(Nx):
        for j in range(Ny):
            p = idx(i, j)
            A[p, p] = -2/dx**2 - 2/dy**2
            A[p, idx((i+1)%Nx, j)] = 1/dx**2
            A[p, idx((i-1)%Nx, j)] = 1/dx**2
            A[p, idx(i, (j+1)%Ny)] = 1/dy**2
            A[p, idx(i, (j-1)%Ny)] = 1/dy**2
    return A

A_lap = build_laplacian_2d()

def pad(vec):
    out = np.zeros(2**n_qubits)
    out[:len(vec)] = vec
    return out

def unitary_from_state(vec):
    vec = vec / np.linalg.norm(vec)
    dim = len(vec)
    M = np.random.randn(dim, dim) + 1j*np.random.randn(dim, dim)
    M[:,0] = vec
    Q,_ = np.linalg.qr(M)
    Q[:,0] = vec
    return Q

class VQLS:
    def __init__(self, A, b):
        self.A = SparsePauliOp.from_operator(Operator(A))
        self.b = pad(b)
        self.nq = n_qubits

        U = unitary_from_state(self.b)
        self.Ub = UnitaryGate(U)

        self.ansatz = EfficientSU2(self.nq, reps=3, entanglement="circular")
        self.params = ParameterVector("θ", self.ansatz.num_parameters)

        self.ansatz = self.ansatz.assign_parameters(
            {self.ansatz.parameters[i]: self.params[i]
             for i in range(len(self.params))}
        )

        self.AdagA = (self.A.conjugate() @ self.A).simplify()

    def cost(self, p, estimator):
        job = estimator.run([self.ansatz.assign_parameters(p)],
                            [self.AdagA])
        return job.result().values[0]

    def solve(self, estimator, initial_params=None):
        if initial_params is None:
            p0 = 0.1 * np.random.randn(len(self.params))
        else:
            p0 = initial_params
        res = minimize(lambda p: self.cost(p, estimator),
                       p0, method="COBYLA", options={"maxiter":200})
        sv = Statevector(self.ansatz.assign_parameters(res.x)).data
        return sv[:N].real, res.x
noise = NoiseModel()
noise.add_all_qubit_quantum_error(depolarizing_error(0.002,1), ["sx","rz"])

estimator = AerEstimator(
    backend_options={
        "method": "density_matrix",
        "noise_model": noise
    }
)

def step(omega):
    psi_vec, last_params = vqls.solve(estimator, last_params)
    b = -flatten(omega)
    vqls = VQLS(A_lap, b)
    psi = unflatten(vqls.solve(estimator))

    u = ddy(psi) + U0
    v = -ddx(psi)

    adv = u * ddx(omega) + v * ddy(omega)
    diff = nu * laplacian(omega)

    return omega + dt * (-adv + diff)

fig, ax = plt.subplots()
im = ax.imshow(omega, origin="lower", cmap="RdBu")
plt.colorbar(im)

def update(frame):
    global omega, last_frame_time

    omega = step(omega)
    im.set_data(omega)
    im.set_clim(-np.max(np.abs(omega)), np.max(np.abs(omega)))
    ax.set_title(f"Step {frame+1}/{n_steps}")

    now = time.time()
    step_time = now - last_frame_time
    last_frame_time = now

    elapsed = now - start_time
    remaining = (n_steps - frame - 1) * step_time

    sys.stdout.write(
        f"\rFrame {frame+1}/{n_steps} | "
        f"Step: {step_time:.2f}s | "
        f"Elapsed: {elapsed/60:.1f} min | "
        f"ETA: {remaining/60:.1f} min"
    )
    sys.stdout.flush()

    return im,

start_time = time.time()
last_frame_time = start_time


ani = animation.FuncAnimation(fig, update, frames=n_steps, interval=120)

print("Saving GIF...")
ani.save("hybrid_2D_uniform_flow_vortex.gif", writer="pillow", fps=8)
print("Saved as hybrid_2D_uniform_flow_vortex.gif")
print("\nDone.")

plt.show()