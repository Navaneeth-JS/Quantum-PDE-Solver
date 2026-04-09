import numpy as np
import sys
import time
import warnings
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.optimize import minimize

warnings.filterwarnings("ignore", category=DeprecationWarning)

from qiskit.circuit.library import EfficientSU2
from qiskit.quantum_info import Statevector, Operator, SparsePauliOp
from qiskit_aer.primitives import Estimator as AerEstimator

Nx, Ny = 8, 8  
Lx, Ly = 1.0, 1.0
dx, dy = Lx / Nx, Ly / Ny
N = Nx * Ny
n_qubits = int(np.log2(N))

x = np.linspace(0, Lx, Nx, endpoint=False)
y = np.linspace(0, Ly, Ny, endpoint=False)
X, Y = np.meshgrid(x, y, indexing="ij")

U0 = 0.5       
nu = 2e-3      
dt = 0.01      
n_steps = 60   

cx, cy = 0.3, 0.5
sigma = 0.1
omega = 5.0 * np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))

def flatten(f): return f.reshape(-1)
def unflatten(v): return v.reshape((Nx, Ny))

def ddx(f): return (np.roll(f, -1, 0) - np.roll(f, 1, 0)) / (2 * dx)
def ddy(f): return (np.roll(f, -1, 1) - np.roll(f, 1, 1)) / (2 * dy)
def laplacian(f):
    return ((np.roll(f, -1, 0) - 2*f + np.roll(f, 1, 0)) / dx**2 +
            (np.roll(f, -1, 1) - 2*f + np.roll(f, 1, 1)) / dy**2)

def poisson_jacobi(omega, iters=800):
    psi = np.zeros_like(omega)
    source = dx**2 * omega 
    for _ in range(iters):
        psi = 0.25 * (np.roll(psi, 1, 0) + np.roll(psi, -1, 0) +
                      np.roll(psi, 1, 1) + np.roll(psi, -1, 1) + source)
    return psi

def build_laplacian_matrix():
    A = np.zeros((N, N))
    for i in range(Nx):
        for j in range(Ny):
            p = i * Ny + j
            A[p, p] = -2/dx**2 - 2/dy**2
            A[p, ((i+1)%Nx)*Ny + j] = 1/dx**2
            A[p, ((i-1)%Nx)*Ny + j] = 1/dx**2
            A[p, i*Ny + (j+1)%Ny] = 1/dy**2
            A[p, i*Ny + (j-1)%Ny] = 1/dy**2
    
    epsilon = 1e-3
    A += epsilon * np.eye(N)
    
    scaling = np.max(np.abs(A))
    return A / scaling, scaling

A_scaled, matrix_scaling = build_laplacian_matrix()

class CorrectedVQLS:
    def __init__(self, A):
        self.A_mat = A
        self.A_op = SparsePauliOp.from_operator(Operator(A))
        self.AdagA = (self.A_op.conjugate() @ self.A_op).simplify()
        # Reps=4 provides the depth needed to resolve the Laplacian inverse
        self.ansatz = EfficientSU2(n_qubits, reps=4, entanglement="circular")
        self.last_params = None

    def cost_func(self, params, b_unit, estimator):
        circ = self.ansatz.assign_parameters(params)
        quad = estimator.run([circ], [self.AdagA]).result().values[0]
        sv = Statevector(circ).data
        linear = np.real(np.vdot(b_unit, (self.A_mat @ sv)))
        return quad - 2 * linear

    def solve(self, b, estimator):
        b_norm = np.linalg.norm(b)
        if b_norm < 1e-10: return np.zeros(N)
        b_unit = b / b_norm
        
        p0 = self.last_params if self.last_params is not None else 0.05 * np.random.randn(self.ansatz.num_parameters)
        
        res = minimize(self.cost_func, p0, args=(b_unit, estimator), 
                       method="COBYLA", options={"maxiter": 200})
        
        self.last_params = res.x
        final_sv_vec = Statevector(self.ansatz.assign_parameters(res.x)).data.real[:N]
        
        numerator = np.vdot(final_sv_vec, self.A_mat.T @ b)
        denominator = np.vdot(final_sv_vec, self.A_mat.T @ self.A_mat @ final_sv_vec)
        alpha = np.real(numerator / (denominator + 1e-12))
        
        return final_sv_vec * alpha

estimator = AerEstimator()
vqls = CorrectedVQLS(A_scaled)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
im_c = axes[0].imshow(np.zeros((Nx, Ny)), cmap="viridis", origin="lower", extent=[0,1,0,1])
im_q = axes[1].imshow(np.zeros((Nx, Ny)), cmap="viridis", origin="lower", extent=[0,1,0,1])
im_e = axes[2].imshow(np.zeros((Nx, Ny)), cmap="magma", origin="lower", extent=[0,1,0,1])

axes[0].set_title(r"Classical Reference ($\psi_c$)")
axes[1].set_title(r"Corrected Quantum ($\psi_q$)")
axes[2].set_title(r"Residual Error ($|\psi_c - \psi_q|$)")

plt.colorbar(im_c, ax=axes[0])
plt.colorbar(im_q, ax=axes[1])
plt.colorbar(im_e, ax=axes[2])

start_time = time.time()
frame_times = []

def update(frame):
    global omega
    t_start = time.time()

    psi_c = poisson_jacobi(omega)
    psi_c -= np.mean(psi_c)

    b = -flatten(omega) * dx**2
    b -= np.mean(b)
    
    psi_q_vec = vqls.solve(b, estimator)
    psi_q = unflatten(psi_q_vec) / matrix_scaling
    psi_q -= np.mean(psi_q)
    
    error = np.abs(psi_c - psi_q)
    u, v = ddy(psi_q) + U0, -ddx(psi_q)
    adv = u * ddx(omega) + v * ddy(omega)
    diff = nu * laplacian(omega)
    omega = omega + dt * (-adv + diff)

    im_c.set_data(psi_c.T); im_q.set_data(psi_q.T); im_e.set_data(error.T)
    vmax = np.max(np.abs(psi_c))
    im_c.set_clim(-vmax, vmax); im_q.set_clim(-vmax, vmax); im_e.set_clim(0, vmax * 0.2)

    frame_times.append(time.time() - t_start)
    avg_t = np.mean(frame_times)
    etc = str(timedelta(seconds=int((n_steps - (frame+1)) * avg_t)))
    sys.stdout.write(f"\rFrame {frame+1}/{n_steps} | Avg: {avg_t:.2f}s | ETC: {etc} ")
    sys.stdout.flush()
    
    return im_c, im_q, im_e

ani = animation.FuncAnimation(fig, update, frames=n_steps, interval=50, blit=True)

print(f"Starting CORRECTED Hybrid Solver (6 qubits)...")
ani.save("corrected_quantum_vortex.gif", writer="pillow", fps=10)
print(f"\nCompleted. Total Time: {str(timedelta(seconds=int(time.time() - start_time)))}")
plt.show()