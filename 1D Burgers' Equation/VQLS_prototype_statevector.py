import numpy as np
from qiskit import QuantumCircuit
from scipy.optimize import minimize
from qiskit.quantum_info import Statevector
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def build_A(N, nu, dt, h):
    alpha = nu * dt / (h**2)
    diag = np.ones(N) * (1 + 2*alpha)
    off = np.ones(N-1) * (-alpha)
    A = np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)
    return A

def advect_upwind(u, dt, h):
    N = len(u)
    ustar = u.copy()
    for i in range(1, N-1):
        dudx = (u[i] - u[i-1]) / h
        ustar[i] = u[i] - dt * u[i] * dudx
    ustar[0] = 0.0
    ustar[-1] = 0.0
    return ustar

def ansatz_circuit(n_qubits, params, reps=3):
    qc = QuantumCircuit(n_qubits)
    idx = 0
    for r in range(reps):
        for q in range(n_qubits):
            qc.ry(params[idx], q)
            idx += 1
        for q in range(n_qubits - 1):
            qc.cx(q, q+1)
    return qc

def get_statevector_from_circuit(qc):
    sv = Statevector.from_instruction(qc)
    return np.array(sv.data, dtype=complex)

def pad_to_power_of_two(vec):
    N = len(vec)
    target = 1
    while target < N:
        target *= 2
    if target == N:
        return vec
    padded = np.zeros(target, dtype=complex)
    padded[:N] = vec
    return padded

def amplitude_state_prep_circuit(vec):
    vec = np.array(vec, dtype=complex)
    n = int(np.log2(len(vec)))
    qc = QuantumCircuit(n)
    qc.initialize(vec.tolist(), list(range(n)))
    return qc

class VQLS_Prototype:
    def __init__(self, A, b_vec, n_qubits, reps=3):
        self.A = np.array(A, dtype=float)
        self.N = A.shape[0]
        self.n_qubits = n_qubits
        self.reps = reps
        b_norm = np.linalg.norm(b_vec)
        if b_norm == 0:
            raise ValueError("b is zero vector")
        self.b = np.array(b_vec, dtype=float)
        self.b_sv = self.b / b_norm
        self.b_sv_padded = pad_to_power_of_two(self.b_sv)
        self.b_prep_circ = amplitude_state_prep_circuit(self.b_sv_padded)
        self.b_norm = b_norm

    def cost(self, params):
        qc = ansatz_circuit(self.n_qubits, params, reps=self.reps)
        sv = get_statevector_from_circuit(qc)
        x_sv = sv[:self.N]
        Ax = self.A.dot(x_sv)
        diff = Ax - self.b_sv
        val = np.vdot(diff, diff).real
        return float(val)

    def reconstruct_solution(self, params):
        u_classical = np.linalg.solve(self.A, self.b)
        norm_u = np.linalg.norm(u_classical)
        qc = ansatz_circuit(self.n_qubits, params, reps=self.reps)
        sv = get_statevector_from_circuit(qc)
        x_sv = sv[:self.N]
        u_rec = x_sv * norm_u
        return u_rec.real, u_classical

# ------------------- Classical solver -------------------
def run_classical_solver(max_steps=500, tolerance=1e-3):
    N = 8
    nu = 0.01
    h = 1.0 / (N + 1)
    dt = 1e-3
    A = build_A(N, nu, dt, h)
    x_grid = np.linspace(1, N, N) * h
    u_n = np.sin(np.pi * x_grid)
    print("Classical solver: Initial u:", np.round(u_n, 5))

    for step in range(max_steps):
        u_star = advect_upwind(u_n.copy(), dt, h)
        u_new = np.linalg.solve(A, u_star)
        residual = np.linalg.norm(u_new - u_n)
        print(f"[Classical] Step {step+1}: Residual = {residual:.5e}")
        if residual < tolerance:
            print(f"[Classical] Converged at step {step+1}, residual={residual:.5e}")
            break
        u_n = u_new

    print("Classical Final solution:", np.round(u_n, 5))
    return u_n

# ------------------- VQLS solver -------------------
def run_vqls_solver(max_steps=500, tolerance=1e-3):
    N = 8
    n_qubits = int(np.log2(N)) 
    nu = 0.01
    h = 1.0 / (N + 1)
    dt = 1e-3
    A = build_A(N, nu, dt, h)
    x_grid = np.linspace(1, N, N) * h
    u_n = np.sin(np.pi * x_grid)
    print("VQLS solver: Initial u:", np.round(u_n, 5))

    n_params = 10 * n_qubits
    last_params = 0.1 * np.random.randn(n_params) 

    for step in range(max_steps):
        print(f"\n--- VQLS Step {step+1} ---")
        u_star = advect_upwind(u_n.copy(), dt, h)
        vqls = VQLS_Prototype(A=A, b_vec=u_star, n_qubits=n_qubits, reps=10)
        
        res = minimize(vqls.cost, 
                       last_params,
                       method='COBYLA', 
                       options={'maxiter': 2000})

        last_params = res.x

        print(f"Optimizer result: cost = {res.fun:.5e}, nfev = {res.nfev}")

        u_rec, u_classical = vqls.reconstruct_solution(res.x)
        residual = np.linalg.norm(u_rec - u_n)
        print(f"[VQLS] Step {step+1}: Residual = {residual:.5e}")

        if residual < tolerance:
            print(f"[VQLS] Converged at step {step+1}, residual={residual:.5e}")
            break

        u_n = u_rec

    print("\nVQLS Final solution:", np.round(u_n, 5))
    return u_n

def run_classical_with_history(max_steps=200):
    """Runs the classical solver and returns the history of solutions."""
    N = 8
    nu = 0.01
    h = 1.0 / (N + 1)
    dt = 1e-3
    A = build_A(N, nu, dt, h)
    x_grid = np.linspace(1, N, N) * h
    u_n = np.sin(np.pi * x_grid)
    
    history = [u_n.copy()]
    for step in range(max_steps):
        u_star = advect_upwind(u_n.copy(), dt, h)
        u_new = np.linalg.solve(A, u_star)
        u_n = u_new
        history.append(u_n.copy())
        
    print(f"Classical simulation finished. Captured {len(history)} frames.")
    return history, x_grid

def run_vqls_with_history(max_steps=200):
    """Runs the VQLS solver and returns the history of solutions."""
    N = 8
    n_qubits = int(np.log2(N)) 
    nu = 0.01
    h = 1.0 / (N + 1)
    dt = 1e-3
    A = build_A(N, nu, dt, h)
    x_grid = np.linspace(1, N, N) * h
    u_n = np.sin(np.pi * x_grid)
    
    n_params = 10 * n_qubits
    last_params = 0.1 * np.random.randn(n_params)
    
    history = [u_n.copy()]
    for step in range(max_steps):
        print(f"\n--- VQLS Animation Step {step+1}/{max_steps} ---")
        u_star = advect_upwind(u_n.copy(), dt, h)
        vqls = VQLS_Prototype(A=A, b_vec=u_star, n_qubits=n_qubits, reps=10)
        
        res = minimize(vqls.cost, last_params, method='COBYLA', options={'maxiter': 2000})
        last_params = res.x
        u_rec, _ = vqls.reconstruct_solution(res.x)
        u_n = u_rec
        history.append(u_n.copy())

    print(f"VQLS simulation finished. Captured {len(history)} frames.")
    return history, x_grid


if __name__ == "__main__":
    num_frames = 500
    classical_history, x_grid = run_classical_with_history(max_steps=num_frames)
    vqls_history, _ = run_vqls_with_history(max_steps=num_frames)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_ylim(-0.2, 1.2)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Position (x)")
    ax.set_ylabel("Velocity (u)")
    ax.grid(True, linestyle='--')

    line_classical, = ax.plot(x_grid, classical_history[0], 'bo', label="Classical Solver", markersize=8)
    line_vqls, = ax.plot(x_grid, vqls_history[0], 'r--', label="VQLS Solver")
    
    ax.legend()
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, va='top')

    def update(frame):
        """Updates the plot for each frame of the animation."""
        line_classical.set_ydata(classical_history[frame])
        line_vqls.set_ydata(vqls_history[frame])
        time_text.set_text(f'Time Step: {frame}/{num_frames}')
        return line_classical, line_vqls, time_text

    ani = animation.FuncAnimation(fig, update, frames=num_frames + 1,
                                  blit=True, interval=50)

    print("\nSaving animation... This will take some time for 500 frames.")
    ani.save('simulation_comparison_500_frames.gif', writer='pillow', fps=20)
    print("Animation saved as simulation_comparison_500_frames.gif")
    
    plt.show()