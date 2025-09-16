# Hybrid Burgers VQLS prototype (Qiskit + statevector simulator)
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from scipy.optimize import minimize
from qiskit.quantum_info import Statevector

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

def ansatz_circuit(n_qubits, params, reps=2):
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
    def __init__(self, A, b_vec, n_qubits, reps=2):
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

# Example
def run_demo_one_timestep():
    N = 8
    n_qubits = int(np.log2(N))
    nu = 0.01
    h = 1.0 / (N + 1)
    dt = 1e-3

    A = build_A(N, nu, dt, h)
    x_grid = np.linspace(1, N, N) * h
    u_n = np.sin(np.pi * x_grid)
    u_star = advect_upwind(u_n.copy(), dt, h)

    print("N:", N, "n_qubits:", n_qubits)
    print("u_star (first 6):", np.round(u_star[:6], 5))

    vqls = VQLS_Prototype(A=A, b_vec=u_star, n_qubits=n_qubits, reps=2)

    n_params = 2 * n_qubits
    x0 = 0.1 * np.random.randn(n_params)

    print("Starting variational optimization (simulator-based cost evaluation)...")
    res = minimize(lambda p: vqls.cost(p), x0, method='COBYLA', options={'maxiter': 120, 'tol':1e-6})

    print("Optimization success:", res.success, "message:", res.message)
    print("Final cost:", res.fun)

    u_rec, u_classical = vqls.reconstruct_solution(res.x)
    print("\nComparison (classical vs reconstructed) — first 8 entries:")
    for i in range(N):
        print(f"{i:2d}: classical={u_classical[i]: .6f}  reconstructed={u_rec[i]: .6f}")

    residual_quantum = np.linalg.norm(A.dot(u_rec) - u_star)
    residual_classic = np.linalg.norm(A.dot(u_classical) - u_star)
    print("\nResiduals: classical: {:.3e}, reconstructed: {:.3e}".format(residual_classic, residual_quantum))

if __name__ == "__main__":
    run_demo_one_timestep()