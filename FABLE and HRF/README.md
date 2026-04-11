# Block-Encoded VQLS + HRF for 2D Lid-Driven Cavity

This module implements a highly scalable, NISQ-friendly hybrid quantum-classical pipeline to solve the 2D Pressure-Poisson Equation (PPE) for the Lid-Driven Cavity benchmark. By addressing the primary bottlenecks of traditional quantum linear solvers, i.e, matrix encoding, state preparation, and state readout, this architecture provides a robust framework for Computational Fluid Dynamics (CFD).

## Key Novelties and Architecture

This pipeline introduces a completely restructured approach compared to standard quantum solvers:

| Bottleneck | Traditional Quantum Approach | This Novel Approach | Reference |
|---|---|---|---|
| **Input (Matrix Encoding)** | Pauli string LCU (Linear Combination of Unitaries) | **FABLE Oracle** $\mathcal{O}(N^2)$ gates | Camps et al. (2022) |
| **Solver (State Preparation)** | Full State-Preparation (e.g., HHL) | **VQLS** with local cost functions | Bravo-Prieto et al. (2023) |
| **Output (State Readout)** | Full Quantum State Tomography $\mathcal{O}(3^n)$ circuits | **HRF Readout** $\mathcal{O}(n)$ circuits | Song et al. (2025) |

### 1. FABLE Block-Encoding
Unlike standard LCU-VQLS, which requires an exponentially scaling $\mathcal{O}(N^4)$ Pauli string decomposition for dense symmetric matrices, this pipeline utilizes PennyLane's `qml.FABLE`. This implements a strictly **data-independent**, hardware-native oracle using single-qubit $R_y$ gates interleaved with CNOTs. This efficiently circumvents the severe decomposition overhead of the Pauli LCU approach and allows for direct compilation on quantum hardware.

### 2. Variational Quantum Linear Solver (VQLS)
The continuous pressure field is variationally prepared as a quantum state utilizing an efficient ansatz and local cost functions to optimize parameters, avoiding the barren plateau problem inherent in deep global circuits.

### 3. Hadamard Random Forest (HRF) Readout
To avoid the exponential scaling of full Quantum State Tomography (QST), this architecture uses the HRF readout method. Exploiting the real-valued nature of the CFD solution, HRF estimates amplitudes in the standard Z-basis and employs a Random Forest classifier over $\mathcal{O}(n)$ Hadamard-rotated bases to recover vector signs, requiring only a linear number of measurement circuits.

## Classical CFD Benchmark Formulation

The physical benchmark relies on Chorin's fractional-step projection method for the incompressible Navier-Stokes equations, where incompressibility is enforced via:

$$\nabla^2 p = \frac{\rho}{\Delta t} \nabla \cdot \mathbf{u}^*$$

**Boundary Conditions (Lid-Driven Cavity):**
* **Top Wall ($y=1$):** $u = U_{\text{lid}} = 1$, $v = 0$ (moving lid)
* **All Other Walls:** $u = v = 0$ (no-slip)
* **Pressure:** Neumann boundary conditions on all walls, with one Dirichlet pin to strictly fix the gauge.

The 5-point finite-difference Laplacian is constructed from this system, normalized, and encoded directly into the quantum pipeline.

## Environment Setup and Installation

### Installation (run once)
```bash
pip install pennylane>=0.38 pennylane-lightning scipy numpy matplotlib pandas scikit-learn
# HRF library (local fork):
cd ~/Quantum-HRF-Tomography && pip install -e .
# OR from GitHub:
pip install git+[https://github.com/comp-physics/Quantum-HRF-Tomography](https://github.com/comp-physics/Quantum-HRF-Tomography)
