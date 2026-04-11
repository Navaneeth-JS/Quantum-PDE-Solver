# Quantum-PDE-Solver

This repository contains a collection of hybrid quantum-classical algorithms developed to solve Partial Differential Equations (PDEs), specifically targeting Computational Fluid Dynamics (CFD) problems.

The project is conducted under the academic guidance of **Prof. Nagabhushan Rao Vadlamani** at the **Indian Institute of Technology (IIT) Madras**.

## Project Focus and Overview

The primary objective of this project is to evaluate and implement near-term quantum algorithms (NISQ-friendly) for solving the governing equations of fluid mechanics. The development has progressed from foundational 1D benchmarks to an end-to-end 2D quantum-computational fluid dynamics (QCFD) pipeline.

### Core Implementation Modules

* **1D Burgers' Equation:** Foundational exploration of non-linear PDE solving techniques using quantum states.
* **2D Lid-Driven Cavity (NISQ-CFD):** A standard CFD benchmark implementation for incompressible Navier-Stokes equations. This module was forked and adapted from the framework established by **Song et al. (2024)** to serve as a comparative baseline for noisy quantum hardware performance.
* **2D Vorticity-Streamfunction Formulation:** Implementation of alternative fluid dynamics formulations to assess mapping efficiency and state preparation on quantum hardware.
* **FABLE + HRF Pipeline (Main Contribution):** The primary research focus of this repository. This module introduces a highly optimized pipeline for the 2D Pressure-Poisson Equation (PPE), integrating:
    * **FABLE (Fast Approximate BLock-Encoding):** For efficient, data-independent matrix encoding.
    * **VQLS (Variational Quantum Linear Solver):** For finding the pressure field solution via variational optimization.
    * **HRF (Hadamard Random Forest):** For an exponential reduction in readout overhead compared to traditional tomography.

## Technical Credentials

The implementations in this repository leverage quantum software frameworks and recent literature (2022–2025) in the field of quantum linear systems and readout optimization.

* **Lead Developer:** Navaneeth J.S., IIT Madras.
* **Principal Investigator:** Prof. Nagabhushan Rao Vadlamani, Dept. of Aerospace Engineering, IIT Madras.
* **Toolstack:** PennyLane, Qiskit, AerSimulator, and custom Hadamard Random Forest implementations.

## Repository Structure
* `/1D Burgers/`: Formulations for non-linear convection-diffusion.
* `/2D Navier Stokes/`: Adapted NISQ-CFD benchmarks based on the work of Song et al.
* `/2D Vorticity-Streamfunction Formulation/`: Hybrid quantum-classical solver for 2D Navier-Stokes equations using a 6-qubit VQLS Poisson solver.
* `/FABLE and HRF/`: Main project files and readout architecture.

## Key References

```bibtex
@article{song25,
  author = {Song, Z. and Deaton, R. and Gard, B. and Bryngelson, S. H.},
  title = {Incompressible {N}avier--{S}tokes solve on noisy quantum hardware via a hybrid quantum--classical scheme},
  journal = {Computers \& Fluids},
  pages = {106507},
  volume = {288},
  doi = {10.1016/j.compfluid.2024.106507},
  year = {2025}
}

@article{bravo2023vqls,
  title   = {Variational Quantum Linear Solver},
  author  = {Bravo-Prieto, Carlos and others},
  journal = {Quantum},
  volume  = {7},
  pages   = {1188},
  year    = {2023}
}

@article{camps2022fable,
  title   = {{FABLE}: Fast Approximate {BL}ock {E}ncodings of Sparse Matrices},
  author  = {Camps, Daan and Staab, Lin and Van Beeumen, Roel and Yang, Chao},
  journal = {arXiv preprint arXiv:2205.00081},
  year    = {2022}
}

@article{song2025reconstructing,
  author       = {Zhixin Song and Hang Ren and Melody Lee and Bryan Gard and Nicolas Renaud and Spencer H. Bryngelson},
  title        = {Hadamard {R}andom {F}orest: {R}econstructing real-valued quantum states with exponential reduction in measurement settings},
  year         = {2025},
  eprint       = {2505.06455},
  archivePrefix= {arXiv},
  primaryClass = {quant-ph}
}
