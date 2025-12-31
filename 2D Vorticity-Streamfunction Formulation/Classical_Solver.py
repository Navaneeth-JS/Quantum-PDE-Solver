import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ------------------ DOMAIN ------------------
Lx, Ly = 1.0, 1.0
Nx, Ny = 128, 128
dx, dy = Lx / Nx, Ly / Ny

x = np.linspace(0, Lx, Nx, endpoint=False)
y = np.linspace(0, Ly, Ny, endpoint=False)
X, Y = np.meshgrid(x, y, indexing="ij")

# ------------------ PHYSICAL PARAMETERS ------------------
U0 = 0.5          # uniform flow in x
nu = 1e-3         # viscosity
dt = 0.002
n_steps = 300

# ------------------ INITIAL VORTEX ------------------
cx, cy = 0.3, 0.5
sigma = 0.05

omega = np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))
psi = np.zeros_like(omega)

# ------------------ PERIODIC DERIVATIVES ------------------
def ddx(f):
    return (np.roll(f, -1, 0) - np.roll(f, 1, 0)) / (2 * dx)

def ddy(f):
    return (np.roll(f, -1, 1) - np.roll(f, 1, 1)) / (2 * dy)

def laplacian(f):
    return (
        (np.roll(f, -1, 0) - 2*f + np.roll(f, 1, 0)) / dx**2 +
        (np.roll(f, -1, 1) - 2*f + np.roll(f, 1, 1)) / dy**2
    )

# ------------------ POISSON SOLVER (JACOBI) ------------------
def solve_poisson(omega, iters=300):
    psi = np.zeros_like(omega)
    for _ in range(iters):
        psi = 0.25 * (
            np.roll(psi, 1, 0) + np.roll(psi, -1, 0) +
            np.roll(psi, 1, 1) + np.roll(psi, -1, 1) +
            dx * dy * omega
        )
    return psi

# ------------------ TIME STEP ------------------
def step_vorticity(omega):
    psi = solve_poisson(-omega)

    # velocity field
    u = ddy(psi) + U0
    v = -ddx(psi)

    # advection
    adv = u * ddx(omega) + v * ddy(omega)

    # diffusion
    diff = nu * laplacian(omega)

    return omega + dt * (-adv + diff), psi, u, v

# ------------------ ANIMATION SETUP ------------------
fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(
    omega,
    extent=[0, 1, 0, 1],
    origin="lower",
    cmap="RdBu"
)
plt.colorbar(im)
ax.set_title("2D Uniform Flow + Vortex")
ax.set_xlabel("x")
ax.set_ylabel("y")

# ------------------ UPDATE FUNCTION ------------------
def update(frame):
    global omega
    omega, psi, u, v = step_vorticity(omega)

    im.set_data(omega)
    vmax = np.max(np.abs(omega))
    im.set_clim(-vmax, vmax)

    ax.set_title(f"Time step {frame}")
    return im,

anim = FuncAnimation(fig, update, frames=n_steps, interval=40)

# ------------------ SAVE GIF ------------------
print("Saving animation...")
anim.save("uniform_flow_plus_vortex_2D.gif", writer="pillow", fps=20)
print("Saved as uniform_flow_plus_vortex_2D.gif")

plt.show()