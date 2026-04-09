import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

Lx, Ly = 1.0, 1.0
Nx, Ny = 128, 128
dx, dy = Lx / Nx, Ly / Ny

x = np.linspace(0, Lx, Nx, endpoint=False)
y = np.linspace(0, Ly, Ny, endpoint=False)
X, Y = np.meshgrid(x, y, indexing="ij")

U0 = 0.5          
nu = 2e-4         
dt = 0.002
n_steps = 600

cx, cy = 0.2, 0.5
sigma = 0.05
omega = 12.0 * np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))
current_psi = np.zeros_like(omega)

def ddx(f):
    return (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (2 * dx)

def ddy(f):
    return (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2 * dy)

def laplacian(f):
    return (np.roll(f, -1, 0) - 2*f + np.roll(f, 1, 0)) / dx**2 + \
           (np.roll(f, -1, 1) - 2*f + np.roll(f, 1, 1)) / dy**2

def solve_poisson(omega, psi_prev, iters=300): # Increased Jacobi iters
    psi = psi_prev.copy()
    source = (dx**2) * omega 
    for _ in range(iters):
        psi = 0.25 * (
            np.roll(psi, 1, 0) + np.roll(psi, -1, 0) +
            np.roll(psi, 1, 1) + np.roll(psi, -1, 1) +
            source
        )
    return psi

current_psi = solve_poisson(omega, current_psi, iters=1000)

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(omega.T, extent=[0, Lx, 0, Ly], origin="lower", cmap="RdBu_r", alpha=0.6)

total_psi = current_psi + (U0 * Y)
cont = ax.contour(X.T, Y.T, total_psi.T, levels=25, colors='black', linewidths=0.8)

ax.set_title("Extended Run: Vortex + Uniform Flow")
plt.colorbar(im, label="Vorticity")

def update(frame):
    global omega, current_psi, cont
    
    current_psi = solve_poisson(omega, current_psi)
    u = ddy(current_psi) + U0
    v = -ddx(current_psi)
    adv = u * ddx(omega) + v * ddy(omega)
    diff = nu * laplacian(omega)
    omega = omega + dt * (-adv + diff)
    
    im.set_data(omega.T)
    im.set_clim(-10, 10)
    
    for c in cont.collections:
        c.remove()
    total_psi = current_psi + (U0 * Y)
    cont = ax.contour(X.T, Y.T, total_psi.T, levels=25, colors='black', linewidths=0.8)
    
    return [im] + cont.collections

anim = FuncAnimation(fig, update, frames=n_steps, interval=30, blit=False)
anim.save("long_vortex_flow.gif", writer="pillow", fps=30)
plt.show()