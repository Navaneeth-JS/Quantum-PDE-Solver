import numpy as np

__all__ = [
    "u_momentum",
    "v_momentum",
    "get_rhs",
    "get_coeff_mat",
    "pressure_correct",
    "update_velocity",
    "check_divergence_free"
]

def u_momentum(imax, jmax, dx, dy, rho, mu, u, v, p, velocity, alpha):
    """
    Solves the x-momentum equation to find the intermediate u-velocity (u_star).
    This accounts for convection, diffusion, and the current pressure gradient.
    """
    u_star = np.zeros((imax + 1, jmax))
    d_u = np.zeros((imax + 1, jmax))

    # Diffusion conductance (D = mu * Area / distance)
    De = mu * dy / dx  
    Dw = mu * dy / dx
    Dn = mu * dx / dy
    Ds = mu * dx / dy

    # Power-law scheme function to handle the Peclet number (ratio of convection to diffusion)
    def A(F, D):
        return max(0, (1 - 0.1 * abs(F / D))**5)

    # Compute u_star for interior nodes
    # Loop over the u-grid (staggered: u exists on vertical faces)
    for i in range(1, imax):
        for j in range(1, jmax - 1):
            # Calculate Mass Fluxes (F = rho * velocity * Area) at the faces of the u-control volume
            # Note the averaging required because v-velocities are on different grid locations
            Fe = 0.5 * rho * dy * (u[i + 1, j] + u[i, j])
            Fw = 0.5 * rho * dy * (u[i - 1, j] + u[i, j])
            Fn = 0.5 * rho * dx * (v[i, j + 1] + v[i - 1, j + 1])
            Fs = 0.5 * rho * dx * (v[i, j] + v[i - 1, j])

            # Calculate neighbor coefficients using the Hybrid/Power-law approach
            aE = De * A(Fe, De) + max(-Fe, 0)
            aW = Dw * A(Fw, Dw) + max(Fw, 0)
            aN = Dn * A(Fn, Dn) + max(-Fn, 0)
            aS = Ds * A(Fs, Ds) + max(Fs, 0)
            
            # Central coefficient (aP) includes the net mass flux out of the cell
            aP = aE + aW + aN + aS + (Fe - Fw) + (Fn - Fs)

            # Pressure force: (P_west - P_east) * Area
            pressure_term = (p[i - 1, j] - p[i, j]) * dy

            # Solve for intermediate velocity u* using under-relaxation (alpha)
            u_star[i, j] = alpha / aP * ((aE * u[i + 1, j] + aW * u[i - 1, j] + aN * u[i, j + 1] + aS * u[i, j - 1]) + pressure_term) + (1 - alpha) * u[i, j]

            # Store the velocity-pressure coupling coefficient d_u = Area / aP
            # This is used later to correct the velocity based on pressure changes
            d_u[i, j] = alpha * dy / aP 

    # Set d_u for top and bottom BCs (needed for the pressure correction matrix)
    for i in range(1, imax):
        j = 0  # bottom
        Fe = 0.5 * rho * dy * (u[i + 1, j] + u[i, j])
        Fw = 0.5 * rho * dy * (u[i - 1, j] + u[i, j])
        Fn = 0.5 * rho * dx * (v[i, j + 1] + v[i - 1, j + 1])
        Fs = 0
        aE = De * A(Fe, De) + max(-Fe, 0)
        aW = Dw * A(Fw, Dw) + max(Fw, 0)
        aN = Dn * A(Fn, Dn) + max(-Fn, 0)
        aS = 0
        aP = aE + aW + aN + aS + (Fe - Fw) + (Fn - Fs)
        d_u[i, j] = alpha * dy / aP

        j = jmax - 1  # top
        Fe = 0.5 * rho * dy * (u[i + 1, j] + u[i, j])
        Fw = 0.5 * rho * dy * (u[i - 1, j] + u[i, j])
        Fn = 0
        Fs = 0.5 * rho * dx * (v[i, j] + v[i - 1, j])
        aE = De * A(Fe, De) + max(-Fe, 0)
        aW = Dw * A(Fw, Dw) + max(Fw, 0)
        aN = 0
        aS = Ds * A(Fs, Ds) + max(Fs, 0)
        aP = aE + aW + aN + aS + (Fe - Fw) + (Fn - Fs)
        d_u[i, j] = alpha * dy / aP

    # Apply Boundary Conditions for u-velocity
    u_star[0, :jmax] = -u_star[1, :jmax]            # left wall (No-slip, ghost cell)
    u_star[imax, :jmax] = -u_star[imax - 1, :jmax]  # right wall (No-slip, ghost cell)
    u_star[:, 0] = 0.0                               # bottom wall (Stationary)
    u_star[:, jmax - 1] = velocity                  # top wall (Moving lid)

    return u_star, d_u

def v_momentum(imax, jmax, dx, dy, rho, mu, u, v, p, alpha):
    """
    Solves the y-momentum equation to find the intermediate v-velocity (v_star).
    Similar logic to u_momentum but staggered in the y-direction.
    """
    v_star = np.zeros((imax, jmax+1))
    d_v = np.zeros((imax, jmax+1))

    # Diffusion coefficients
    De = mu * dy / dx  
    Dw = mu * dy / dx
    Dn = mu * dx / dy
    Ds = mu * dx / dy

    A = lambda F, D: max(0, (1-0.1 * abs(F/D))**5)

    # Compute v_star for interior nodes
    for i in range(1, imax-1):
        for j in range(1, jmax):
            # Mass Fluxes for the v-control volume faces
            Fe = 0.5 * rho * dy * (u[i+1, j] + u[i+1, j-1])
            Fw = 0.5 * rho * dy * (u[i, j] + u[i, j-1])
            Fn = 0.5 * rho * dx * (v[i, j] + v[i, j+1])
            Fs = 0.5 * rho * dx * (v[i, j-1] + v[i, j])

            aE = De * A(Fe, De) + max(-Fe, 0)
            aW = Dw * A(Fw, Dw) + max(Fw, 0)
            aN = Dn * A(Fn, Dn) + max(-Fn, 0)
            aS = Ds * A(Fs, Ds) + max(Fs, 0)
            aP = aE + aW + aN + aS + (Fe - Fw) + (Fn - Fs)

            # Pressure force: (P_south - P_north) * Area
            pressure_term = (p[i, j-1] - p[i, j]) * dx

            v_star[i, j] = alpha / aP * (aE * v[i+1, j] + aW * v[i-1, j] + aN * v[i, j+1] + aS * v[i, j-1] + pressure_term) + (1-alpha) * v[i, j]

            # Store velocity-pressure coupling d_v
            d_v[i, j] = alpha * dx / aP  

    # Set d_v for left and right BCs
    for j in range(1, jmax):
        i = 0  # left BC
        Fe = 0.5 * rho * dy * (u[i+1, j] + u[i+1, j-1])
        Fw = 0
        Fn = 0.5 * rho * dx * (v[i, j] + v[i, j+1])
        Fs = 0.5 * rho * dx * (v[i, j-1] + v[i, j])
        aE = De * A(Fe, De) + max(-Fe, 0)
        aW = 0
        aN = Dn * A(Fn, Dn) + max(-Fn, 0)
        aS = Ds * A(Fs, Ds) + max(Fs, 0)
        aP = aE + aW + aN + aS + (Fe - Fw) + (Fn - Fs)
        d_v[i, j] = alpha * dx / aP

        i = imax - 1  # right BC
        Fe = 0
        Fw = 0.5 * rho * dy * (u[i, j] + u[i, j-1])
        Fn = 0.5 * rho * dx * (v[i, j] + v[i, j+1])
        Fs = 0.5 * rho * dx * (v[i, j-1] + v[i, j])
        aE = 0
        aW = Dw * A(Fw, Dw) + max(Fw, 0)
        aN = Dn * A(Fn, Dn) + max(-Fn, 0)
        aS = Ds * A(Fs, Ds) + max(Fs, 0)
        aP = aE + aW + aN + aS + (Fe - Fw) + (Fn - Fs)
        d_v[i, j] = alpha * dx / aP

    # Apply Boundary Conditions for v-velocity
    v_star[0, :] = 0.0  # left wall
    v_star[imax-1, :] = 0.0  # right wall
    v_star[:, 0] = -v_star[:, 1]  # bottom wall (No-slip ghost cell)
    v_star[:, jmax] = -v_star[:, jmax-1]  # top wall (No-slip ghost cell)

    return v_star, d_v

def get_rhs(imax, jmax, dx, dy, rho, u_star, v_star):
    """
    Calculates the source term (b) for the pressure correction equation.
    This represents the 'mass imbalance' of the u_star and v_star fields.
    If RHS = 0, the flow is perfectly divergence-free.
    """
    stride = jmax
    bp = np.zeros((jmax) * (imax))

    for j in range(jmax):
        for i in range(imax):
            position = i + j * stride
            # Net mass flux into the cell: (Mass In - Mass Out)
            bp[position] = rho * (u_star[i,j] * dy - u_star[i+1,j] * dy + v_star[i,j] * dx - v_star[i,j+1] * dx)

    # Pressure is relative, so we anchor one point (usually node 0) to zero
    bp[0] = 0

    return bp

def get_coeff_mat(imax, jmax, dx, dy, rho, d_u, d_v):
    """
    Constructs the Lapalcian-like matrix (Ap) for the pressure correction equation.
    This describes how a pressure change in one cell affects mass balance in neighbors.
    """
    N = imax * jmax
    stride = jmax
    Ap = np.zeros((N, N))

    for j in range(jmax):
        for i in range(imax):
            position = i + j * stride
            aE, aW, aN, aS = 0, 0, 0, 0

            # Reference node (anchor point for pressure)
            if i == 0 and j == 0:
                Ap[position, position] = 1
                continue

            # Matrix entries are based on rho * d_velocity * Area
            # This logic fills the matrix based on cell position (corners, boundaries, or interior)
            
            # Example for interior nodes:
            # West neighbor
            if i > 0:
                Ap[position, position-1] = -rho * d_u[i,j] * dy
                aW = -Ap[position, position-1]
            
            # East neighbor
            if i < imax - 1:
                Ap[position, position+1] = -rho * d_u[i+1,j] * dy
                aE = -Ap[position, position+1]

            # South neighbor
            if j > 0:
                Ap[position, position-stride] = -rho * d_v[i,j] * dx
                aS = -Ap[position, position-stride]

            # North neighbor
            if j < jmax - 1:
                Ap[position, position+stride] = -rho * d_v[i,j+1] * dx
                aN = -Ap[position, position+stride]

            # Diagonal element: sum of all neighbor coefficients
            Ap[position, position] = aE + aW + aN + aS

    return Ap

def pressure_correct(imax, jmax, rhsp, Ap, p, alpha):
    """
    Solves the system Ap * p' = rhsp to find the pressure correction p'.
    Then updates the pressure field.
    """
    pressure = np.copy(p)  
    p_prime = np.zeros((imax, jmax))  
    
    # Solve the linear system (Ax = b)
    p_prime_interior = np.linalg.solve(Ap, rhsp)

    # Map the 1D solution vector back to the 2D grid
    z = 0 
    for j in range(jmax):
        for i in range(imax):
            p_prime[i, j] = p_prime_interior[z]
            z += 1
            # Update pressure with under-relaxation
            pressure[i, j] = p[i, j] + alpha * p_prime[i, j]

    pressure[0, 0] = 0  # Re-enforce reference pressure

    return pressure, p_prime

def update_velocity(imax, jmax, u_star, v_star, p_prime, d_u, d_v, velocity):
    """
    Final step of the iteration: Use the pressure correction (p') to nudge
    u_star and v_star so that they satisfy the conservation of mass.
    """
    u = np.zeros((imax+1, jmax))
    v = np.zeros((imax, jmax+1))

    # Update interior nodes: u_new = u_star + d_u * (P'_west - P'_east)
    for i in range(1, imax):
        for j in range(1, jmax-1):
            u[i,j] = u_star[i,j] + d_u[i,j] * (p_prime[i-1,j] - p_prime[i,j])

    for i in range(1, imax-1):
        for j in range(1, jmax):
            v[i,j] = v_star[i,j] + d_v[i,j] * (p_prime[i,j-1] - p_prime[i,j])

    # Re-apply Boundary Conditions to the corrected velocities
    v[0,:] = 0.0          # left wall
    v[imax-1,:] = 0.0     # right wall
    v[:,0] = -v[:,1]      # bottom
    v[:,-1] = -v[:,-2]    # top

    u[0,:] = -u[1,:]      # left
    u[imax,:] = -u[imax-1,:] # right
    u[:,0] = 0.0          # bottom
    u[:,-1] = velocity    # top

    return u, v

def check_divergence_free(imax, jmax, dx, dy, u, v):
    """
    Diagnostic: Calculates the continuity error in each cell.
    Ideally, this should approach zero as the simulation converges.
    """
    div = np.zeros((imax, jmax))

    for i in range(imax):
        for j in range(jmax):
            # Divergence = du/dx + dv/dy
            div[i, j] = (1/dx) * (u[i, j] - u[i+1, j]) + (1/dy) * (v[i, j] - v[i, j+1])

    return div