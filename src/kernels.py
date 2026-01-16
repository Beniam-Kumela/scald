from numba import njit 
import numpy as np
from .constants import *

@njit
def compute_macroscopic(f, g, nx, ny, nl, solid, density):
    density = np.zeros((nx, ny))

    v_x = np.zeros((nx, ny))
    v_y = np.zeros((nx, ny))

    if g is not None:
        T = np.zeros((nx, ny))
    else:
        T = None
    
    for i in range(nx):
        for j in range(ny):
            for k in range(nl):
                density[i, j] += f[i, j, k]
                v_x[i, j] += c_x[k] * f[i, j, k]
                v_y[i, j] += c_y[k] * f[i, j, k]

                if g is not None:
                    T[i, j] += g[i, j, k]
            
            if density[i, j] > 0: # avoid division by zero
                v_x[i, j] /= density[i, j]
                v_y[i, j] /= density[i, j]
            
            if solid[i, j]:
                v_x[i, j] = 0.0
                v_y[i, j] = 0.0

    return density, T, v_x, v_y

@njit
def psi(density: np.ndarray):
    return 1 - np.exp(-density)

@njit
def compute_shan_chen_force(density, nx, ny, nl, G):
    Fx = np.zeros_like(density)
    Fy = np.zeros_like(density)

    for i in range(nx):
        for j in range(ny):
            center_psi = psi(density[i, j])
            fx = 0.0
            fy = 0.0

            for k in range(1, nl):
                
                x_new = (i + c_x[k]) % nx
                y_new = j + c_y[k]

                if 0 <= y_new < ny:
                    neighbor_psi = psi(density[x_new, y_new])
                    fx += w[k] * neighbor_psi * c_x[k]
                    fy += w[k] * neighbor_psi * c_y[k]
                
            Fx[i, j] = -G * center_psi * fx
            Fy[i, j] = -G * center_psi * fy
    return Fx, Fy

@njit
def collision(density, T, v_x, v_y, f, g, nx, ny, nl, tau_f, tau_g, alpha, T_ref, gravity, G):
    f_new = np.zeros_like(f)

    if g is not None:
        g_new = np.zeros_like(g)
        buoyancy_force = alpha * (T - T_ref) * gravity
    
    else:
        g_new = None
        buoyancy_force = np.zeros((nx, ny))
    
    if G is not None:
        Fx, Fy = compute_shan_chen_force(density, nx, ny, nl, G)
        Fy += buoyancy_force - gravity * density
        v_x = v_x + (0.5 * Fx) / density
        v_y = v_y + (0.5 * Fy) / density

    else:
        Fx = np.zeros((nx, ny))
        Fy = buoyancy_force
    
    for k in range(nl):
        cdotv = c_x[k] * v_x + c_y[k] * v_y

        f_eq = w[k] * density * (1 + cdotv/cs2 + (cdotv ** 2) / (2 * cs2 ** 2) - (v_x ** 2 + v_y ** 2) / (2 * cs2))
        collision_term = -(f[:, :, k] - f_eq) / tau_f

        # if g is not None: Fx = 0, Fy = 0, source_term = 0
        source_term = (1 - 0.5 / tau_f) * w[k] * (
        ((c_x[k] - v_x) * Fx + (c_y[k] - v_y) * Fy) / cs2 + 
        (cdotv * (c_x[k] * Fx + c_y[k] * Fy) / cs2 ** 2))
 
        
        f_new[:, :, k] = f[:, :, k] + collision_term + source_term

        if g is not None:
            g_eq = w[k] * T * (1 + cdotv/cs2 + (cdotv ** 2) / (2 * cs2 ** 2) - (v_x ** 2 + v_y ** 2) / (2 * cs2))
            g_new[:, :, k] = g[:, :, k] - (g[:, :, k] - g_eq) / tau_g
            
    return f_new, g_new

@njit
def streaming(f, g, nx, ny, nl, periodic, density, multiphase):
    f_new = np.zeros_like(f)

    if g is not None:
        g_new = np.zeros_like(g)
    else:
        g_new = None

    for i in range(nx):
        for j in range(ny):
            for k in range(nl):
                next_i = i + c_x[k]
                next_j = j + c_y[k]

                if periodic == "x":
                    next_i = next_i % nx

                    if multiphase == "True":
                        if next_j < 0 or next_j >= ny:
                            continue
                
                elif periodic == "y":
                    next_j = next_j % ny

                    if next_i < 0 or next_i >= nx:
                        continue
                
                elif periodic == "xy":
                    next_i = next_i % nx
                    next_j = next_j % ny
                
                # check if next cell is out-of-bounds (non-periodic) or solid
                if (next_i < 0 or next_i >= nx or next_j < 0 or next_j >= ny):
                    # bounce-back
                    f_new[i, j, opp_dir[k]] = f[i, j, k]
                    if g is not None:
                        g_new[i, j, opp_dir[k]] = g[i, j, k]
                else:
                    # normal streaming
                    f_new[next_i, next_j, k] = f[i, j, k]
                    if g is not None:
                        g_new[next_i, next_j, k] = g[i, j, k]
    
    if density is not None:
        density[:, :] = 0
        for k in range(nl):
            density += f_new[:, :, k]

    return f_new, g_new, density