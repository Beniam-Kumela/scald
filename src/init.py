import numpy as np
from .constants import *

def rayleigh_bernard(nx, ny, nl, T_hot, T_cold):
    f = np.zeros((nx, ny, nl), dtype=np.float64)
    g = np.zeros_like(f)
    noise = 0.005 * np.random.randn(nx, ny)
    num_cells = 4
    k = 1

    for i in range(nx):
        for j in range(ny):
            # Linear temperature gradient where y = 0, T = T_hot and y = 1, T = T_cold
            T_init = T_hot + (T_cold - T_hot) * (j/ny)

            # # Sigmoid temperature gradient where y = 0, T = T_hot and y = 1, T = T_cold
            # T_init = (T_cold - T_hot) / (1 + np.exp(-k*(j/ny - ny // 2))) + T_cold

            # Sine wave with num_cells peaks across the simulation width (x-direction)
            T_init += 0.005 * np.sin(num_cells * 2 * np.pi * i / nx)
            T_init += noise[i, j] # break symmetry with noise

            for k in range(nl):
                f[i, j, k] = w[k]
                g[i, j, k] = w[k] * T_init
    
    return f, g

def thermal_sim(nx, ny, nl, T_cold):
    f = np.zeros((nx, ny, nl), dtype=np.float64)
    g = np.zeros_like(f)

    for i in range(nx):
        for j in range(ny):
            for k in range(nl):
                f[i, j, k] = w[k]
                g[i, j, k] = w[k] * T_cold
    
    return f, g

def thermal_bubble(f, g, nx, ny, nl, bubble_center_x, bubble_center_y, radius, T_hot):
    for i in range(nx):
        for j in range(ny):
            if ((i - bubble_center_x) ** 2
                    + (j - bubble_center_y) ** 2) < (radius + 1) ** 2:
                for k in range(nl):
                    g[i, j, k] = w[k] * T_hot

    return f, g

def thermal_obstacle_flow(f, g, n_obstacles, obstacle_centers_x, obstacle_centers_y, length, nx, ny, nl, obstacle, T_hot):
    for b in range(n_obstacles):
        for i in range(nx):
            for j in range(ny):
                if obstacle == "cylinder":
                    if ((i - obstacle_centers_x[b]) ** 2
                                + (j - obstacle_centers_y[b]) ** 2) < (length + 1) ** 2:
                        for k in range(nl):
                            g[i, j, k] = w[k] * T_hot
                elif obstacle == "box":
                    if (np.abs((i - obstacle_centers_x[b]) + (j - obstacle_centers_y[b])) +
                                np.abs((i - obstacle_centers_x[b]) - (j - obstacle_centers_y[b])) < length):
                        for k in range(nl):
                            g[i, j, k] = w[k] * T_hot
                else:
                    print("Obstacle can only be 'cylinder' or 'box'.")

    return f, g

def droplet_collision(nx, ny, density, liquid_density, gas_density, droplet_center_x, droplet_center_y, radius, pool_height):
    X, Y = np.ogrid[:nx, :ny]
    droplet_distance = np.sqrt((X - droplet_center_x)**2 + (Y - droplet_center_y)**2)
    droplet_mask = 0.5 * (1 - np.tanh((droplet_distance - radius) * 2))  # *2 controls sharpness
    pool_mask = 0.5 * (1 - np.tanh((Y - pool_height) * 2))

    mask = np.maximum(droplet_mask, pool_mask)
    density = gas_density + (liquid_density - gas_density) * mask

    return density

def create_obstacle_mask(nx, ny, obstacle_centers_x, obstacle_centers_y, n_obstacles, obstacle, length, solid):
    for i in range(nx):
        for j in range(ny):
            for b in range(n_obstacles):
                if obstacle == "cylinder":
                    if ((i - obstacle_centers_x[b]) ** 2 + (j - obstacle_centers_y[b]) ** 2) < length ** 2:
                        solid[i, j] = True
                        break
                
                if obstacle == "box":
                    if (np.abs((i - obstacle_centers_x[b]) + (j - obstacle_centers_y[b])) +
                    np.abs((i - obstacle_centers_x[b]) - (j - obstacle_centers_y[b])) < length):
                        solid[i, j] = True
                        break
    return solid

def wind_tunnel(f, nx, ny, nl, wind_speed):
    for i in range(nx):
        for j in range(ny):
            for k in range(nl):
                v_x0 = c_x[k] * wind_speed
                f[i, j, k] = w[k] * (1 + v_x0 / cs2 + (v_x0 ** 2) / (2 * cs2 ** 2) - (wind_speed ** 2) / (2 * cs2))
    
    return f

def rest(nl, f):
    for k in range(nl):
        f[:, :, k] = w[k]
    
    return f