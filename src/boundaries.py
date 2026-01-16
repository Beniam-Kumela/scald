import numpy as np
from .constants import *

# Encode error handling here later
def obstacle_bc(solid, nx, ny, nl, f, g, obstacle, temp_cond, n_obstacles, obstacle_centers_x, obstacle_centers_y, length, T_hot, T_cold):

    if temp_cond == "heat flux":
        for i in range(nx):
            for j in range(ny):
                if solid[i, j]:
                    if obstacle is not None:
                        is_obstacle = False
                        for b in range(n_obstacles):
                            if obstacle == "cylinder":
                                if ((i - obstacle_centers_x[b]) ** 2
                                + (j - obstacle_centers_y[b]) ** 2) < (length + 1) ** 2:
                                    is_obstacle = True
                                    break
                            
                            elif obstacle == "box":
                                if (np.abs((i - obstacle_centers_x[b]) + (j - obstacle_centers_y[b])) +
                                np.abs((i - obstacle_centers_x[b]) - (j - obstacle_centers_y[b])) < length):
                                    is_obstacle = True
                                    break
                            
                            else:
                                print("Obstacle can only be 'cylinder' or 'box'.")
                            
                        
                        T_target = T_hot if is_obstacle else T_cold

                        for k in range(nl):
                            f[i, j, opp_dir[k]] = f[i, j, k]
                            g[i, j, opp_dir[k]] = 2 * w[k] * T_target - g[i, j, k]
        
        return f, g

    else:
        for i in range(nx):
            for j in range(ny):
                if solid[i, j]:
                    for k in range(nl):
                        f[i, j, opp_dir[k]] = f[i, j, k]
        return f

def thermal_flow_inlet_bc(ny, nl, wind_speed, f, g, T_cold, solid):
    for j in range(1, ny-1):
        if not solid[0, j]:
            for k in range(nl):
                cdotv = c_x[k] * wind_speed
                f[0, j, k] = w[k] * (1 + cdotv/cs2 + (cdotv ** 2) / (2 * cs2 ** 2) - (wind_speed ** 2) / (2 * cs2))
                g[0, j, k] = w[k] * T_cold * (1 + cdotv/cs2 + (cdotv ** 2) / (2 * cs2 ** 2) - (wind_speed ** 2) / (2 * cs2))
    
    return f, g

def wind_tunnel_inlet_bc(f, nx, ny, wind_speed):
    for i in range(nx):
        # No slip, bounce-back velocity boundary conditions
        # Bottom wall
        f[i, 0, 2] = f[i, 0, 4]
        f[i, 0, 5] = f[i, 0, 7]
        f[i, 0, 6] = f[i, 0, 8]

        # Top wall
        f[i, ny-1, 4] = f[i, ny-1, 2]
        f[i, ny-1, 7] = f[i, ny-1, 5]
        f[i, ny-1, 8] = f[i, ny-1, 6]

    for j in range(1, ny-1):
        wall_density = f[0, j, 0] + f[0, j, 2] + f[0, j, 4] + 2 * (f[0, j, 3] + f[0, j, 6] + f[0, j, 7])
        f[0, j, 1] = f[0, j, 3] + 2/3 * wall_density * wind_speed
        f[0, j, 5] = f[0, j, 7] - 1/2 * (f[0, j, 2] - f[0, j, 4]) + 1/6 * wall_density * wind_speed
        f[0, j, 8] = f[0, j, 6] + 1/2 * (f[0, j, 2] - f[0, j, 4]) + 1/6 * wall_density * wind_speed 
        
    return f    

def lid_bc(f, lid_speed, nx, ny):
    j = ny - 1
    for i in range(nx):
        wall_density = (f[i, j, 0] + f[i, j, 1] + f[i, j, 3] + 2 * (f[i, j, 2] + f[i, j, 5] + f[i, j, 6]))
        
        # Consistent Zou-He updates for top wall (Moving in +x direction)
        f[i, j, 4] = f[i, j, 2]
        f[i, j, 7] = (f[i, j, 5] + 
                                     0.5 * (f[i, j, 1] - f[i, j, 3]) - 
                                     0.5 * wall_density * lid_speed)
        f[i, j, 8] = (f[i, j, 6] - 
                                     0.5 * (f[i, j, 1] - f[i, j, 3]) + 
                                     0.5 * wall_density * lid_speed)
    return f
    

def outlet_bc(nx, ny, nl, f, g, wall):
    if wall == "right":
        for k in range(nl):
            f[nx-1, :, k] = f[nx-2, :, k]
            if g is not None:
                g[nx-1, :, k] = g[nx-2, :, k]

    elif wall == "top":
        for i in range(nx):
            for k in range(nl):
                f[i, ny-1, k] = f[i, ny-2, k]
            for k in [4, 7, 8]:
                g[i, ny-1, k] = g[i, ny-2, k]
    
    else:
        print("Please choose either 'right' or 'top' for outlet wall conditions.")
    
    if g is not None:
        return f, g
    else:
        return f

def heat_flux_bc(f, g, nx, ny, T_cold, T_hot, source_start, source_end):
    for i in range(nx):
        # No slip, bounce-back velocity boundary conditions
        # Bottom wall
        f[i, 0, 2] = f[i, 0, 4]
        f[i, 0, 5] = f[i, 0, 7]
        f[i, 0, 6] = f[i, 0, 8]

        if source_start is None:
            # Top wall
            f[i, ny-1, 4] = f[i, ny-1, 2]
            f[i, ny-1, 7] = f[i, ny-1, 5]
            f[i, ny-1, 8] = f[i, ny-1, 6]
        
        else:
            is_source = (i >= source_start and i <= source_end)
            noise = 1 + 0.1 * (np.random.rand() - 0.5)
            if is_source:
                T_local = T_hot #* noise
            else:
                T_local = T_cold

        if g is not None:
            if source_start is None:
                # Anti-bounce back, temperature boundary conditions
                # Bottom wall
                g[i, 0, 2] = 2 * w[2] * T_hot - g[i, 0, 4]
                g[i, 0, 5] = 2 * w[5] * T_hot - g[i, 0, 7]
                g[i, 0, 6] = 2 * w[6] * T_hot - g[i, 0, 8]

                # Top wall
                g[i, ny-1, 4] = 2 * w[4] * T_cold - g[i, ny-1, 2]
                g[i, ny-1, 7] = 2 * w[7] * T_cold - g[i, ny-1, 5]
                g[i, ny-1, 8] = 2 * w[8] * T_cold - g[i, ny-1, 6]
            else:
                # Enforce local temperature distribution boundary on bottom wall where source is
                g[i, 0, 2] = 2 * w[2] * T_local - g[i, 0, 4]
                g[i, 0, 5] = 2 * w[5] * T_local - g[i, 0, 7]
                g[i, 0, 6] = 2 * w[6] * T_local - g[i, 0, 8]

    return f, g

def wall_bc(f, g, nx, ny, w, opp, T_cold):
    for i in range(nx):
        # bottom wall
        for k in [2,5,6]:
            f[i,0,k] = f[i,0,opp[k]]
            g[i,0,k] = 2*w[k]*T_cold - g[i,0,opp[k]]

        # top wall
        for k in [4,7,8]:
            f[i,ny-1,k] = f[i,ny-1,opp[k]]
            g[i,ny-1,k] = 2*w[k]*T_cold - g[i,ny-1,opp[k]]
    return f, g