import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
from src.init import *
from src.kernels import *
from src.boundaries import *

# Simulation parameters
nx = 500
ny = 100
nl = 9
tau_f = 0.8
tau_g = 0.6
iterations = 100
wind_speed = 0.1
gravity = 0.0
alpha = 0.01
T_hot = 1
T_cold = 0

# Initialize velocity (f) and temperature (g) arrays
f = np.zeros((nx, ny, nl), dtype=np.float64)
g = np.zeros((nx, ny, nl), dtype=np.float64)

# For forced convection we initialize distributions with initially cold fluid (using src.init.thermal_sim) flowing from the left (using src.init.wind_tunnel)
f, g = thermal_sim(nx, ny, nl, T_cold)
f = wind_tunnel(f, nx, ny, nl, wind_speed)

# Set up cylindrical obstacles
solid = np.zeros((nx, ny), dtype=bool)
radius = 5
n_per_side = 4
spacing = 2 * radius + 10  #2 * radius + 50
start_x = nx // 4
start_y = ny // 3 - (n_per_side // 3) * spacing
cylinder_centers_x = []
cylinder_centers_y = []

for ix in range(n_per_side):
    y_offset = 0#(spacing // 4) if(ix % 2 == 1) else 0 # can be used to offset cylinder arrangements if asymmetric pattern desired

    for iy in range(n_per_side):
        cx = start_x + ix * spacing
        cy = start_y + iy * spacing + y_offset

        if 0 < cy < ny-1:
            cylinder_centers_x.append(cx)
            cylinder_centers_y.append(cy)

n_cylinders = len(cylinder_centers_x)
solid = create_obstacle_mask(nx, ny, np.array(cylinder_centers_x), np.array(cylinder_centers_y), n_cylinders, "cylinder", radius, solid)
solid[:, 0] = True      # Bottom wall
solid[:, ny-1] = True   # Top wall

# Initialize obstacles to be initially hot
f, g = thermal_obstacle_flow(f, g, n_cylinders, cylinder_centers_x, cylinder_centers_y, radius, nx, ny, nl, "cylinder", T_hot)

# Dimensionless number calculation for file marking
viscosity = (tau_f - 0.5) / 3
diffusivity = (tau_g - 1/2) / 3
reynolds_number = (wind_speed * radius * 2) / viscosity
prandtl_number = viscosity / diffusivity

# Initialize graph to be updated each frame
print(f"Starting thermal flow past a cylinder simulation.")
fig, ax = plt.subplots()
T_init = np.zeros((nx, ny))
im = ax.imshow(T_init.T, origin = "lower", cmap = "magma", vmin = T_cold, vmax = T_hot)

# Each LBM step is defined by the following procedure
def update(frame):
    global f, g

    density, T, v_x, v_y = compute_macroscopic(f, g, nx, ny, nl, solid, None)
    f, g = collision(density, T, v_x, v_y, f, g, nx, ny, nl, tau_f, tau_g, alpha, T_cold, gravity, None)
    f_new, g_new, _ = streaming(f, g, nx, ny, nl, None, None, False)
    f_new, g_new = obstacle_bc(solid, nx, ny, nl, f_new, g_new, "cylinder", "heat flux", n_cylinders, cylinder_centers_x, cylinder_centers_y, radius, T_hot, T_cold)
    f_new, g_new = thermal_flow_inlet_bc(ny, nl, wind_speed, f_new, g_new, T_cold, solid)
    f_new, g_new = outlet_bc(nx, ny, nl, f_new, g_new, "right")

    f, g = f_new, g_new

    im.set_array(T.T)
    ax.set_title(f"Thermal flow past cylinder, Re = {reynolds_number:.2e}, Pr = {prandtl_number:.2e}, Time = {frame}")
    return [im]

# Save animation and free up memory usage
ani = FuncAnimation(fig, update, frames = tqdm(range(iterations)), blit = True)
ani.save(filename=f"cylinder_thermal_flow_re_{reynolds_number:.2e}_pr_{prandtl_number:.2e}.mp4", writer = "ffmpeg", fps=60)
print(f"Simulation completed successfully and saved as cylinder_thermal_flow_re_{reynolds_number:.2e}_pr_{prandtl_number:.2e}.mp4 in the main directory.")
del ani