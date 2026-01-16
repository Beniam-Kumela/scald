import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
from src.init import *
from src.kernels import *
from src.boundaries import *

# Simulation parameters
nx = 200
ny = 400
nl = 9
tau_f = 0.8
tau_g = 0.6
iterations = 1000
gravity = 0.005
alpha = 0.4
T_hot = 1
T_cold = 0.75
T_ref = T_cold

f = np.zeros((nx, ny, nl), dtype=np.float64)
g = np.zeros((nx, ny, nl), dtype=np.float64)
f, g = thermal_sim(nx, ny, nl, T_cold)
solid = np.zeros((nx, ny), dtype=bool)

source_width = nx // 10
source_start = nx // 2 - source_width // 2
source_end = nx // 2 + source_width // 2

# Rayleigh number calculation
viscosity = (tau_f - 1/2) / 3
diffusivity = (tau_g - 1/2) / 3
rayleigh_number = gravity * alpha * (T_hot - T_cold) * ny ** 3 / (viscosity * diffusivity)

print(f"Starting rising smoke simulation.")
fig, ax = plt.subplots()
temp_init = np.zeros((nx, ny))
im = ax.imshow(temp_init.T, origin = "lower", cmap = "magma", vmin = T_cold, vmax = T_hot)

def update(frame):
    global f, g

    density, T, v_x, v_y = compute_macroscopic(f, g, nx, ny, nl, solid, None)
    f, g = collision(density, T, v_x, v_y, f, g, nx, ny, nl, tau_f, tau_g, alpha, T_ref, gravity, None)
    f_new, g_new, _ = streaming(f, g, nx, ny, nl, "x", None, False)
    f_new, g_new = heat_flux_bc(f_new, g_new, nx, ny, T_cold, T_hot, source_start, source_end)
    f_new, g_new = outlet_bc(nx, ny, nl, f_new, g_new, "top")
    f, g = f_new, g_new

    im.set_array(T.T)
    ax.set_title(f"Rising Smoke, Ra = {rayleigh_number:.2e}, Time = {frame}")
    return [im]

ani = FuncAnimation(fig, update, frames = tqdm(range(iterations)), blit = True)
ani.save(filename=f"rising_smoke_ra_{rayleigh_number:.2e}.mp4", writer = "ffmpeg", fps=60)
print(f"Simulation completed successfully and saved as rising_smoke_ra_{rayleigh_number:.2e}.mp4 in the main directory.")
del ani