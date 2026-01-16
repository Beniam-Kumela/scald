import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
from src.init import *
from src.kernels import *
from src.boundaries import *

# Simulation parameters
nx = 400
ny = 200
nl = 9
tau_f = 0.9
iterations = 2000
gravity = 0.0005
liquid_density = 2
gas_density = 0.1
G = -5.5 

# Set up droplet simulation
density = np.ones((nx, ny)) * gas_density
droplet_center_x = nx // 2
droplet_center_y = int(ny * 0.7)
radius = nx // 8
pool_height = ny // 4

density = droplet_collision(nx, ny, density, liquid_density, gas_density, droplet_center_x, droplet_center_y, radius, pool_height)

# Initialize velocity distribution and solid mask (optional for this simulation)
f = np.zeros((nx, ny, nl), dtype=np.float64)
for k in range(nl):
    f[:, :, k] = w[k] * density
solid = np.zeros((nx, ny), dtype=bool)

print(f"Starting droplet collision simulation.")
fig, ax = plt.subplots()
temp_init = np.zeros((nx, ny))
im = ax.imshow(temp_init.T, origin = "lower", cmap = "Blues", vmin = gas_density, vmax = liquid_density)

def update(frame):
    global f, density, liquid_density, gas_density, radius, G

    density, _, v_x, v_y = compute_macroscopic(f, None, nx, ny, nl, solid, density)
    f, _ = collision(density, None, v_x, v_y, f, None, nx, ny, nl, tau_f, None, None, None, gravity, G)
    f_new, _, density = streaming(f, None, nx, ny, nl, "x", density)
    f = f_new

    # Calculation of dimensionless (but time-dependent) Weber number (for formula see: https://en.wikipedia.org/wiki/Weber_number) 
    U = np.max(np.abs(v_y))
    sigma = (liquid_density - gas_density) ** 2 * abs(G) / 6
    weber_number = liquid_density * U ** 2 * radius / sigma

    im.set_array(density.T)
    ax.set_title(f"Droplet Collision, We = {weber_number:.2e}, Time = {frame}")
    return [im]

ani = FuncAnimation(fig, update, frames = tqdm(range(iterations)), blit = True)
ani.save(filename=f"droplet_R_{radius:.2e}.mp4", writer = "ffmpeg", fps=60)
print(f"Simulation completed successfully and saved as droplet_R_{radius:.2e}.mp4 in the main directory.")
del ani