import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
from src.init import *
from src.kernels import *
from src.boundaries import *

# Simulation parameters
nx = 500
ny = 250
nl = 9
tau_f = 0.7
iterations = 100
wind_speed = 0.1

f = np.zeros((nx, ny, nl), dtype=np.float64)
f = wind_tunnel(f, nx, ny, nl, wind_speed)

# Set up obstacles
solid = np.zeros((nx, ny), dtype=bool)
width = 10
n_per_side = 4
spacing = 2 * width + 50
start_x = nx // 4
start_y = ny // 3 - (n_per_side // 3) * spacing
box_centers_x = []
box_centers_y = []

for ix in range(n_per_side):
    y_offset = (spacing // 4) if(ix % 2 == 1) else 0

    for iy in range(n_per_side):
        cx = start_x + ix * spacing
        cy = start_y + iy * spacing + y_offset


        if 0 < cy < ny-1:
            box_centers_x.append(cx)
            box_centers_y.append(cy)

n_boxes = len(box_centers_x)
solid = create_obstacle_mask(nx, ny, np.array(box_centers_x), np.array(box_centers_y), n_boxes, "box", width, solid)

# Reynolds Number Calculation
viscosity = (tau_f - 0.5) / 3
reynolds_number = (wind_speed * width * 2) / viscosity

print(f"Starting flow past a box simulation.")
fig, ax = plt.subplots()
speed_init = np.zeros((nx, ny))
im = ax.imshow(speed_init.T, origin = "lower", cmap = "viridis", vmin = 0, vmax = 1.5*wind_speed)

def update(frame):
    global f

    density, _, v_x, v_y = compute_macroscopic(f, None, nx, ny, nl, solid, None)
    f, _ = collision(density, None, v_x, v_y, f, None, nx, ny, nl, tau_f, None, None, None, None, None)
    f_new, _, _ = streaming(f, None, nx, ny, nl, None, None, False)
    f_new = wind_tunnel_inlet_bc(f_new, nx, ny, wind_speed)
    f_new = outlet_bc(nx, ny, nl, f_new, None, "right")
    f = f_new

    speed = np.sqrt(v_x ** 2 + v_y ** 2)
    im.set_array(speed.T)
    ax.set_title(f"Flow past Box, Re = {reynolds_number:.2e}, Time = {frame}")
    return [im]

ani = FuncAnimation(fig, update, frames = tqdm(range(iterations)), blit = True)
ani.save(filename=f"box_flow_re_{reynolds_number:.2e}.mp4", writer = "ffmpeg", fps=60)
print(f"Simulation completed successfully and saved as box_flow_re_{reynolds_number:.2e}.mp4 in the main directory.")
del ani