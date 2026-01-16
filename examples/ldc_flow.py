import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
from src.init import rest
from src.kernels import *
from src.boundaries import *

nx = 200
ny = 200
nl = 9
tau_f = 0.6               
iterations = 5000
lid_speed = 0.2

f = np.zeros((nx, ny, nl), dtype=np.float64)
f = rest(nl, f)

solid = np.zeros((nx, ny), dtype=bool)
solid[0, :] = True
solid[nx-1, :] = True
solid[:, 0] = True
solid[:, ny-1] = False 

# Reynolds Number Calculation
viscosity = (tau_f - 0.5) / 3
reynolds_number = (lid_speed * nx) / viscosity

print(f"Starting lid-driven cavity flow simulation.")
fig, ax = plt.subplots()
speed_init = np.zeros((nx, ny))
im = ax.imshow(speed_init.T, origin = "lower", cmap = "viridis", vmin = 0, vmax = lid_speed)

def update(frame):
    global f

    density, _, v_x, v_y = compute_macroscopic(f, None, nx, ny, nl, solid, None)
    f, _ = collision(density, None, v_x, v_y, f, None, nx, ny, nl, tau_f, None, None, None, None, None)
    f_new, _, _ = streaming(f, None, nx, ny, nl, None, None, False)

    curr_speed = lid_speed * min(1.0, (frame + 1) / 100)
    f = lid_bc(f_new, curr_speed, nx, ny)

    speed = np.sqrt(v_x ** 2 + v_y ** 2)
    im.set_array(speed.T)
    ax.set_title(f"Lid Driven Cavity Flow, Re = {reynolds_number:.2e}, Time = {frame}")
    return [im]

ani = FuncAnimation(fig, update, frames = tqdm(range(iterations)), blit = True)
ani.save(filename=f"ldc_flow_re_{reynolds_number:.2e}.mp4", writer = "ffmpeg", fps=60)
print(f"Simulation completed successfully and saved as ldc_flow_re_{reynolds_number:.2e}.mp4 in the main directory.")
del ani