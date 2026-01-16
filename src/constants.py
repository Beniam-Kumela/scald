import numpy as np

# D2Q9 weights (w) and particle velocities (c_i)
w = np.array([4/9, 1/9, 1/9, 
                    1/9, 1/9, 1/36, 
                    1/36, 1/36, 1/36], dtype=np.float64)
c_x = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.int64)
c_y = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=np.int64)
opp_dir = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int64) # opposite particle velocity directions on the D2Q9 grid (ex: 1 - east and 3 - west are opposite to each other)
cs2 = 1/3 # LBM speed of sound