# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 17:35:16 2025

@author: samue
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


# Base lens parameters

sigma0 = 1500   # km/s
R_c0  = 0.07    # h^-1 Mpc
D_s   = 878
D_l   = 637
D_ls  = 441
c     = 299792.458
epsilon0 = 0.0


# Lens equations

def compute_r_c_and_thetaE(sigma, R_c):
    theta_E = (4 * np.pi * sigma**2 / c**2) * (D_ls / D_s)
    r_c = R_c / (D_l * theta_E)
    return r_c, theta_E

def lens_map(r1, r2, r_c, eps):
    denom = np.sqrt(r_c**2 + (1-eps)*r1**2 + (1+eps)*r2**2)
    s1 = r1 - (1-eps)*r1/denom
    s2 = r2 - (1+eps)*r2/denom
    return s1, s2

def norm_to_index(s, N):
    idx = ((s+1)/2)*N
    idx = np.floor(idx).astype(int)
    return np.clip(idx, 0, N-1)

# Setup grid & source plane

N = 200
xs = np.linspace(-1, 1, N)
r1, r2 = np.meshgrid(xs, xs, indexing='xy')

# Artificial extended source (bright grid pattern)
source = np.zeros((N,N))
source[::20, :] = 1
source[:, ::20] = 1


# Compute initial lens image
def compute_image(sigma, R_c, eps):
    r_c, theta_E = compute_r_c_and_thetaE(sigma, R_c)
    s1, s2 = lens_map(r1, r2, r_c, eps)
    i = norm_to_index(s1, N)
    j = norm_to_index(s2, N)
    return source[j, i]

img0 = compute_image(sigma0, R_c0, epsilon0)


# Matplotlib GUI Setup
fig, ax = plt.subplots(figsize=(7,7))
plt.subplots_adjust(left=0.25, bottom=0.25)

im = ax.imshow(img0, origin='lower', cmap='inferno')
ax.set_title("Interactive Lens GUI")
ax.set_xlabel("r1")
ax.set_ylabel("r2")


# Sliders
ax_sigma = plt.axes([0.25, 0.15, 0.65, 0.03])
ax_Rc    = plt.axes([0.25, 0.10, 0.65, 0.03])
ax_eps   = plt.axes([0.25, 0.05, 0.65, 0.03])

sl_sigma = Slider(ax_sigma, "sigma", 500, 2500, valinit=sigma0)
sl_Rc    = Slider(ax_Rc,   "R_c",   0.01, 0.2,  valinit=R_c0)
sl_eps   = Slider(ax_eps,  "eps",  -0.3, 0.3,  valinit=epsilon0)


# Slider update
def update(val):
    sigma = sl_sigma.val
    R_c   = sl_Rc.val
    eps   = sl_eps.val

    new_img = compute_image(sigma, R_c, eps)
    im.set_data(new_img)
    fig.canvas.draw_idle()

sl_sigma.on_changed(update)
sl_Rc.on_changed(update)
sl_eps.on_changed(update)

plt.show()

