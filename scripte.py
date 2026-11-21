# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 17:34:41 2025

@author: samue
"""

import numpy as np
import matplotlib.pyplot as plt


# Lens parameters
sigma = 1500
R_c = 0.07
D_s = 878
D_l = 637
D_ls = 441
c = 299792.458
eps = 0


# Lens equations
def compute_r_c_and_thetaE():
    theta_E = (4*np.pi*sigma**2/c**2)*(D_ls/D_s)
    r_c = R_c/(D_l*theta_E)
    return r_c, theta_E

def lens_map(r1, r2, r_c, e):
    denom = np.sqrt(r_c**2 + (1-e)*r1**2 + (1+e)*r2**2)
    s1 = r1 - (1-e)*r1/denom
    s2 = r2 - (1+e)*r2/denom
    return s1, s2

def norm_to_index(s, N):
    idx = ((s+1)/2)*N
    idx = np.floor(idx).astype(int)
    return np.clip(idx, 0, N-1)


# Grid
N = 200
xs = np.linspace(-1,1,N)
r1, r2 = np.meshgrid(xs, xs, indexing='xy')


# Source plane: chessboard
source = np.zeros((N,N))
source[::20, :] = 1
source[:, ::20] = 1


# Lensing
r_c, theta_E = compute_r_c_and_thetaE()
s1, s2 = lens_map(r1, r2, r_c, eps)
i = norm_to_index(s1, N)
j = norm_to_index(s2, N)
image = source[j, i]


# Distortion field (shear)
dr = 0.02
r1p = r1 + dr
r2p = r2 + dr

s1p, s2p = lens_map(r1p, r2p, r_c, eps)

u = s1p - s1
v = s2p - s2


# Plot
fig, axs = plt.subplots(1, 3, figsize=(15,5))

# Original chessboard
axs[0].imshow(source, origin='lower', cmap='binary')
axs[0].set_title("Original Chessboard Source")

# Lensed image
axs[1].imshow(image, origin='lower', cmap='inferno')
axs[1].set_title("Lensed Chessboard")

# Distortion field
axs[2].quiver(r1, r2, u, v, color='red', scale=0.5)
axs[2].set_title("Distortion Field (Shear)")

plt.tight_layout()
plt.show()

