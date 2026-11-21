# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 17:31:02 2025

@author: samue
"""

import numpy as np
import matplotlib.pyplot as plt


# Lens parameters

sigma = 1500
R_c = 0.07
epsilon = 0.0
D_s = 878
D_l = 637
D_ls = 441
c = 299792.458

# Grid and lens equations
N = 201
xs = np.linspace(-1,1,N)
r1, r2 = np.meshgrid(xs, xs, indexing='xy')

def compute_thetaE_r_c():
    theta_E = (4*np.pi*sigma**2/c**2)*(D_ls/D_s)
    r_c = R_c/(D_l*theta_E)
    return theta_E, r_c

def lens_map(r1, r2, r_c, eps):
    denom = np.sqrt(r_c**2 + (1-eps)*r1**2 + (1+eps)*r2**2)
    s1 = r1 - (1-eps)*r1/denom
    s2 = r2 - (1+eps)*r2/denom
    return s1, s2

def norm_to_index(s, N):
    idx = ((s+1)/2)*N
    idx = np.floor(idx).astype(int)
    return np.clip(idx, 0, N-1)


# Galaxy generator
def make_random_galaxies(N, num_gal=20):
    S = np.zeros((N,N))
    Y, X = np.indices((N,N))

    for _ in range(num_gal):
        cx = np.random.uniform(0, N)
        cy = np.random.uniform(0, N)
        a = np.random.uniform(5, 20)      # scale length
        peak = np.random.uniform(0.5, 1.0)
        ell = np.random.uniform(0.6, 1.0) # ellipticity

        dx = X - cx
        dy = (Y - cy) / ell

        R = np.sqrt(dx*dx + dy*dy)
        S += peak * np.exp(-R / a)

    S /= np.max(S)
    return S


# Make source plane of galaxies

source = make_random_galaxies(N, num_gal=25)


# Lens the galaxies

theta_E, r_c = compute_thetaE_r_c()
s1, s2 = lens_map(r1, r2, r_c, epsilon)
i = norm_to_index(s1, N)
j = norm_to_index(s2, N)
image = source[j, i]


# Plot SOURCE vs LENSED (side-by-side)

fig, axs = plt.subplots(1, 2, figsize=(9,5))   

axs[0].imshow(source, origin="lower", cmap="inferno")
axs[0].set_title("Source Plane (Galaxies)")
axs[0].set_xlabel("r1")
axs[0].set_ylabel("r2")

axs[1].imshow(image, origin="lower", cmap="inferno")
axs[1].set_title("Lensed Image")
axs[1].set_xlabel("r1")
axs[1].set_ylabel("r2")

plt.tight_layout()
plt.show()


