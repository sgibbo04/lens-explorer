# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 18:12:18 2025

@author: samue
"""

import numpy as np
import matplotlib.pyplot as plt

# 
# LENS PARAMETER BLOCK — CHANGE THESE TO EXPLORE LENSING
# 

sigma   = 1500        # lens velocity dispersion (km/s)
R_c     = 0.07        # core radius (h^-1 Mpc)
epsilon = 0.2        # ellipticity (0 = circular)
N       = 201         # grid resolution (increase for smoother images)
source_type = "point" # "point", "ring", "grid", or "gaussian"


# Cosmology / lens distance constants

D_s  = 878
D_l  = 637
D_ls = 441
c    = 299792.458


#Lens Equations


def compute_thetaE_r_c(sigma, R_c):
    theta_E = (4*np.pi*sigma**2/c**2)*(D_ls/D_s)
    r_c = R_c/(D_l * theta_E)
    return theta_E, r_c

def lens_map(r1, r2, r_c, e):
    denom = np.sqrt(r_c**2 + (1-e)*r1**2 + (1+e)*r2**2)
    s1 = r1 - (1-e)*r1/denom
    s2 = r2 - (1+e)*r2/denom
    return s1, s2

def norm_to_index(s, N):
    idx = ((s+1)/2)*N
    idx = np.floor(idx).astype(int)
    return np.clip(idx, 0, N-1)


#Choose a source plane model


def make_source_plane(N, mode):
    S = np.zeros((N,N))

    if mode == "point":
        S[N//2, N//2] = 1

    elif mode == "grid":
        S[::20,:] = 1
        S[:,::20] = 1

    elif mode == "ring":
        cx, cy = N//2, N//2
        Y,X = np.indices((N,N))
        R = np.sqrt((X-cx)**2 + (Y-cy)**2)
        S[np.abs(R-30) < 2] = 1

    elif mode == "gaussian":
        cx, cy = N//2, N//2
        Y,X = np.indices((N,N))
        R2 = (X-cx)**2 + (Y-cy)**2
        S = np.exp(-R2/(2*20*20))

    return S


# Rendering function: this is what changes each run


def render():
    # setup grid
    xs = np.linspace(-1,1,N)
    r1, r2 = np.meshgrid(xs, xs, indexing='xy')

    # get lens quantities
    theta_E, r_c = compute_thetaE_r_c(sigma, R_c)

    # lens mapping
    s1, s2 = lens_map(r1, r2, r_c, epsilon)
    i = norm_to_index(s1, N)
    j = norm_to_index(s2, N)

    # source plane
    source = make_source_plane(N, source_type)
    image  = source[j,i]

    # analytic (elliptical) critical curve
    r_crit = np.sqrt(1 - r_c**2)
    theta  = np.linspace(0,2*np.pi,500)
    a = r_crit / np.sqrt(1-epsilon)    # x-axis stretch
    b = r_crit / np.sqrt(1+epsilon)    # y-axis stretch
    xc = a*np.cos(theta)
    yc = b*np.sin(theta)
    xc_pix = ((xc+1)/2)*N
    yc_pix = ((yc+1)/2)*N

    # plotting
    plt.figure(figsize=(7,7))
    plt.imshow(image, origin='lower', cmap='inferno')
    plt.plot(xc_pix, yc_pix, 'r', lw=1.2)
    plt.title(f"Lensed Image\nsigma={sigma}, R_c={R_c}, eps={epsilon}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

render()
