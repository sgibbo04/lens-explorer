# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 17:32:36 2025

@author: samue
"""

# script_D JPEG lensing
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Parameter values
sigma = 1500
R_c = 0.07
D_s = 878; D_l = 637; D_ls = 441
c = 299792.458
epsilon = 0.0

def compute_thetaE(sigma):
    return (4 * np.pi * sigma**2 / c**2) * (D_ls / D_s)
theta_E = compute_thetaE(sigma)
r_c = R_c /( D_l* theta_E)

def lens_map(r1,r2,r_c,epsilon):
    denominator = np.sqrt(r_c**2 + (1-epsilon) * r1**2 + (1+epsilon) * r2**2)
    s1 = r1 - (1-epsilon) * r1 / denominator 
    s2 = r2 - (1+epsilon) * r2 / denominator
    return s1, s2

# helper to convert normalized coords [-1,1] to float pixel indices [0,N-1]
def norm_to_pixel(s, N):
    return ((s + 1) / 2.0) * (N - 1)

# bilinear sampler (added extension)
def bilinear_sample(img, x, y):
    H, W = img.shape[:2]
    x0 = np.floor(x).astype(int); y0 = np.floor(y).astype(int)
    x1 = x0 + 1; y1 = y0 + 1
    x0 = np.clip(x0, 0, W-1); x1 = np.clip(x1, 0, W-1)
    y0 = np.clip(y0, 0, H-1); y1 = np.clip(y1, 0, H-1)
    dx = (x - x0)[..., None]; dy = (y - y0)[..., None]
    Ia = img[y0, x0]; Ib = img[y1, x0]; Ic = img[y0, x1]; Id = img[y1, x1]
    return (Ia * (1-dx)*(1-dy) + Ic * dx*(1-dy) + Ib * (1-dx)*dy + Id * dx*dy)

# grid
N = 201
xs = np.linspace(-1, 1, N)
r1, r2 = np.meshgrid(xs, xs, indexing='xy')

# load source.jpg and resize to NxN
try:
    pil = Image.open("source.jpg").convert("RGB")
    pil = pil.resize((N, N))
    source = np.array(pil).astype(np.float32) / 255.0
except FileNotFoundError:
    print("source.jpg not found. Create a simple synthetic color pattern instead.")
    # generate a synthetic colorful pattern if no file
    source = np.zeros((N, N, 3), dtype=np.float32)
    for i in range(N):
        for j in range(N):
            source[j,i,0] = (i % 20) / 20.0
            source[j,i,1] = (j % 30) / 30.0
            source[j,i,2] = ((i+j) % 25) / 25.0

# lens mapping (inverse mapping)
s1, s2 = lens_map(r1, r2, r_c, epsilon)
u = norm_to_pixel(s1, N)
v = norm_to_pixel(s2, N)

lensed = bilinear_sample(source, u, v)
lensed = np.clip(lensed, 0, 1)

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.imshow(source, origin='lower')
plt.title("Source image (resized)")
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(lensed, origin='lower')
plt.title("Lensed image (bilinear sampling)")
plt.axis('off')
plt.show()
