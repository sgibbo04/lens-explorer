# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 17:30:15 2025

@author: samue
"""

# script_B: Magnification 
import numpy as np
import matplotlib.pyplot as plt

#Lens Parameters
sigma = 1500 # kms^-1
R_c = 0.07 # h^-1Mpc
D_s = 878
D_l = 637
D_ls = 441
c = 299792.458
epsilon = 0.0

def compute_thetaE(sigma):
    return (4 * np.pi * sigma**2 / c**2) * (D_ls / D_s)

theta_E = compute_thetaE(sigma)
r_c = R_c /( D_l* theta_E)
print("theta_E =", theta_E, "r_c =", r_c)

def lens_map(r1,r2,r_c,epsilon):
    denominator = np.sqrt(r_c**2 + (1-epsilon) * r1**2 + (1+epsilon) * r2**2)
    s1 = r1 - (1-epsilon) * r1 / denominator 
    s2 = r2 - (1+epsilon) * r2 / denominator
    return s1, s2

# grid
N = 101
xs = np.linspace(-1, 1, N)
r1, r2 = np.meshgrid(xs, xs, indexing='xy')
dx = xs[1] - xs[0]

# compute s = lens_map and derive alpha = theta - s
s1, s2 = lens_map(r1, r2, r_c, epsilon)
alpha1 = r1 - s1
alpha2 = r2 - s2

# numerical derivatives (central differences via numpy.gradient)
# gradient axis 1 -> x (columns), axis 0 -> y (rows)
dalpha1_dx = np.gradient(alpha1, dx, axis=1)
dalpha1_dy = np.gradient(alpha1, dx, axis=0)
dalpha2_dx = np.gradient(alpha2, dx, axis=1)
dalpha2_dy = np.gradient(alpha2, dx, axis=0)

# Jacobian A = I - dAlpha/dTheta
A11 = 1.0 - dalpha1_dx
A12 = - dalpha1_dy
A21 = - dalpha2_dx
A22 = 1.0 - dalpha2_dy

detA = A11 * A22 - A12 * A21
mu = 1.0 / (detA + 1e-12)  # signed magnification

# Plot detA and magnification
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.imshow(detA, origin='lower', cmap='seismic', extent=[-1,1,-1,1])
plt.colorbar(label='det(A)')
plt.title('det(A) (Jacobian determinant)')

plt.subplot(1,2,2)
plt.imshow(np.sign(mu) * np.log10(np.abs(mu)+1e-6), origin='lower', cmap='RdBu', extent=[-1,1,-1,1])
plt.colorbar(label='sign*log10(|mu|)')
plt.title('Magnification (log scale)')

plt.tight_layout()
plt.show()

# Find approximate critical curve points in image plane: where detA ~ 0
threshold = np.percentile(np.abs(detA), 2)  # small fraction of smallest values
mask = np.abs(detA) < (threshold + 1e-12)
print("Found", np.count_nonzero(mask), "pixels with |detA| small -> candidate critical points")

# Map those image-plane points to source-plane coordinates (caustic points)
beta1 = r1 - alpha1
beta2 = r2 - alpha2
caust_x = beta1[mask]
caust_y = beta2[mask]

plt.figure(figsize=(5,5))
plt.scatter(caust_x, caust_y, s=2, c='red')
plt.title("Caustic points (mapped to source plane)")
plt.xlabel("beta1")
plt.ylabel("beta2")
plt.xlim(-1,1); plt.ylim(-1,1)
plt.gca().set_aspect('equal', 'box')
plt.show()
