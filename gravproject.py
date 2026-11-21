# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 12:24:01 2025

@author: samue
"""

import numpy as np
import matplotlib.pyplot as plt
import math 

#%%
#Parameter values 
sigma = 1500 #kms^-1
R_c = 0.07 #h^-1Mpc 
D_s = 878 #h^-1Mpc
D_l = 637 #h^-1Mpc
D_ls = 441 #h^-1Mpc 
c = 299792.458 #kms^-1
epsilon = 0 # Test case of ellipticity

#%%
#Compute Einstein radius and core radius
theta_E = (4 * np.pi * sigma**2 / c**2) * (D_ls / D_s)
r_c = R_c /( D_l* theta_E)
r_analytic = np.sqrt(1 - r_c**2)
print("theta_E =", theta_E)
print("r_c  =", r_c)
print("r_analytic = ", r_analytic )

#%%
#Lens mapping 
def lens_map(r1,r2,r_c,epsilon):
    """
    Given r1, r2 (arrays), we can compute s1, s2 and transfer between
    lens and source image 
    
    """
    denominator = np.sqrt(r_c**2 + (1-epsilon) * r1**2 + (1+epsilon) * r2**2)
    s1 = r1 - (1-epsilon) * r1 / denominator 
    s2 = r2 - (1+epsilon) * r2 / denominator
    return s1, s2

#%%
#Create a 101x101 grid of r1, r2 in [-1,1]
N = 101 
xs = np.linspace(-1, 1, N)
r1, r2 = np.meshgrid(xs, xs, indexing='xy') #r1: x r2: y
mid = N//2
#%%
#Define a linear map from normalised s1,s2 to pixel indices
def norm_to_index(s, N=101):
    """
    Convert from s in [-1,1] to integer array index in [0,N]
    
    """
    idx = ((s+1) / 2)*(N) # map [-1,1]-->[0,100]
    idx = np.floor(idx).astype(int) #convert to integer for pixel
    return np.clip(idx, 0, N-1) #Keeps array within bounds

#Convert s1,s2 arrays produced by lens map
s1, s2 = lens_map(r1, r2, r_c, epsilon)
#Rescale s1,s2 to pixel indices based on min/max

#Clip indices 
i = norm_to_index(s1) #x pixel 
j = norm_to_index(s2) #y pixel


print(i.shape, j.shape)

#Source plane with single pixel at the centre
source_plane = np.zeros((N , N))
source_plane[N//2 , N//2] = 1

#Image plane using linear mapping 
image_plane = np.zeros((N , N))
image_plane = source_plane[j , i] #y = j , x = i


#Parametric circle coords
theta = np.linspace(0, 2*np.pi, 500)
circle_x = r_analytic * np.cos(theta)
circle_y = r_analytic * np.sin(theta)

#Convert circle coords to pixel coords
circle_x_pix = ( (circle_x + 1) / 2 ) * N
circle_y_pix = ( (circle_y + 1) / 2 ) * N

#Plot image plane 
plt.figure(figsize=(8,8))
plt.imshow(image_plane, origin='lower', cmap='inferno')
plt.xlabel("r_1 pixel number")
plt.ylabel("r_2 pixel number")
plt.plot(circle_x_pix, circle_y_pix, 'r', linewidth=1)
plt.title("Einstein Ring for ε = 0")
plt.show()