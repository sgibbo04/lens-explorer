# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 11:57:54 2025

@author: samue
"""

"""
lensing_explorer.py

Dependencies:
 - numpy
 - matplotlib
 - Pillow (PIL)

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from PIL import Image, ImageDraw
import os


#PHYSICAL PARAMETERS
sigma0 = 1500.0   # km/s (used to compute theta_E)
R_c0 = 0.07       # h^-1 Mpc
D_s = 878.0
D_l = 637.0
D_ls = 441.0
c_kms = 299792.458

# grid resolution (image-plane). Increase for more quality (cost ~N^2).
N = 601  # choose 401..1201 depending on computational power

# source options
USE_REAL_IMAGE = os.path.exists("source.jpg")  # if source.jpg exists, will use it via button
CHESSBOARD_SIZE = 8  # tiles across when using synthetic chessboard


#LENS MODEL 

def compute_thetaE_from_sigma(sigma):
    return (4 * np.pi * sigma**2 / c_kms**2) * (D_ls / D_s)

def lens_map(r1, r2, r_c, epsilon):
    """
    Given r1, r2 (arrays), we can compute s1, s2 and transfer between
    lens and source image 
    
    """
    denominator = np.sqrt(r_c**2 + (1-epsilon) * r1**2 + (1+epsilon) * r2**2)
    s1 = r1 - (1-epsilon) * r1 / denominator
    s2 = r2 - (1+epsilon) * r2 / denominator
    return s1, s2

# Optionally add external shear deflection to alpha by adding the linear shear matrix (used later)
def shear_deflection(dx, dy, gamma1, gamma2):
    # returns ax_shear, ay_shear
    ax = gamma1 * dx + gamma2 * dy
    ay = gamma2 * dx - gamma1 * dy
    return ax, ay


#COORDINATES / GRIDS
xs = np.linspace(-1, 1, N)
r1_grid, r2_grid = np.meshgrid(xs, xs, indexing="xy")  # r1 = x, r2 = y

# utility: convert normalized [-1,1] coords to pixel float indices in [0, N-1]
def norm_to_pixel_coords(s, N):
    # returns float pixel coords
    return ((s + 1.0) / 2.0) * (N - 1)


#BILINEAR SAMPLER (works for RGB arrays HxWx3)

def bilinear_sample(img, x, y):
    """
    img: H x W x C (C=3)
    x, y: arrays of same shape giving float pixel coordinates in img space
    returns sampled array shape (..., C)
    """
    H, W = img.shape[:2]
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1

    x0 = np.clip(x0, 0, W-1)
    x1 = np.clip(x1, 0, W-1)
    y0 = np.clip(y0, 0, H-1)
    y1 = np.clip(y1, 0, H-1)

    dx = (x - x0)[..., None]
    dy = (y - y0)[..., None]

    Ia = img[y0, x0]  # HxWx3
    Ib = img[y1, x0]
    Ic = img[y0, x1]
    Id = img[y1, x1]

    return (Ia * (1-dx)*(1-dy) + Ic * dx*(1-dy) +
            Ib * (1-dx)*dy + Id * dx*dy)


#SOURCE PLANE GENERATORS

def generate_chessboard(N, tiles=8):
    """Generates a colorful chessboard-like image (RGB float 0..1)."""
    img = Image.new("RGB", (N, N), (255,255,255))
    draw = ImageDraw.Draw(img)
    tile = N // tiles
    colors = [(240, 80, 80), (80, 200, 120), (50, 120, 240), (240, 200, 60)]
    for i in range(tiles):
        for j in range(tiles):
            color = colors[(i+j) % len(colors)]
            x0, y0 = i*tile, j*tile
            x1, y1 = x0 + tile, y0 + tile
            draw.rectangle([x0, y0, x1, y1], fill=color)
    return np.array(img).astype(np.float32) / 255.0

def exponential_galaxy(xx, yy, x0, y0, a, q=1.0, phi=0.0, I0=1.0):
    """
    2D elliptical exponential profile centered at (x0,y0) with scale length a.
    xx,yy are coordinates in normalized [-1,1] space mapped to pixel coords or similar.
    q = axis ratio b/a, phi in radians rotation of major axis.
    Returns intensity (float) same shape as xx/yy
    """
    # rotate coordinate frame by -phi so major axis aligned with x'
    xr = (xx - x0) * np.cos(phi) + (yy - y0) * np.sin(phi)
    yr = -(xx - x0) * np.sin(phi) + (yy - y0) * np.cos(phi)
    # elliptical radius
    r_ell = np.sqrt((xr**2) + (yr**2) / (q**2))
    return I0 * np.exp(-r_ell / a)

def random_galaxy_population(N, n_gal=30, a_min=0.02, a_max=0.12, seed=123):
    """
    Build a source plane with n_gal random exponential galaxies.
    Positions chosen in normalized [-1,1] coords.
    Returns an RGB float image (HxWx3).
    """
    rng = np.random.default_rng(seed)
    H = W = N
    img = np.zeros((H, W, 3), dtype=np.float32)
    # normalized coordinate grid in [-1,1]
    xs = np.linspace(-1,1,N)
    X, Y = np.meshgrid(xs, xs, indexing="xy")
    for k in range(n_gal):
        x0 = rng.uniform(-0.9, 0.9)
        y0 = rng.uniform(-0.9, 0.9)
        a = rng.uniform(a_min, a_max)  # scale length
        q = rng.uniform(0.5, 1.0)      # axis ratio
        phi = rng.uniform(0, 2*np.pi)
        I0 = rng.uniform(0.3, 1.2)
        color = rng.uniform(0.2, 1.0, size=3)
        profile = exponential_galaxy(X, Y, x0, y0, a, q=q, phi=phi, I0=I0)
        # mix into RGB channels
        for ch in range(3):
            img[..., ch] += profile * color[ch]
    # clip and normalize moderately
    img = img / (np.max(img) + 1e-12)
    img = np.clip(img, 0, 1)
    return img

# load optional real JPEG if present
if USE_REAL_IMAGE:
    try:
        pil = Image.open("source.jpg").convert("RGB")
        pil = pil.resize((N, N))
        IMG_REAL = np.array(pil).astype(np.float32) / 255.0
    except Exception as e:
        print("Could not load source.jpg:", e)
        USE_REAL_IMAGE = False
        IMG_REAL = None
else:
    IMG_REAL = None

# initial source
source_random = random_galaxy_population(N, n_gal=40, a_min=0.02, a_max=0.12, seed=42)
source_chess = generate_chessboard(N, tiles=CHESSBOARD_SIZE)
source_current = source_random.copy()


# MAGNIFICATION / JACOBIAN CALCULATION
def compute_deflection_field(thetaE_pix, r_c, epsilon, gamma1=0.0, gamma2=0.0):
    """
    Compute deflection alpha_x, alpha_y (in pixel units consistent with normalized coords -> pixels).
    We'll compute directly on the r1_grid,r2_grid normalized coordinates and then convert to pixel coords later.
    Return arrays ax, ay (shape NxN).
    """
    # Use normalized coordinates r1_grid,r2_grid (these are in [-1,1]).
    # The deflection formula you used implicitly works in normalized units; to combine with pixel sampling
    # To remain consistent: compute alpha in same normalized units.
    ax, ay = lens_map(r1_grid, r2_grid, r_c, epsilon)
    # lens_map returned s1 = theta - alpha => alpha = theta - s
    alpha_x = r1_grid - ax
    alpha_y = r2_grid - ay

    # add external shear contribution (in normalized coord units)
    # shear deflection (linear) computed with normalized coords
    ash_x, ash_y = shear_deflection(r1_grid, r2_grid, gamma1, gamma2)
    alpha_x += ash_x
    alpha_y += ash_y

    return alpha_x, alpha_y

def compute_jacobian_and_magnification(alpha_x, alpha_y, xs_grid):
    """
    Compute partial derivatives d alpha_x / d theta_x etc via central differences on the grid.
    xs_grid is the 1D xs array (linspace -1..1) used for spacing.
    Returns (A11,A12,A21,A22) where A = I - dAlpha/dTheta, and magnification mu and detA.
    """
    # grid spacing in normalized coords (theta)
    dx = xs_grid[1] - xs_grid[0]
    # compute derivatives (central difference)
    dax_dx = np.gradient(alpha_x, dx, axis=1)  # d alpha_x / d theta_x
    dax_dy = np.gradient(alpha_x, dx, axis=0)  # d alpha_x / d theta_y
    day_dx = np.gradient(alpha_y, dx, axis=1)
    day_dy = np.gradient(alpha_y, dx, axis=0)

    # Jacobian A = d beta / d theta = I - d alpha / d theta
    A11 = 1.0 - dax_dx
    A12 = - dax_dy
    A21 = - day_dx
    A22 = 1.0 - day_dy

    detA = A11 * A22 - A12 * A21
    # magnification (signed)
    mu = 1.0 / (detA + 1e-12)
    return A11, A12, A21, A22, detA, mu


# LENSING & RENDERING
def render_lensed_image(source_img, thetaE, r_c, epsilon, gamma1=0.0, gamma2=0.0):
    """
    source_img: HxWx3 float in [0,1]
    thetaE: Einstein radius in *normalized units* (we will derive mapping between normalized coords and pixels)
           BUT to keep your original approach consistent we scale thetaE (which came from sigma) by hand.
           Simpler approach: treat normalized coords as angular units; thetaE provided in same normalized units.
    Returns lensed_rgb (HxWx3), magnification mu, detA, beta_x,beta_y (source coords)
    """
    H, W = source_img.shape[:2]
    # compute deflection alpha on normalized grid
    # since lens_map computes in normalized r1,r2 coords, we pass r1_grid,r2_grid
    alpha_x, alpha_y = compute_deflection_field(thetaE, r_c, epsilon, gamma1, gamma2)

    # beta = theta - alpha
    beta_x = r1_grid - alpha_x
    beta_y = r2_grid - alpha_y

    # convert beta normalized [-1,1] to pixel coords of source_img
    u = norm_to_pixel_coords(beta_x, W)
    v = norm_to_pixel_coords(beta_y, H)

    # sample source image using bilinear interpolation
    lensed = bilinear_sample(source_img, u, v)

    # compute Jacobian & magnification
    A11, A12, A21, A22, detA, mu = compute_jacobian_and_magnification(alpha_x, alpha_y, xs)

    return lensed, mu, detA, beta_x, beta_y, alpha_x, alpha_y


# DISTORTION MEASURES

def shear_and_direction(A11, A12, A21, A22):
    """
    From Jacobian A = [[A11, A12],[A21,A22]] compute the 2D symmetric shear-like matrix
    and get principal stretching directions and magnitudes. Returns eigenvalues and eigenvectors.
    """
    # Compute deformation matrix M = A.T @ A (or consider the amplification matrix?), but for direction of stretch
    # One simple approach: compute symmetric part of A (S=(A + A^T)/2) and take eigenvectors.
    # Alternatively compute (A^-1)^T (A^-1) for source-plane distortion. We'll provide principal axes of A.
    # We'll compute eigen-decomposition of (A^T A) which gives squared stretching factors.
    # For each grid point this is small; compute vectorised using closed form for 2x2 matrices.

    # trace and det of AtA
    t = (A11**2 + A12**2 + A21**2 + A22**2)
    # not using t, instead compute eigenvalues of AtA explicitly:
    # For 2x2 [[a,b],[c,d]]: AtA = [[a^2+c^2, a*b+c*d],[a*b+c*d, b^2+d^2]]
    m11 = A11**2 + A21**2
    m12 = A11*A12 + A21*A22
    m22 = A12**2 + A22**2

    # eigenvalues lam = 0.5*(m11+m22 +/- sqrt((m11-m22)^2 + 4 m12^2))
    trace = m11 + m22
    diff = m11 - m22
    disc = np.sqrt(diff**2 + 4.0 * m12**2)
    lam1 = 0.5 * (trace + disc)
    lam2 = 0.5 * (trace - disc)

    # principal stretching magnitudes = sqrt(eigenvalues)
    s1 = np.sqrt(np.abs(lam1))
    s2 = np.sqrt(np.abs(lam2))

    # principal axis directions (angle) from eigenvector of m:
    # eigenvector for lam1 is [m12, lam1 - m11] or fallback
    vx = m12
    vy = lam1 - m11
    # normalize and compute angle
    ang = np.arctan2(vy, vx + 1e-18)  # angle in radians (may be noisy where vx~0)
    return s1, s2, ang


# PLOTTING / GUI

# initial lens parameters
sigma = sigma0
thetaE0 = compute_thetaE_from_sigma(sigma0)  # NOTE: this thetaE is in the same normalized units as r1,r2 grids for our mapping
r_c = R_c0 / (D_l * thetaE0)
epsilon0 = 0.0
gamma1_0 = 0.0
gamma2_0 = 0.0

# treat thetaE simply as a free normalized parameter in lens_map 
# For UI convenience. expose a "thetaE_scale" slider (scale multiplier)
thetaE_scale0 = 1.0

# initial render
source_current = source_random.copy()
thetaE_norm = thetaE0 * thetaE_scale0  # normalized Einstein radius (we keep the same units)
lensed_img, mu_map, detA_map, beta_x, beta_y, ax, ay = render_lensed_image(
    source_current, thetaE_norm, r_c, epsilon0, gamma1=gamma1_0, gamma2=gamma2_0
)

# Create figure with subplots:
fig = plt.figure(figsize=(14,7))
ax_img = fig.add_subplot(2,3,1)
ax_mu = fig.add_subplot(2,3,2)
ax_det = fig.add_subplot(2,3,3)
ax_source = fig.add_subplot(2,3,4)
ax_quiver = fig.add_subplot(2,3,5)
ax_caust = fig.add_subplot(2,3,6)

im_im = ax_img.imshow(lensed_img, origin='lower', extent=[0, N, 0, N])
ax_img.set_title("Lensed image (RGB)")
ax_img.set_xticks([]); ax_img.set_yticks([])

im_mu = ax_mu.imshow(np.log10(np.abs(mu_map)+1e-6), origin='lower', cmap='RdBu', extent=[0,N,0,N])
ax_mu.set_title("log10 |magnification|")
ax_mu.set_xticks([]); ax_mu.set_yticks([])

im_det = ax_det.imshow(detA_map, origin='lower', cmap='seismic', extent=[0,N,0,N])
ax_det.set_title("det(A)")
ax_det.set_xticks([]); ax_det.set_yticks([])

im_src = ax_source.imshow(source_current, origin='lower', extent=[0,N,0,N])
ax_source.set_title("Source plane")
ax_source.set_xticks([]); ax_source.set_yticks([])

ax_quiver.set_title("Local distortion (principal stretch direction)")
ax_quiver.set_xlim(0,N); ax_quiver.set_ylim(0,N)
ax_caust.set_title("Caustics (red) over source")
ax_caust.set_xlim(0,N); ax_caust.set_ylim(0,N)

plt.tight_layout()

# Add sliders / buttons below the plots
axcolor = 'lightgoldenrodyellow'
ax_sigma = plt.axes([0.15, 0.02, 0.18, 0.03], facecolor=axcolor)
ax_eps   = plt.axes([0.40, 0.02, 0.18, 0.03], facecolor=axcolor)
ax_rc    = plt.axes([0.65, 0.02, 0.18, 0.03], facecolor=axcolor)
ax_g1    = plt.axes([0.15, 0.06, 0.18, 0.03], facecolor=axcolor)
ax_g2    = plt.axes([0.40, 0.06, 0.18, 0.03], facecolor=axcolor)
ax_toggle = plt.axes([0.65, 0.06, 0.10, 0.04])
ax_reset  = plt.axes([0.78, 0.06, 0.10, 0.04])
ax_radio = plt.axes([0.88, 0.02, 0.10, 0.10], facecolor=axcolor)

s_sigma = Slider(ax_sigma, 'sigma', 100.0, 3000.0, valinit=sigma0)
s_eps   = Slider(ax_eps, 'epsilon', -0.6, 0.6, valinit=epsilon0)
s_rc    = Slider(ax_rc, 'R_c ', 0.0, 0.5, valinit=R_c0)
s_g1    = Slider(ax_g1, 'g1', -0.5, 0.5, valinit=gamma1_0)
s_g2    = Slider(ax_g2, 'g2', -0.5, 0.5, valinit=gamma2_0)
b_toggle = Button(ax_toggle, 'Toggle Source')
b_reset  = Button(ax_reset, 'Reset')

radio = RadioButtons(ax_radio, ('random galaxies','chessboard','real jpg'), active=0 if not USE_REAL_IMAGE else 2)

# helper to compute & update all visuals
def update_all(val=None):
    global source_current, IMG_REAL
    sigma = s_sigma.val
    epsilon = s_eps.val
    R_c = s_rc.val
    gamma1 = s_g1.val
    gamma2 = s_g2.val

    thetaE = compute_thetaE_from_sigma(sigma)
    # Compute r_c
    r_c_norm = R_c / (D_l * thetaE + 1e-12)  # avoid division by zero

    # pick source based on radio
    choice = radio.value_selected
    if choice == 'random galaxies':
        source_current = source_random
    elif choice == 'chessboard':
        source_current = source_chess
    else:
        if IMG_REAL is not None:
            source_current = IMG_REAL
        else:
            source_current = source_chess

    # compute lensed image
    thetaE_norm = thetaE  # we pass thetaE(norm) directly; consistent with how r_c_norm was built
    lensed_img, mu_map, detA_map, beta_x, beta_y, ax_field, ay_field = render_lensed_image(
        source_current, thetaE_norm, r_c_norm, epsilon, gamma1=gamma1, gamma2=gamma2
    )

    # update images
    im_im.set_data(lensed_img)
    im_mu.set_data(np.log10(np.abs(mu_map)+1e-6))
    im_det.set_data(detA_map)
    im_src.set_data(source_current)

    # compute principal stretches & directions for quiver plot
    A11, A12, A21, A22, detA, mu = compute_jacobian_and_magnification(ax_field, ay_field, xs)
    s1, s2, ang = shear_and_direction(A11, A12, A21, A22)

    # sample quiver on a coarse grid for readability
    step = max(8, N // 50)
    yi = np.arange(0, N, step)
    xi = np.arange(0, N, step)
    Xi, Yi = np.meshgrid(xi, yi, indexing="xy")
    # angle array indexed by Yi,Xi
    ang_small = ang[Yi, Xi]
    # arrow components proportional to differential (s1-s2)
    mag_small = (s1 - s2)[Yi, Xi]
    U = mag_small * np.cos(ang_small)
    V = mag_small * np.sin(ang_small)

    ax_quiver.clear()
    ax_quiver.imshow(lensed_img, origin='lower', extent=[0,N,0,N])
    ax_quiver.quiver(Xi, Yi, U, V, scale=10, color='white', width=0.004)
    ax_quiver.set_title("Local distortion (white arrows)")
    ax_quiver.set_xticks([]); ax_quiver.set_yticks([])

    # compute caustic: where detA ~ 0 in image plane mapped into source plane
    # choose threshold relative to detA abs median
    thr = np.percentile(np.abs(detA_map), 99) * 0.01  # small threshold
    mask = np.abs(detA_map) < (thr + 1e-10)

    # extract beta positions only where mask true
    bx = beta_x[mask]
    by = beta_y[mask]
    if bx.size == 0:
        caust_pts_i = np.array([])
        caust_pts_j = np.array([])
    else:
        # convert normalized beta positions to pixel coords
        caust_pts_i = norm_to_pixel_coords(bx, N)
        caust_pts_j = norm_to_pixel_coords(by, N)

    ax_caust.clear()
    ax_caust.imshow(source_current, origin='lower', extent=[0,N,0,N])
    if caust_pts_i.size:
        ax_caust.scatter(caust_pts_i, caust_pts_j, s=1, c='red')
    ax_caust.set_title("Caustic points (red) projected onto source")
    ax_caust.set_xticks([]); ax_caust.set_yticks([])

    # refresh colorbars / images
    fig.canvas.draw_idle()

# initial update to set everything up
update_all()

# hook sliders
s_sigma.on_changed(update_all)
s_eps.on_changed(update_all)
s_rc.on_changed(update_all)
s_g1.on_changed(update_all)
s_g2.on_changed(update_all)

def toggle_source(event):
    # cycle radio selection
    options = radio.labels
    # radio has attribute value_selected; but we'll just toggle programmatically
    opts = ['random galaxies','chessboard','real jpg'] if IMG_REAL is not None else ['random galaxies','chessboard']
    current = radio.value_selected
    idx = opts.index(current)
    idx = (idx + 1) % len(opts)
    radio.set_active(idx)
    update_all()

def reset(event):
    s_sigma.reset(); s_eps.reset(); s_rc.reset(); s_g1.reset(); s_g2.reset()
    radio.set_active(0)
    update_all()

b_toggle.on_clicked(toggle_source)
b_reset.on_clicked(reset)

# Also allow radio to update
def on_radio(label):
    update_all()
radio.on_clicked(on_radio)

plt.show()


