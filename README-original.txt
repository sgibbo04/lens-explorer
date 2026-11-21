README.txt 

This package contains Python scripts for simulating gravitational lensing. 
The scripts demonstrate lensing of point sources, structured patterns, random galaxies, JPEG images, and include interactive visualization and distortion analysis.



GravProject: Original test case, basic parameter changes and pixel mapping.

Script A: Basic lens mapping; allows changing lens parameters to explore lensed images.

Script B: Magnification map; shows areas of high/low magnification and caustics.

Script C: Random galaxy field; demonstrates lensing on multiple extended sources.

Script D: Lensing applied to a JPEG image; visualizes distortion of complex sources.

Script E: Chessboard source with distortion field; shows vector arrows of local displacements.

Script F: Interactive GUI; allows real-time adjustment of lens parameters and source type.

Lensexplorer.py: Combines all features, including source selection and real-time lensing.

The only non-standard library used is Pillow for JPEG images. 

Requirements

Python 3.8+

Packages: numpy, matplotlib, optionally Pillow for JPEG images.

Source JPEG for the code is found in this folder as well "source.jpg"