# Periodic TERS

This repository contains code and data related to tip-enhanced Raman spectra simulations using embedding near fields in periodic boundary conditions in [FHI-aims](https://fhi-aims.org).
The original numerical near field framework was introduced in 

Y. Litman, F. P. Bonafe, A. Akkoush, H. Appel and M. Rossi: First-Principles Simulations of Tip Enhanced Raman Scattering Reveal Active Role of Substrate on High-Resolution Images. *J. Phys. Chem. Lett.* 14(30), **2023**.

The finite-field periodic framework is introduced in

K. Brezina, Y. Litman and M. Rossi: Explaining Principles of Tip-Enhanced Raman Images from Fully Ab Initio Modeling. arXiv:2509.13075, **2025**.

In order to use the module, please first run 

```
source ./python/env.sh 
```

The directory `python` contains the module itself. This can prepare the directory tree and all the FHI-aims related input for running the TERS calculations either as a 1D spectrum (full frequency range for a fixed tip position) 
or as a 2D image for a given mode (single frequency and a grid of tip positions within a selected scan range).
The directory `examples` contains a comprehensive collection of code and data to run an analyze these simulations.
Each run mode contains a `data` directory, which represents a realistic calculation directory with the output already provided. 
Feel free to prepare or run the calculations using

```
python run-ters.py
```

As of now, the module is capable of submitting the individual single points as SLURM jobs automatically as demonstrated the examples and docstrings.
In addition, we have provided the cube files with two different tip geometries ("tip A" and "tipB') which are discussed and used in both above publications.
The examples also contain Jupyter notebooks with a workflow to analyze the output of the calculations.

> Note: due to a historical convention of flipped direction of E-fields in FHI-aims (well documented in the FHI-aims manual), the `finite_field_ters` module expects electric fields to be given as a negative of what is wanted.
> For example, to employ a field of 0.1 V/Angstrom in the *z*-direction, one should write -0.1 in the input (as shown in the examples).
