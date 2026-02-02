import pickle
from pathlib import Path

import numpy as np

import ase.io
import finite_field_ters as ffters

# masses
g = ase.io.read('geometry.in')
masses = g.get_masses()
idx_frozen = np.arange(256) # indices of frozen Ag atoms

# read in hessian and modify to comply with the full dimensionality
hessian = pickle.load(open('hessian.pickle', 'rb'))
h, ifix = ffters.pad_frozen_hessian(hessian, masses, idx_frozen) 

# choose a mode
imode = 103 # A2g
imode += ifix # fix for the dummy degrees of freedom compensating the frozen surface

# set up initial TERS object, fill with required data
ters = ffters.FiniteFieldTERS(
hessian = h,
modes = None,
masses = masses,
dq = 5e-3,
efield = -1e-1,
submit_style = 'draft',
fn_control_template = Path('template.in'),
aims_dir = Path('/u/brek/build/FHIaims/'),
fn_batch = Path('run.sbatch'),
fn_tip_groundstate = None,
fn_tip_derivative = Path('tipA.cube'),
fn_elsi_restart = None,
fn_geometry = Path('geometry.in'),
)

# test 2D infrastructure
ters.run_2d_grid(
idx_mode = imode,
tip_origin = (-0.000030, -1.696604, -4.6140),
sys_origin = (0.0, 0.0, 0.0),
tip_height = 4.0,
scan_range = (-6.5, 6.5, -6.5, 6.5),
bins=(12, 12)
)
