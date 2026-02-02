import pickle
from pathlib import Path

import numpy as np

import ase.io
from finite_field_ters import FiniteFieldTERS

# masses
g = ase.io.read('geometry.in')
masses = g.get_masses()

# load expanded mode
mode = pickle.load(open('mode.pickle', 'rb'))
mode = mode[None, :]

# set up initial TERS object, fill with required data
ters = FiniteFieldTERS(
hessian = None,
modes = mode,
masses = masses,
dq = 5e-3,
efield = -1e-1,
submit_style = 'slurm',
fn_control_template = Path('template.in'),
aims_dir = Path('/u/brek/build/FHIaims/'),
fn_batch = Path('run.sbatch'),
fn_tip_derivative = Path('tipA_05_vh_ft_0049_3221meV_x1000.cube'),
fn_tip_groundstate = None,
fn_elsi_restart = None,
fn_geometry = Path('geometry.in'),
)


# test 2D infrastructure
ters.run_2d_grid(
idx_mode = 0, # prodiving a single mode here, so idx=0
tip_origin = (-0.000030, -1.696604, -4.6140),
sys_origin = (0.0, 0.0, 0.0), # S-atom in the center of out slab
tip_height = 4.0,
scan_range = (-8.745, 8.745, -7.5734, 7.5734),
bins=11
)
