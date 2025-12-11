import pickle
from pathlib import Path

import numpy as np

import ase.io
from finite_field_ters import FiniteFieldTERS

# masses
g = ase.io.read('geometry.in')
masses = g.get_masses()

# read in test hessian
hessian = pickle.load(open('hessian.pickle', 'rb'))

# set up initial TERS object, fill with required data
ters = FiniteFieldTERS(
hessian = hessian,
modes = None,
masses = masses,
dq = 5e-3,
efield = -1e-1,
submit_style = 'slurm',
fn_control_template = Path('template.in'),
aims_dir = Path('/u/brek/build/FHIaims/'),
fn_batch = Path('run.sbatch'),
fn_elsi_restart = None,
fn_tip_derivative = Path('tipA_05_vh_ft_0049_3221meV_x1000.cube'),
fn_tip_groundstate = None,
fn_geometry = Path('geometry.in'),
)


# test 2D infrastructure
ters.run_2d_grid(
idx_mode = 14,
tip_origin = (-0.000030, -1.696604, -4.6140),
sys_origin = (0.0, 0.0, 0.0),
tip_height = 4.0,
scan_range = (-5.0, 5.0, -5.0, 5.0),
bins = (10, 10)
)
