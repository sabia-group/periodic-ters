import pickle
from pathlib import Path

import numpy as np

from finite_field_ters import FiniteFieldTERS

# masses
masses = np.array([24.305] + [14.007] * 4 + [12.011] * 20 + [1.006] * 12)

# read in test hessian
hessian = pickle.load(open('hessian.pickle', 'rb'))

# set up initial TERS object, fill with required data
ters = FiniteFieldTERS(
hessian = hessian,
masses = masses,
dq = 5e-3,
efield = -1e-2,
submit_style = 'slurm',
fn_control_template = Path('template.in'),
aims_dir = Path('/u/brek/build/FHIaims_periodic_ters/'),
fn_batch = Path('run.sbatch'),
fn_tip = Path('tipA_05_vh_ft_0049_3221meV_x1000.cube'),
#fn_elsi_restart = Path('D_spin_01_kpt_000001.csc'),
fn_elsi_restart = None,
fn_geometry = Path('geometry.in'),
)


# test 2D infrastructure
ters.run_2d_grid(
idx_mode = 110,
tip_origin = (-0.000030, -1.696604, -4.6140),
sys_origin = (0.0, 0.0, 0.0),
tip_height = 4.0,
scan_range = (-6.5, 6.5, -6.5, 6.5),
bins=5
)
