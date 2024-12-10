import pickle
from pathlib import Path

import numpy as np

from finite_field_ters import FiniteFieldTERS
from toolbox import normal_mode_sampling as nms

# masses
masses = np.array([24.305] + [14.007] * 4 + [12.011] * 20 + [1.006] * 12)
masses_free = masses[:]

# read in test hessian
hessian = pickle.load(open('hessian.pickle', 'rb'))
m = nms.NormalModes.from_ase(hessian, masses_free)
h = m.hessian

# here, we have a fixed layer in the surface and need to pad the hessian with zeros
# TODO: this is also super hacky, make it more robust.
# also its probably more robust to pad modes with zero that the hessian, because of negative R+T modes,
# so this is just a placeholder. Let's discuss how to do this.
if(len(masses) * 3) > hessian.shape[0]:
    print('Hessian will be padded by zeros for fixed DOFs.')
    nfixed = 256
    ndof_fixed = 3 * nfixed
    full_dim = ndof_fixed + hessian.shape[0]
    full_h = np.zeros(shape=(full_dim, full_dim))
    full_h[ndof_fixed:, ndof_fixed:] = h.copy()
    fix_for_idx_mode = ndof_fixed
else:
    full_h = h.copy()
    fix_for_idx_mode = 0

print(f'Adding {fix_for_idx_mode:d} DOFs to the selected mode index.')

# set up initial TERS object, fill with required data
ters = FiniteFieldTERS(
hessian = full_h,
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
idx_mode = 110 + fix_for_idx_mode,
tip_origin = (-0.000030, -1.696604, -4.6140),
tip_height = 4.0,
scan_range = (-6.5, 6.5, -6.5, 6.5),
bins=5
)
