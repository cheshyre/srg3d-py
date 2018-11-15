# pylint: skip-file
"""Run SRG on an NN potential."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from numpy.linalg import eigh

# Check if module is installed, otherwise load locally
try:
    import srg3d.potential as potential
    import srg3d.srg as srg
except ImportError:
    from context import srg3d
    potential = srg3d.potential
    srg = srg3d.srg

# Physical constants
hbarc = 197.327
proton_mass = 938.272
neutron_mass = 939.565
red_mass = proton_mass * neutron_mass / (proton_mass + neutron_mass)

# Load unevolve potential
a = potential.load(2, 3, 'EM420new', '00001', 50, 'np')
dim = len(a.nodes)

# Set up T_rel flow operator
v_mask = np.array([[0 for _ in range(dim)] for _ in range(dim)])
k_mask = np.array([[1 for _ in range(dim)] for _ in range(dim)])

# Set up SRG
srg_obj = srg.SRG(a, v_mask, k_mask)

# Create list of lambdas to which to evolve
lambdas = [25] + list(range(10, 4, -1)) + list(np.arange(4, 3.1, -0.5)) \
    + list(np.arange(3.0, 2.5, -0.2)) + list(np.arange(2.4, 1.38, -0.05))

for l in lambdas:
    # Evolve to lambda
    srg_obj.evolve(l, verbose=False, integrator='dopri5', atol=10**(-6),
                   rtol=10**(-6), nsteps=10**(5))

    # Load reference potential (calculated by different code)
    b = potential.load(2, 3, 'EM420new', '00001', l, 'np')

    # Extract evolved potential
    a = srg_obj.get_potential()

    # Compute Hamiltonians
    hamiltonian_ref = b.with_weights() + b.kinetic_energy()
    hamiltonian = a.with_weights() + a.kinetic_energy()

    # Get lowest eigenvalues
    ev_ref = np.amin(eigh(hamiltonian_ref)[0])
    ev = np.amin(eigh(hamiltonian)[0])

    # Output values
    print('Lambda: {}'.format(l))
    print('E_ref = {} MeV'.format(hbarc**2 / (2 * red_mass) * ev_ref))
    print('E_srg = {} MeV\n'.format(hbarc**2 / (2 * red_mass) * ev))
