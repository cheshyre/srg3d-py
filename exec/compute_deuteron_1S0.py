# pylint: skip-file
"""Compute deuteron lowest energy state in 1s0 channel (unbound)."""
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
hbarc = 200
nucleon_mass = 940
red_mass = nucleon_mass / 2

# Load unevolve potential
a = potential.load(2, 3, 'EM420new', '00001', 50, 'np')

# Create list of lambdas to which to evolve
lambdas = [25] + list(range(10, 4, -1)) + list(np.arange(4, 3.1, -0.5)) \
    + list(np.arange(3.0, 2.5, -0.2)) + list(np.arange(2.4, 1.38, -0.05))

for l in lambdas:
    # Load reference potential (calculated by different code)
    b = potential.load(2, 3, 'EM420new', '00001', l, 'np')

    # Compute Hamiltonian
    potential_energy = b.with_weights()
    kinetic_energy = b.kinetic_energy()

    hamiltonian = potential_energy + kinetic_energy

    # Get lowest eigenvalue
    ev = np.amin(eigh(hamiltonian)[0])
    print('Lambda: {}'.format(l))
    print(hbarc**2 / (2 * red_mass) * ev)
