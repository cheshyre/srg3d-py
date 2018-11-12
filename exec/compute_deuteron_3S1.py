# pylint: skip-file
"""Compute deuteron lowest energy state in 1s0 channel (unbound)."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from math import sqrt

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


def add_weights(potential_energy, weights):
    weights_matrix = np.diag([w**(0.5) for w in weights])
    return np.dot(weights_matrix, np.dot(potential_energy, weights_matrix))


# Load unevolve potential
a = potential.load(2, 3, 'EM420new', '00001', 50, 'np')
dim = len(a.nodes)

# Set up T_rel flow operator
v_mask = np.array([[0 for i in range(dim)] for j in range(dim)])
k_mask = np.array([[1 for _ in range(dim)] for _ in range(dim)])

# Create list of lambdas to which to evolve
lambdas = [25] + list(range(10, 4, -1)) + list(np.arange(4, 3.1, -0.5)) \
    + list(np.arange(3.0, 2.5, -0.2)) + list(np.arange(2.4, 1.38, -0.05))

for l in lambdas:
    # Load reference potential (calculated by different code)
    b = potential.load(2, 3, 'EM420new', '00001', l, 'np')

    # Compute Hamiltonian
    potential_energy = b.without_weights()
    kinetic_energy = b.kinetic_energy()
    hamiltonian = np.zeros_like(potential_energy)

    nodes = b.nodes
    weights = b.weights
    weights_sqrt = np.array([sqrt(w) for w in weights])
    for i in range(b.dim):
        for j in range(b.dim):
            hamiltonian[i][j] = nodes[i] * weights_sqrt[i] \
                * potential_energy[i][j] * nodes[j] * weights_sqrt[j] \
                + kinetic_energy[i][j]

    # Get lowest eigenvalue
    ev = np.amin(eigh(hamiltonian)[0])
    print('Lambda: {}'.format(l))
    print(200 * 200 / 940 * ev)
