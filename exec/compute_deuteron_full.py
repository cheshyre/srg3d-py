# pylint: skip-file
"""Compute deuteron lowest energy state in 3s1 3d1 coupled channel (bound)."""
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


def get_coupled_channel_hamiltonian(channels):
    chan_dim = channels[0].dim
    dim = int(sqrt(len(channels)))
    hamiltonian = np.zeros((dim * chan_dim, dim * chan_dim))

    for i in range(dim):
        for j in range(dim):
            chan = channels[i * dim + j]
            if i == j:
                chan_hamiltonian = chan.with_weights() + chan.kinetic_energy()
            else:
                chan_hamiltonian = chan.with_weights()
            r_s = i * chan_dim
            r_e = (i + 1) * chan_dim
            c_s = j * chan_dim
            c_e = (j + 1) * chan_dim
            hamiltonian[r_s:r_e, c_s:c_e] = chan_hamiltonian
    return hamiltonian


# Physical constants
hbarc = 200
nucleon_mass = 940
red_mass = nucleon_mass / 2

# Load unevolve potential
chan_3s1 = potential.load(2, 3, 'EM420new', '10010', 50, 'np')
chan_3s1_3d1 = potential.load(2, 3, 'EM420new', '10210', 50, 'np')
chan_3d1_3s1 = potential.load(2, 3, 'EM420new', '12010', 50, 'np')
chan_3d1 = potential.load(2, 3, 'EM420new', '12210', 50, 'np')

hamiltonian = get_coupled_channel_hamiltonian([chan_3s1, chan_3s1_3d1,
                                               chan_3d1_3s1, chan_3d1])

ev = np.amin(eigh(hamiltonian)[0])
print('Lambda: {}'.format(50))
print(hbarc**2 / (2 * red_mass) * ev)

# Create list of lambdas to which to evolve
lambdas = [25] + list(range(10, 4, -1)) + list(np.arange(4, 3.1, -0.5)) \
    + list(np.arange(3.0, 2.5, -0.2)) + list(np.arange(2.4, 1.38, -0.05))

for l in lambdas:
    # Load reference potential (calculated by different code)
    chan_3s1 = potential.load(2, 3, 'EM420new', '10010', 50, 'np')
    chan_3s1_3d1 = potential.load(2, 3, 'EM420new', '10210', 50, 'np')
    chan_3d1_3s1 = potential.load(2, 3, 'EM420new', '12010', 50, 'np')
    chan_3d1 = potential.load(2, 3, 'EM420new', '12210', 50, 'np')

    hamiltonian = get_coupled_channel_hamiltonian([chan_3s1, chan_3s1_3d1,
                                                   chan_3d1_3s1, chan_3d1])

    # Get lowest eigenvalue
    ev = np.amin(eigh(hamiltonian)[0])
    print('Lambda: {}'.format(l))
    print(hbarc**2 / (2 * red_mass) * ev)
