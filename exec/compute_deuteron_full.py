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
hbarc = 197.327
proton_mass = 938.272
neutron_mass = 939.565
red_mass = proton_mass * neutron_mass / (proton_mass + neutron_mass)

# Load unevolve potential
chan_3s1 = potential.load(2, 3, 'EM420new', '10010', 50, 'np')
chan_3s1_3d1 = potential.load(2, 3, 'EM420new', '10210', 50, 'np')
chan_3d1_3s1 = potential.load(2, 3, 'EM420new', '12010', 50, 'np')
chan_3d1 = potential.load(2, 3, 'EM420new', '12210', 50, 'np')

hamiltonian = get_coupled_channel_hamiltonian([chan_3s1, chan_3s1_3d1,
                                               chan_3d1_3s1, chan_3d1])

c_potential = potential.CoupledPotential([chan_3s1, chan_3s1_3d1, chan_3d1_3s1,
                                          chan_3d1])
hamiltonian2 = c_potential.with_weights() + c_potential.kinetic_energy()

ev = np.amin(eigh(hamiltonian)[0])
ev2 = np.amin(eigh(hamiltonian2)[0])
print('Lambda: {}'.format(50))
print(hbarc**2 / (2 * red_mass) * ev)
print(hbarc**2 / (2 * red_mass) * ev2)

v_mask = np.array([[0 for _ in range(len(c_potential.nodes))] for _ in
                   range(len(c_potential.nodes))])
k_mask = np.array([[1 for _ in range(len(c_potential.nodes))] for _ in
                   range(len(c_potential.nodes))])

srg_obj = srg.SRG(c_potential, v_mask, k_mask)

# Create list of lambdas to which to evolve
lambdas = [25] + list(range(10, 4, -1)) + list(np.arange(4, 3.1, -0.5)) \
    + list(np.arange(3.0, 2.5, -0.2)) + list(np.arange(2.4, 1.38, -0.05))
num_pts = [100]*9 + [83] + [79] + [76] + [73] + [72]*2 + [71] + [70]*2 + [69] \
    + [68]*2 + [67]*2 + [66] + [65]*2 + [64]*2 + [63] + [62]*2 + [61]*2

for l, n in zip(lambdas, num_pts):
    srg_obj.evolve(l, verbose=False, integrator='lsoda', atol=10**(-6),
                   rtol=10**(-6), nsteps=10**(5))
    c_potential = srg_obj.get_potential()

    if n < c_potential.dim:
        c_potential = c_potential.reduce_dim(n)
        v_mask = np.array([[0 for _ in range(len(c_potential.nodes))] for _ in
                           range(len(c_potential.nodes))])
        k_mask = np.array([[1 for _ in range(len(c_potential.nodes))] for _ in
                           range(len(c_potential.nodes))])
        srg_obj.replace_potential(c_potential, v_mask, k_mask)
    hamiltonian2 = c_potential.with_weights() + c_potential.kinetic_energy()
    # Load reference potential (calculated by different code)
    chan_3s1 = potential.load(2, 3, 'EM420new', '10010', l, 'np')
    chan_3s1_3d1 = potential.load(2, 3, 'EM420new', '10210', l, 'np')
    chan_3d1_3s1 = potential.load(2, 3, 'EM420new', '12010', l, 'np')
    chan_3d1 = potential.load(2, 3, 'EM420new', '12210', l, 'np')

    hamiltonian = get_coupled_channel_hamiltonian([chan_3s1, chan_3s1_3d1,
                                                   chan_3d1_3s1, chan_3d1])

    # Get lowest eigenvalue
    ev = np.amin(eigh(hamiltonian)[0])
    ev2 = np.amin(eigh(hamiltonian2)[0])
    print('Lambda: {}'.format(l))
    print(hbarc**2 / (2 * red_mass) * ev)
    print(hbarc**2 / (2 * red_mass) * ev2)
