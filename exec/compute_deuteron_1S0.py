# pylint: skip-file
"""Compute deuteron lowest energy state in 1s0 channel (unbound).

Computes lowest energy eigenvalue for a set of reference SRG evolved potentials
as well as the manually SRG evolved potential to show the implementation is
numerically equivalent to the reference implementation.

"""
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

# Load unevolved potential
a = potential.load(2, 3, 'EM420new', '00001', 50, 'np')

# Compute Hamiltonian and eigenvalue
potential_energy = a.with_weights()
kinetic_energy = a.kinetic_energy()

hamiltonian = potential_energy + kinetic_energy

ev = np.amin(eigh(hamiltonian)[0])

# Output unevolved lowest energy
print('Unevolved')
print('E_ref = E_srg = {} MeV\n'.format(hbarc**2 / (2 * red_mass) * ev))

# Set up SRG evolution
v_mask = np.array([[0 for _ in range(len(a.nodes))] for _ in
                   range(len(a.nodes))])
k_mask = np.array([[1 for _ in range(len(a.nodes))] for _ in
                   range(len(a.nodes))])

srg_obj = srg.SRG(a, v_mask, k_mask)

# Create list of lambdas to which to evolve
lambdas = [25] + list(range(10, 4, -1)) + list(np.arange(4, 3.1, -0.5)) \
    + list(np.arange(3.0, 2.5, -0.2)) + list(np.arange(2.4, 1.38, -0.05))
num_pts = [100]*9 + [83] + [79] + [76] + [73] + [72]*2 + [71] + [70]*2 + [69] \
    + [68]*2 + [67]*2 + [66] + [65]*2 + [64]*2 + [63] + [62]*2 + [61]*2

for l, n in zip(lambdas, num_pts):
    # Evolve potential
    srg_obj.evolve(l, verbose=False, integrator='lsoda', atol=10**(-6),
                   rtol=10**(-6), nsteps=10**(5))

    # Extract evolved potential
    a = srg_obj.get_potential()

    # Reduce dimension of potential if necessary
    if n < a.dim:
        a = a.reduce_dim(n)
        v_mask = np.array([[0 for _ in range(len(a.nodes))] for _ in
                           range(len(a.nodes))])
        k_mask = np.array([[1 for _ in range(len(a.nodes))] for _ in
                           range(len(a.nodes))])
        srg_obj.replace_potential(a, v_mask, k_mask)

    # Load reference potential (calculated by different code)
    b = potential.load(2, 3, 'EM420new', '00001', l, 'np')

    # Compute Hamiltonians
    potential_energy = b.with_weights()
    kinetic_energy = b.kinetic_energy()

    hamiltonian_ref = potential_energy + kinetic_energy

    potential_energy = a.with_weights()
    kinetic_energy = a.kinetic_energy()

    hamiltonian = potential_energy + kinetic_energy

    # Get lowest eigenvalues
    ev_ref = np.amin(eigh(hamiltonian_ref)[0])
    ev = np.amin(eigh(hamiltonian)[0])

    # Output values
    print('Lambda: {}'.format(l))
    print('E_ref = {} MeV'.format(hbarc**2 / (2 * red_mass) * ev_ref))
    print('E_srg = {} MeV\n'.format(hbarc**2 / (2 * red_mass) * ev))
