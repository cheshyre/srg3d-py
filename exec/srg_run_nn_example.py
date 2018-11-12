"""Run SRG on an NN potential."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from context import srg3d

potential = srg3d.potential
srg = srg3d.srg

a = potential.load(2, 3, 'EM420new', '00001', 50, 'np')
dim = len(a.nodes)

v_mask = np.array([[0 for i in range(dim)] for j in range(dim)])
k_mask = np.array([[1 for _ in range(dim)] for _ in range(dim)])

srg_obj = srg.SRG(a, v_mask, k_mask)

lambdas = [25] + list(range(10, 4, -1)) + list(np.arange(4, 3.1, -0.5)) \
    + list(np.arange(3.0, 2.5, -0.2)) + list(np.arange(2.4, 1.38, -0.05))

for l in lambdas:
    srg_obj.evolve(l, verbose=False, integrator='lsoda', atol=10**(-6),
                   rtol=10**(-6), nsteps=10**(5))
    c = potential.load(2, 3, 'EM420new', '00001', l, 'np')
    b = srg_obj.get_potential()
    if c.dim < b.dim:
        b = b.reduce_dim(c.dim)
        v_mask = np.array([[0 for i in range(c.dim)] for j in range(c.dim)])
        k_mask = np.array([[1 for _ in range(c.dim)] for _ in range(c.dim)])
        srg_obj.replace_potential(b, v_mask, k_mask)
    print('lambda: {}'.format(l))
    print('Same: {}'.format(c == b))
