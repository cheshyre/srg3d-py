"""Run SRG on an NN potential."""
import srg3d.potential as potential
import srg3d.srg as srg
import numpy as np


def _mask(i, j):
    if i < 20 and j < 20:
        return 1
    return 0


a = potential.load('VNN_N3LO_EM420new_SLLJT_00001_lambda_50.00_Np_100_np.dat')
dim = len(a.nodes)

v_mask = np.array([[0 for i in range(dim)] for j in range(dim)])
#  v_mask = np.array([[_mask(i, j) for i in range(dim)] for j in range(dim)])
k_mask = np.array([[1 for _ in range(dim)] for _ in range(dim)])

#  potential.plot(a, -2, 2)
#  potential.plot(c, -2, 2)

srg_obj = srg.SRG(a, v_mask, k_mask)

lambdas = [25] + list(range(10, 4, -1)) + list(np.arange(4, 3.1, -0.5)) \
    + list(np.arange(3.0, 2.5, -0.2)) + list(np.arange(2.4, 1.38, -0.05))

num_pts = [100]*9 + [83] + [79] + [76] + [73] + [72]*2 + [71] + [70]*2 + [69] \
    + [68]*2 + [67]*2 + [66] + [65]*2 + [64]*2 + [63] + [62]*2 + [61]*2

#  print(lambdas)
#  print(num_pts)
#  print(len(lambdas))
#  print(len(num_pts))
#  for l, n in zip(lambdas, num_pts):
#      print('{} {}'.format(l, n))

for l, n in zip(lambdas, num_pts):
    srg_obj.evolve(l, verbose=False, integrator='lsoda', atol=10**(-6),
                   rtol=10**(-6), nsteps=10**(5))
    b = srg_obj.get_potential()
    if n < b.dim:
        b = b.reduce_dim(n)
        v_mask = np.array([[0 for i in range(n)] for j in range(n)])
        k_mask = np.array([[1 for _ in range(n)] for _ in range(n)])
        srg_obj.replace_potential(b, v_mask, k_mask)
    c = potential.load('VNN_N3LO_EM420new_SLLJT_00001_lambda_{:.2f}_Np_{}_np.dat'.format(float(l), n))
    #  potential.plot(b, -2, 2)
    #  if n < 100:
    #      diff = c.without_weights() - b.without_weights()[np.ix_(list(range(n)),
    #                                                              list(range(n)))]
    #  else:
    #      diff = c.without_weights() - b.without_weights()
    print(l)
    print(c == b)
    #  print(diff.max())
    #  print(diff.min())

#  potential.plot(b, -2, 2)
