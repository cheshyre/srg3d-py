from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from math import pi

import numpy as np
import scipy.integrate as integ


class SRG:
    """Interface for the 3-D SRG evolution."""

    def __init__(self, potential, flow_operator_mask_v, flow_operator_mask_k):
        """Initialize SRG evolution object.

        Parameters
        ----------
        potential : srg3d.potential.Potential
            Potential object to be evolved. Comes with weights, lam, and
            kinetic energy, which is everything you need to do the SRG
            evolution.
        flow_operator_mask_v : matrix of floats
            Matrix which masks parts of the potential when computing the
            flow operator.
        flow_operator_mask_k : matrix of floats
            Matrix which masks parts of the kinetic energy when computing the
            flow operator.

        """
        self._potential = potential
        self._v = potential.without_weights()
        self._k = potential.kinetic_energy()
        self._lam = potential.lam
        self._flow_op_mask_v = flow_operator_mask_v
        self._flow_op_mask_k = flow_operator_mask_k
        self._flow = 'lambda'

    def evolve(self, lam, verbose=False, integrator='dopri5',
               **integrator_params):
        """Evolve potential to specified lam.

        Evolve the SRG object (more specifically the potential) to the
        specified value of lambda according to the SRG differential equation.

        Parameters
        ----------
        lam : float
            Value of lambda to which the potential should be evolved.
        verbose : bool, optional
            Flag which, if set, will print out the current lam of the evolution
        integrator : string, optional
            Name of integrator used.
        integrator_params :
            Additional parameters for the integrator. If none are specified,
            a default set of parameters will be used. See scipy.integrate.ode
            for details.

        See also
        --------
        scipy.integrate.ode : User-friendly interface to various numerical
            integrators. Options for integrator_params can be found here.

        Notes
        -----
        For numerical reasons, we use lambda = 1/s^(1/4) as our evolution
        parameter, as the differential equation with respect to s is rather
        stiff. Thus, the unevolved potential should be at lambda=Infinity,
        corresponding to s=0. This can be reasonably approximated by
        lambda=50.0. Evolving to lambda larger than this is pointless, but
        possible.

        """
        solver = integ.ode(_srg_rhs)

        # Default parameters for integrator chosen based optimal results while
        # testing
        if not integrator_params:
            solver.set_integrator(integrator, atol=10**(-12), rtol=10**(-12),
                                  nsteps=10**(9))
        else:
            solver.set_integrator(integrator, **integrator_params)
        solver.set_f_params(self._k, self._flow_op_mask_v,
                            self._potential.weights, self._flow, verbose)

        solver.set_initial_value(_flatten(self._v), self._lam)
        solver.integrate(lam)
        if solver.successful():
            self._v = _unflatten(solver.y)
        else:
            raise Exception('Integration failed.')

        # If successful, update value of lambda
        self._lam = lam

        return self


# ---------------------------- Internal methods ---------------------------- #


def _srg_rhs(s, potential, kinetic, potential_weight, weights, flow,
             verbose):
    T = kinetic
    V = _unflatten(potential)
    TT = np.dot(T, T)
    X = np.multiply(V, potential_weight)
    Vdiff = V - X
    VT = np.dot(V, T)
    TV = np.dot(T, V)
    XT = np.dot(X, T)
    TX = np.dot(T, X)
    W = np.diag(weights)

    rhs = -1 * (_mm(Vdiff, TT) + _mm(TT, Vdiff)) + 2 * _mmm(T, Vdiff, T) \
        + 2 / pi * (_mmm(TV, W, TV) + _mmm(VT, W, VT) + _mmm(VT, W, XT)
                    + _mmm(TV, W, XT) + _mmm(XT, W, VT) + _mmm(XT, W, TV)
                    - 2 * (_mmm(VT, W, TV) + _mmm(VT, W, TX)
                           + _mmm(TX, W, TV))) \
        + 4 / pi**2 * (_mmmmm(VT, W, VT, W, X) + _mmmmm(XT, W, VT, W, V)
                       - 2 * _mmmmm(VT, W, X, W, TV))

    if verbose:
        print(s)

    # Use commutator defined below
    if flow == 'lambda':
        factor = (-4.0/(s**5))
        rhs *= factor
    return _flatten(rhs)


def _mm(matrix1, matrix2):
    return np.dot(matrix1, matrix2)


def _mmm(matrix1, matrix2, matrix3):
    return np.dot(matrix1, np.dot(matrix2, matrix3))


def _mmmmm(matrix1, matrix2, matrix3, matrix4, matrix5):
    return _mm(_mm(matrix1, matrix2), _mmm(matrix3, matrix4, matrix5))


def _commutator(a, b):
    c = np.dot(a, b) - np.dot(b, a)
    return c


def _flatten(m):
    flattened = np.reshape(m, m.size)
    return flattened


def _unflatten(m):
    unflattened = np.reshape(m, (int(m.size**0.5), int(m.size**0.5)))
    return unflattened
